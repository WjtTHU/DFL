import numpy as np
import polars as pl
from loguru import logger
from scipy.optimize import minimize

from wip_analysis.utils.env_num import env_num
from wip_analysis.utils.logging import format_polars_df

_TOTAL_DURATION_PCT = 100.0
_REGULARIZATION_FACTOR = 1e-3
_SLSQP_DEFAULT_MAXITER = 2000


def _solve_constrained_weighted_ls(
    A: np.ndarray,
    b: np.ndarray,
    weights: np.ndarray,
    min_stage_pct: float,
) -> np.ndarray:
    num_stages = A.shape[1]
    x0 = np.full(num_stages, _TOTAL_DURATION_PCT / num_stages)
    weighted_A = A * weights[:, np.newaxis]
    quadratic = A.T @ weighted_A + _REGULARIZATION_FACTOR * np.identity(num_stages)
    linear = A.T @ (weights * b)

    def objective(x: np.ndarray) -> float:
        return float(x @ quadratic @ x - 2 * linear @ x)

    def jacobian(x: np.ndarray) -> np.ndarray:
        return 2 * (quadratic @ x - linear)

    result = minimize(
        objective,
        x0,
        jac=jacobian,
        bounds=[(min_stage_pct, None)] * num_stages,
        constraints=(
            {
                "type": "eq",
                "fun": lambda x: float(x.sum() - _TOTAL_DURATION_PCT),
                "jac": lambda x: np.ones_like(x),
            },
        ),
        method="SLSQP",
        options={
            "maxiter": env_num("WIP_ANALYSIS_STAGE_DUR_SOLVER_MAXITER", _SLSQP_DEFAULT_MAXITER, 1)
        },
    )
    if not result.success:
        logger.warning("Solution did not converge, Falling back to uniform stage durations")
        return x0

    return np.asarray(result.x, dtype=float)


def solve_stage_durations(
    df: pl.DataFrame,
    stg_to_idx: dict[str, int],
    min_dur_prior: float | None = None,
    prior_weight: float = 20.086,
) -> pl.DataFrame:
    df_processed = (
        df.with_columns(pl.col("stage").replace_strict(stg_to_idx, default=-1).alias("stage_idx"))
        .filter(pl.col("stage_idx") >= 0)  # filter: stages not in primary path
        .sort("lot_id", "created_at")
        .with_columns(
            (
                (pl.col("stage_idx") != pl.col("stage_idx").shift(1).over("lot_id"))
                | (pl.col("stage_idx").shift(1).over("lot_id").is_null())
            ).alias("stage_change")
        )
        .with_columns(pl.col("stage_change").cum_sum().over("lot_id").alias("stage_block"))
        .group_by("lot_id", "stage_block")
        .agg(
            pl.col("stage_idx").first().alias("stage_idx"),
            pl.col("created_at").min().alias("start_of_stage"),
        )
        .sort("lot_id", "start_of_stage")
        .with_columns(
            pl.col("stage_idx").shift(-1).over("lot_id").alias("next_stage_idx"),
            pl.col("start_of_stage").shift(-1).over("lot_id").alias("next_stage_start"),
        )
        .drop_nulls()
        .filter(pl.col("next_stage_idx") > pl.col("stage_idx"))  # filter: inversed stage order
        .with_columns(
            (pl.col("next_stage_start") - pl.col("start_of_stage"))
            .dt.total_seconds()
            .alias("time_delta_sec")
        )
    )

    logger.opt(lazy=True).trace(
        "Processed DataFrame: \n{}",
        lambda: format_polars_df(df_processed),
    )

    # Use the processing speed as weight of the linear equations
    lot_summary = (
        df_processed.group_by("lot_id")
        .agg(
            pl.col("start_of_stage").min().alias("min_time"),
            pl.col("start_of_stage").max().alias("max_time"),
            # pl.col("next_stage_start").max().alias("max_time"),
        )
        .with_columns(
            (pl.col("max_time") - pl.col("min_time")).dt.total_seconds().alias("duration_sec")
        )
        .filter(pl.col("duration_sec") > 0)
        .with_columns((1 / pl.col("duration_sec")).alias("speed"))
    )
    mean_speed = lot_summary.select(pl.col("speed").mean()).item()
    lot_summary = lot_summary.with_columns((pl.col("speed") / mean_speed).sqrt().alias("weight"))

    logger.opt(lazy=True).trace("Lot summary: \n{}", lambda: format_polars_df(lot_summary))

    # Data for solving
    df_with_weight = df_processed.join(
        lot_summary.select(["lot_id", "weight", "duration_sec"]), on="lot_id"
    )
    num_equations = df_with_weight.height
    num_stages = len(stg_to_idx)
    if num_equations == 0:
        # No usable stage-to-stage transition equations: the snapshot window collapses to a
        # single instant (every lot observed at one created_at), so no duration is solvable.
        # This is a recoverable data condition, NOT an exception -- use warning() so we don't
        # emit a misleading "NoneType: None" traceback (logger.exception with no active error)
        # and don't raise the line to ERROR. Caller gets a neutral uniform fallback; callers
        # that persist results skip caching this degenerate window (see _is_degenerate_path_window).
        logger.warning(
            "No valid equations for stage-duration solve "
            f"({num_stages} stages, single-instant snapshot window); returning uniform fallback."
        )
        uniform_duration = _TOTAL_DURATION_PCT / num_stages if num_stages > 0 else 0.0
        result_df = pl.DataFrame(
            {
                "stage": stg_to_idx.keys(),
                "duration": [uniform_duration] * num_stages,
            }
        )
        return result_df.with_columns(pl.col("duration").alias("duration_ratio"))

    A = np.zeros((num_equations, num_stages))
    b = np.zeros(num_equations)
    weights = np.zeros(num_equations)

    # iter over equations
    for i, record in enumerate(df_with_weight.iter_rows(named=True)):
        start_idx = record["stage_idx"]
        end_idx = record["next_stage_idx"]

        A[i, start_idx:end_idx] = 1
        b[i] = 100 * record["time_delta_sec"] / record["duration_sec"]
        weights[i] = record["weight"]

    min_stage_pct = min_dur_prior or env_num("WIP_ANALYSIS_MIN_STAGE_PCT", -1.0, min_value=-1.0)
    if min_stage_pct < 0:
        min_stage_pct = 10.0 / num_stages
    min_stage_pct = min(min_stage_pct, _TOTAL_DURATION_PCT / num_stages)

    # Add prior equations to regularize the solution. With the sum-to-100 constraint,
    # equal prior targets still let ambiguous same-coverage intervals split evenly.
    A_prior = np.identity(num_stages)
    b_prior = np.full(num_stages, min_stage_pct)
    avg_data_weight = weights.mean() if num_equations > 0 else 1.0
    weights_prior = np.full(num_stages, prior_weight * avg_data_weight)

    A_full = np.vstack([A, A_prior])
    b_full = np.concatenate([b, b_prior])
    weights_full = np.concatenate([weights, weights_prior])

    x = _solve_constrained_weighted_ls(
        A_full,
        b_full,
        weights_full,
        min_stage_pct,
    )

    result_df = pl.DataFrame({"stage": stg_to_idx.keys(), "duration": x})
    total = result_df["duration"].sum()
    result_df = result_df.with_columns((100 * pl.col("duration") / total).alias("duration_ratio"))

    logger.opt(lazy=True).debug(
        "Primary path intervals: \n{}",
        lambda: format_polars_df(result_df),
    )

    return result_df
