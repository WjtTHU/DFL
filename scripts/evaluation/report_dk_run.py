"""Report a completed DK run using the enterprise datasets and saved checkpoint.

Writes a local HTML experiment report, figures, metrics and per-window predictions.
Does not train, edit source CSVs, change enterprise configs, or publish any output.
"""

import argparse
import base64
import gc
import hashlib
import html
import json
import re
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from local_paths import configure_imports, dataset_paths, project_path


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_run_sources(run_dir):
    """Do not overwrite a historical evaluation using a different training pipeline."""
    records = json.loads((run_dir / "source_hashes.json").read_text(encoding="utf-8-sig"))
    changed = []
    for record in records:
        relative = record["path"].replace("\\", "/")
        if relative.startswith(("src/wip_pred/", "scripts/training/")):
            path = PROJECT_ROOT / relative
            if not path.is_file() or sha256(path) != record["sha256"]:
                changed.append(relative)
    if changed:
        raise RuntimeError(
            "Training source differs from this run; use the recorded source version "
            "to reproduce it, and leave existing reports intact: " + ", ".join(changed)
        )


def compare_baseline(run_dir, baseline_dir, frame, summary):
    import numpy as np
    import pandas as pd

    previous = json.loads((baseline_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
    old = pd.read_csv(baseline_dir / "validation_predictions.csv")
    same_windows = (
        old.lot_id.tolist() == frame.lot_id.tolist()
        and np.allclose(old.actual_remaining_days, frame.actual_remaining_days, atol=1e-7)
        and np.array_equal(pd.to_datetime(old.reference_snapshot_time).to_numpy(),
                           pd.to_datetime(frame.reference_snapshot_time).to_numpy())
    )
    same_configs = all(sha256(run_dir / name) == sha256(baseline_dir / name)
                       for name in ("source_core.yaml", "source_train.yaml"))
    if not (same_windows and same_configs and previous["data"]["sha256"] == summary["data"]["sha256"]):
        raise RuntimeError("Baseline comparison requires identical data, configurations and validation windows")
    before = previous["model_all_windows"]
    after = summary["model_all_windows"]
    return {
        "baseline_run": str(baseline_dir), "identical_validation_windows": True,
        "identical_source_configs": True, "identical_csv_sha256": True,
        "before": before, "after": after,
        "mae_change_days": after["mae_days"] - before["mae_days"],
        "mae_reduction_fraction": 1 - after["mae_days"] / before["mae_days"],
        "before_best_epoch": previous["best_epoch_from_rounded_log"],
        "after_best_epoch": summary["best_epoch_from_rounded_log"],
    }


def metrics(actual, predicted):
    import numpy as np

    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if not len(actual):
        return {"n": 0}
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError("Non-finite targets or predictions")
    error = predicted - actual
    ae = np.abs(error)
    denominator = np.square(actual - actual.mean()).sum()
    return {
        "n": len(actual),
        "mae_days": float(ae.mean()),
        "rmse_days": float(np.sqrt(np.square(error).mean())),
        "median_ae_days": float(np.median(ae)),
        "p90_ae_days": float(np.quantile(ae, 0.9)),
        "bias_days": float(error.mean()),
        "r2": float(1 - np.square(error).sum() / denominator) if denominator > 0 else None,
        "within_1_day": float((ae <= 1).mean()),
        "within_3_days": float((ae <= 3).mean()),
        "within_7_days": float((ae <= 7).mean()),
        "negative_predictions": int((predicted < 0).sum()),
    }


def read_history(path):
    import pandas as pd

    pattern = re.compile(
        r"^(\d{4}-\d{2}-\d{2} [\d:,]+).*Epoch (\d+)/(\d+) - "
        r"Train Loss: ([\d.eE+-]+) \| Val Loss: ([\d.eE+-]+)"
        r" \| Train MAE: ([\d.eE+-]+) \| Val MAE: ([\d.eE+-]+)", re.M
    )
    rows = [dict(timestamp=t, epoch=int(e), configured_epochs=int(n),
                 train_loss=float(tl), val_loss=float(vl),
                 train_mae_days=float(tm), val_mae_days=float(vm))
            for t, e, n, tl, vl, tm, vm in pattern.findall(path.read_text(encoding="utf-8"))]
    if not rows:
        raise ValueError(f"No completed epochs in {path}")
    return pd.DataFrame(rows)


def metric_table(named_metrics):
    rows = []
    for name, values in named_metrics.items():
        if not values["n"]:
            continue
        rows.append([name, values["n"], f'{values["mae_days"]:.3f}',
                     f'{values["rmse_days"]:.3f}', f'{values["median_ae_days"]:.3f}',
                     f'{values["p90_ae_days"]:.3f}', f'{values["within_3_days"]:.1%}'])
    return table(["对象", "窗口数", "MAE / 天", "RMSE / 天", "中位绝对误差 / 天",
                  "P90 绝对误差 / 天", "误差 ≤ 3 天"], rows)


def table(headers, rows):
    def cells(tag, values):
        return "".join(f"<{tag}>{html.escape(str(value))}</{tag}>" for value in values)
    return "<table><thead><tr>" + cells("th", headers) + "</tr></thead><tbody>" + "".join(
        "<tr>" + cells("td", row) + "</tr>" for row in rows
    ) + "</tbody></table>"


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    run_dir = project_path(args.run_dir)
    checkpoint = project_path(args.checkpoint)
    experiment = project_path(args.experiment)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("training_exit_code") != 0:
        raise RuntimeError("Training must finish successfully before evaluation; no partial checkpoint report")
    verify_run_sources(run_dir)

    configure_imports()
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    from loguru import logger
    from train_DK_csv import load_snapshots
    from wip_pred.inference.model_loader import load_model_and_config
    from wip_pred.prep_data.normalize_data import transform_df
    from wip_pred.prep_data.prepare_data import get_df_of_completed_lots
    from wip_pred.training.helper import get_train_dataset, get_val_dataset

    torch.set_num_threads(args.threads)
    sink = logger.add(str(run_dir / "evaluation.log"), encoding="utf-8")
    history = read_history(experiment / "training.log")
    history.to_csv(run_dir / "epoch_history.csv", index=False, encoding="utf-8-sig")
    training = yaml.safe_load((experiment / "config.yaml").read_text(encoding="utf-8"))
    train_end = pd.Timestamp(training["data"]["cuts"]["train_end"])
    val_end = pd.Timestamp(training["data"]["cuts"]["val_end"])
    model, preprocessor, core = load_model_and_config(str(checkpoint))
    source = dataset_paths("dk")["csv"]
    source_hash = sha256(source)
    expected_hash = json.loads((PROJECT_ROOT / "config/data_sources.json").read_text(
        encoding="utf-8"))["datasets"]["dk"]["sha256"]
    if source_hash != expected_hash:
        raise RuntimeError("DK source CSV no longer matches the recorded baseline")

    snapshots = load_snapshots(core)
    completed = get_df_of_completed_lots(snapshots)
    df_train = transform_df(snapshots, core["embed"], preprocessor=preprocessor, split_time=train_end)
    transformed = transform_df(snapshots, core["embed"], preprocessor=preprocessor)
    train_dataset = get_train_dataset(df_train, completed, deepcopy(core))
    scaler = (float(core["target_scaler"]["min"]), float(core["target_scaler"]["max"]))
    if not np.allclose([train_dataset.min_target, train_dataset.max_target], scaler):
        raise RuntimeError("Reconstructed training target range differs from saved configuration")
    train_ids = set(train_dataset.lot_ids)
    train_lots = completed.loc[sorted(train_ids)].copy()
    training_info = {
        "windows": len(train_dataset), "lots": len(train_ids),
        "products": int(train_lots.product_id.nunique()),
        "max_sequence_length": train_dataset.max_seq_len,
        "last_input_timestamp": str(df_train.created_at.max()),
        "max_label_end_time": str(train_lots.end_time.max()),
    }
    del train_dataset
    gc.collect()
    val_dataset = get_val_dataset(transformed, completed, deepcopy(core), train_end, val_end)
    val_scaler = (float(val_dataset.min_target), float(val_dataset.max_target))
    if manifest.get("target_scaler_policy") == "training_fixed" and val_scaler != scaler:
        raise RuntimeError("Corrected run must retain the training target scaler in validation")
    val_ids = set(val_dataset.lot_ids)
    if train_ids & val_ids:
        raise RuntimeError("Training and validation contain overlapping labelled Lots")
    # Exactly the saved model and enterprise dataset, with stable order for exporting rows.
    loader = val_dataset.to_dataloader(batch_size=training["training"]["batch_size"], shuffle=False)
    predictions, logged_targets, ordered_ids = [], [], []
    with torch.inference_mode():
        for ids, inputs, target, mask in loader:
            prediction = model(inputs).view(-1).cpu().numpy()
            keep = mask.cpu().numpy().astype(bool)
            predictions.extend(prediction[keep])
            logged_targets.extend(target.cpu().numpy()[keep])
            ordered_ids.extend(ids[keep].tolist())
    samples = val_dataset.samples
    if ordered_ids != [sample["lot_id"] for sample in samples]:
        raise RuntimeError("Export order does not match validation sample order")
    predictions = np.asarray(predictions, dtype=np.float32) * (scaler[1] - scaler[0]) + scaler[0]
    logged_targets = np.asarray(logged_targets, dtype=np.float32) * (scaler[1] - scaler[0]) + scaler[0]
    frame = pd.DataFrame({"lot_id": ordered_ids,
                          "actual_remaining_days": [s["RCT_in_days"] for s in samples],
                          "predicted_remaining_days": predictions,
                          "sequence_length": [s["length"] for s in samples]})
    for name in ("product_id", "start_time", "end_time", "duration", "LT"):
        frame[name] = frame.lot_id.map(completed[name])
    # The target is relative to the LAST OBSERVED SNAPSHOT, not the requested cutoff.
    frame["reference_snapshot_time"] = (
        frame.end_time - pd.to_timedelta(frame.actual_remaining_days, unit="D")
    ).dt.round("s")
    frame["predicted_completion_time"] = frame.reference_snapshot_time + pd.to_timedelta(
        frame.predicted_remaining_days, unit="D")
    frame["error_days"] = frame.predicted_remaining_days - frame.actual_remaining_days
    frame["absolute_error_days"] = frame.error_days.abs()
    frame["seen_product_in_labelled_training"] = frame.product_id.isin(train_lots.product_id)
    # A descriptive comparator fitted on TRAINING LABELS ONLY. It never changes the model.
    product_lt = train_lots.groupby("product_id").duration.mean()
    train_lt = frame.product_id.map(product_lt).fillna(train_lots.duration.mean())
    elapsed = (frame.reference_snapshot_time - frame.start_time).dt.total_seconds() / 86400
    frame["baseline_remaining_days"] = (train_lt - elapsed).clip(lower=0)
    frame.to_csv(run_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    frame.sort_values("absolute_error_days", ascending=False).head(50).to_csv(
        run_dir / "largest_errors.csv", index=False, encoding="utf-8-sig")
    latest = frame.sort_values("reference_snapshot_time").groupby("lot_id", as_index=False).tail(1)
    latest.to_csv(run_dir / "validation_latest_per_lot.csv", index=False, encoding="utf-8-sig")

    main_metrics = metrics(frame.actual_remaining_days, predictions)
    enterprise_metrics = metrics(logged_targets, predictions)
    if manifest.get("target_scaler_policy") == "training_fixed":
        if not np.allclose(logged_targets, frame.actual_remaining_days, atol=1e-5, rtol=1e-6):
            raise RuntimeError("Validation label round-trip does not preserve original days")
        if abs(main_metrics["mae_days"] - enterprise_metrics["mae_days"]) > 1e-4:
            raise RuntimeError("Logged MAE and original-day MAE must match for the corrected run")
    best_logged = float(history.val_mae_days.min())
    if abs(enterprise_metrics["mae_days"] - best_logged) > 0.001:
        raise RuntimeError(f"Reloaded checkpoint MAE {enterprise_metrics['mae_days']} differs from log {best_logged}")
    by_product = {}
    for seen, group in frame.groupby("seen_product_in_labelled_training"):
        by_product["已见商品" if seen else "未见商品"] = metrics(
            group.actual_remaining_days, group.predicted_remaining_days)
    horizons = pd.cut(frame.actual_remaining_days, [0, 3, 7, 14, float("inf")],
                      labels=["≤3天", "3–7天", "7–14天", ">14天"], include_lowest=True)
    by_horizon = {str(k): metrics(g.actual_remaining_days, g.predicted_remaining_days)
                  for k, g in frame.groupby(horizons, observed=True)}
    raw = pd.read_csv(source, usecols=["LotId", "stage", "file_timestamp"])
    raw_time = pd.to_datetime(raw.file_timestamp)
    data_info = {
        "file": str(source), "sha256": source_hash, "baseline_hash_matches": True,
        "raw_rows": len(raw), "raw_lots": int(raw.LotId.nunique()),
        "raw_missing_stage": int(raw.stage.isna().sum()),
        "raw_time_min": str(raw_time.min()), "raw_time_max": str(raw_time.max()),
        "normalized_rows": len(snapshots), "normalized_lots": int(snapshots.lot_id.nunique()),
        "normalized_time_min": str(snapshots.created_at.min()),
        "normalized_time_max": str(snapshots.created_at.max()),
        "completed_label_lots": len(completed), "preprocessor_fit_rows": len(df_train),
    }
    del raw
    validation_info = {
        "windows": len(frame), "lots": len(val_ids), "products": int(frame.product_id.nunique()),
        "train_val_labelled_lot_overlap": len(train_ids & val_ids),
        "reference_time_min": str(frame.reference_snapshot_time.min()),
        "reference_time_max": str(frame.reference_snapshot_time.max()),
        "reference_time_on_or_before_train_cut_windows": int((frame.reference_snapshot_time <= train_end).sum()),
        "reference_time_on_or_before_train_cut_lots": int(frame.loc[
            frame.reference_snapshot_time <= train_end, "lot_id"].nunique()),
        "max_label_end_time": str(frame.end_time.max()),
        "max_sequence_length": val_dataset.max_seq_len,
    }
    ended = datetime.fromisoformat(manifest["training_finished_at"])
    started = datetime.fromisoformat(manifest["started_at"])
    summary = {
        "run_id": manifest["run_id"], "evaluated_at": datetime.now().astimezone().isoformat(),
        "checkpoint": str(checkpoint), "experiment": str(experiment), "data": data_info,
        "training": training_info, "validation": validation_info,
        "epoch_count": len(history), "best_epoch_from_rounded_log": int(history.loc[
            history.val_mae_days.idxmin(), "epoch"]),
        "best_logged_val_mae_days": best_logged, "checkpoint_log_mae_matches": True,
        "training_wall_minutes": (ended - started).total_seconds() / 60,
        "early_stopped": "Early stopping triggered!" in (run_dir / "console.log").read_text(encoding="utf-8-sig"),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "python": sys.version, "torch": torch.__version__, "device": "cpu",
        "cpu_threads_training": manifest["cpu_threads"],
        "target_scaler_training": scaler, "target_scaler_validation": val_scaler,
        "raw_validation_target_range": [float(frame.actual_remaining_days.min()),
                                        float(frame.actual_remaining_days.max())],
        "target_scaler_matches": bool(np.allclose(scaler, val_scaler)),
        "training_source_hashes_match": True,
        "model_all_windows": main_metrics, "enterprise_logged_target_metrics": enterprise_metrics,
        "baseline_train_product_mean_lt": metrics(frame.actual_remaining_days, frame.baseline_remaining_days),
        "lot_macro_mae_days": float(frame.groupby("lot_id").absolute_error_days.mean().mean()),
        "latest_per_lot": metrics(latest.actual_remaining_days, latest.predicted_remaining_days),
        "by_product_seen": by_product, "by_actual_remaining_days": by_horizon,
        "source_config_unchanged": {
            name: sha256(run_dir / archived) == sha256(dataset_paths("dk")["config_dir"] / name)
            for name, archived in [("core_template.yaml", "source_core.yaml"),
                                    ("train_template.yaml", "source_train.yaml")]
        },
        "artifact_sha256": {name: sha256(checkpoint / name)
                            for name in ["best_model.pth", "config.yaml", "preprocessor.pkl"]},
    }
    if not all(summary["source_config_unchanged"].values()):
        raise RuntimeError("Enterprise source configuration changed during this run")
    after_cut = frame[frame.reference_snapshot_time > train_end]
    summary["reference_after_train_cut_only"] = metrics(
        after_cut.actual_remaining_days, after_cut.predicted_remaining_days)
    if manifest.get("baseline_run"):
        summary["baseline_comparison"] = compare_baseline(
            run_dir, project_path(manifest["baseline_run"]), frame, summary)
    write_json(run_dir / "evaluation_summary.json", summary)
    render_report(run_dir, summary, history, frame, manifest, training, core)
    manifest.update(status="report_completed", report=str(run_dir / "experiment_report.html"),
                    evaluation_summary=str(run_dir / "evaluation_summary.json"),
                    checkpoint=str(checkpoint), experiment=str(experiment),
                    report_completed_at=datetime.now().astimezone().isoformat())
    write_json(manifest_path, manifest)
    logger.remove(sink)
    print(json.dumps({"report": manifest["report"], "metrics": main_metrics,
                      "checkpoint_log_mae_matches": True}, ensure_ascii=False, indent=2))


def render_report(run_dir, summary, history, frame, manifest, training, core):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    axes[0].plot(history.epoch, history.train_mae_days, label="Training")
    fixed_scale = summary["target_scaler_matches"]
    axes[0].plot(history.epoch, history.val_mae_days,
                 label="Validation" if fixed_scale else "Validation (original log)")
    axes[0].set(xlabel="Epoch", ylabel="MAE (days)" if fixed_scale else "Logged MAE (see scaling note)",
                title="Training history" if fixed_scale else "Original training log")
    axes[0].legend()
    upper = float(max(frame.actual_remaining_days.max(), frame.predicted_remaining_days.max()))
    axes[1].scatter(frame.actual_remaining_days, frame.predicted_remaining_days, s=9, alpha=0.35)
    axes[1].plot([0, upper], [0, upper], "--", color="#56616f", linewidth=1)
    axes[1].set(xlabel="Observed remaining time (days)", ylabel="Predicted remaining time (days)",
                title="Validation windows")
    errors = np.sort(frame.absolute_error_days)
    axes[2].plot(errors, np.arange(1, len(errors) + 1) / len(errors))
    axes[2].set(xlabel="Absolute error (days)", ylabel="Cumulative fraction", title="Error distribution", ylim=(0, 1))
    figure = run_dir / "experiment_figures.png"
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    graphic = base64.b64encode(figure.read_bytes()).decode("ascii")
    d, tr, va, m = (summary[k] for k in ["data", "training", "validation", "model_all_windows"])
    title = manifest.get("experiment_title", "DK 首次全量训练实验报告")
    corrected = manifest.get("target_scaler_policy") == "training_fixed"
    scope = ("企业 v0.9.5 本地 CSV 基线 + 标签缩放修正" if corrected
             else "企业 v0.9.5 本地 CSV 基线")
    purpose = ("本次只修复验证标签缩放范围被重新拟合的问题，并增加范围一致性检查。"
               "原数据、清洗、标签定义、划分日期、模型参数、随机种子和早停耐心值均沿用前次设置。"
               if corrected else "按企业认可的处理与训练流程建立 DK 的首个全量本地基线。")
    baseline = summary["baseline_train_product_mean_lt"]
    gain = (baseline["mae_days"] - m["mae_days"]) / baseline["mae_days"]
    outcome = f"相对于商品平均周期基线，MAE 降低 {gain:.1%}。" if gain >= 0 else f"MAE 比商品平均周期基线高 {-gain:.1%}。"
    config_rows = [
        ["训练/验证划分", f"训练截止 {training['data']['cuts']['train_end']}；配置验证截止 {training['data']['cuts']['val_end']}"],
        ["历史窗口 / 步长", f"{core['dataset']['params']['window_days']} 天 / {core['dataset']['params']['step_days']} 天"],
        ["模型", f"Transformer：d_model={core['model']['params']['d_model']}，{core['model']['params']['n_head']} heads，{core['model']['params']['n_layers']} layers"],
        ["输入", "waferQty；woItem、stage、tranCode、oper；年/月/日/周几/时/分"],
        ["训练", f"batch={training['training']['batch_size']}；seed={training['training']['seed']}；最多 {training['training']['epochs']} 轮"],
        ["优化器", f"AdamW，lr={training['optimizer']['params']['lr']}，weight_decay={training['optimizer']['params']['weight_decay']}"],
        ["损失与调度", "L1Loss；10 轮 warmup + CosineAnnealingWarmRestarts"],
        ["早停", f"监控 val_MAE，patience={training['early_stopping']['patience']}，min_delta={training['early_stopping']['min_delta']}"],
        ["运行环境", f"Python {sys.version.split()[0]}；PyTorch {summary['torch']}；CPU {summary['cpu_threads_training']} 个计算线程；未做预训练"],
    ]
    scale_note = "验证与训练的目标缩放范围一致，重载后的 MAE 与训练日志最佳值一致（允许 0.001 天数值误差）。"
    if not summary["target_scaler_matches"]:
        scale_note = ("检测到原数据集实现在验证阶段扩展了目标缩放范围；训练日志 MAE 与按物理天数计算的 MAE 不同。"
                      f"日志口径重算 MAE={summary['enterprise_logged_target_metrics']['mae_days']:.4f} 天；"
                      "本报告主表采用原始 RCT_in_days 标签，须先与数据同学核对这一实现。")
    scaling_section = ""
    if corrected:
        scaling_section = (
            "<div class='callout'><b>修正验收：</b>训练、验证及推理复用训练标签范围；"
            f"本次固定范围为 {summary['target_scaler_training']}。"
            f"验证原始标签范围为 {summary['raw_validation_target_range']}，超界标签完整保留。"
            "标签往返转换、原始天数 MAE 与训练器 MAE 均已核对，且重载模型与日志最佳值一致。</div>"
        )
    if not summary["target_scaler_matches"]:
        t_min, t_max = summary["target_scaler_training"]
        v_min, v_max = summary["target_scaler_validation"]
        scaling_section = (
            "<div class='attention'><b>已确认的指标口径问题：</b>"
            f"训练标签范围为 {t_min:g}–{t_max:g} 天；验证数据集将范围扩展为 {v_min:g}–{v_max:g} 天，"
            "但训练器仍按训练范围反归一化验证标签。原日志实际比较的标签变成 "
            f"{t_min:g} + (原始剩余天数 − {v_min:g}) × {(t_max-t_min):g}/{(v_max-v_min):g}，"
            f"例如原始 {v_max:g} 天被还原成 {t_max:g} 天。"
            f"因此日志最佳 MAE {summary['best_logged_val_mae_days']:.4f} 不能直接解释为原始天数误差。"
            f"保持同一模型预测不变、与原始天数标签比较，MAE 为 {m['mae_days']:.4f} 天。"
            f"早停与模型选择也使用了这一日志指标，所以第 {summary['best_epoch_from_rounded_log']} 轮仅代表原代码选出的最佳模型，"
            "不能据此断言它在原始天数指标下也是所有轮次中最优。本次没有修改企业训练代码。</div>"
        )
    stopped = "触发早停" if summary["early_stopped"] else "达到配置训练轮数"
    comparison_section = ""
    if "baseline_comparison" in summary:
        comparison = summary["baseline_comparison"]
        change = comparison["mae_reduction_fraction"]
        change_text = f"降低 {change:.1%}" if change >= 0 else f"增加 {-change:.1%}"
        comparison_section = (
            "<h3>与修正前保存模型比较</h3>"
            "<p>已核对 CSV 哈希、配置文件以及逐窗口 Lot、参考快照时间和原始标签一致。"
            "下表两次结果均使用原始天数标签计算；修正前日志的错误缩放指标不作为比较基准。</p>"
            + metric_table({"修正前模型": comparison["before"], "修正后模型": comparison["after"]})
            + f"<p>MAE {change_text}；两次选出的轮次分别为 {comparison['before_best_epoch']} 和 "
            f"{comparison['after_best_epoch']}。修正的目的是保证评估与选模正确，效果是否提升以实际结果为准。</p>"
        )
    next_step = ("建议下一步核对最大误差 Lot 的真实完工记录，再按企业确认的时间口径安排独立回测。"
                 if corrected else
                 "建议下一步优先请数据同学确认目标缩放范围应在训练与验证中固定一致；确认后单独修复并重新实验，"
                 "同时核对最大误差 Lot 的真实完工记录，再按企业确认的时间口径安排独立回测。")
    sections = [
        f"<h1>{html.escape(title)}</h1><p class='muted'>实验 {html.escape(manifest['run_id'])} · {scope}</p>",
        f"<div class='callout'><b>训练完成：</b>共 {summary['epoch_count']} 轮，{stopped}，全程约 {summary['training_wall_minutes']:.1f} 分钟。"
        f"本次日志口径选出的最佳模型为第 {summary['best_epoch_from_rounded_log']} 轮（以四位小数日志定位）。"
        f"保存模型的验证 MAE 为 <b>{m['mae_days']:.3f} 天</b>，{m['within_3_days']:.1%} 的窗口误差不超过 3 天。{outcome}</div>",
        scaling_section,
        "<h2>1. 实验目的与口径</h2><p>" + purpose +
        "本次单一随机种子、单次训练，验证集同时用于早停和选模；结果不属于独立测试集表现。"
        "所有误差以天为单位，预测目标是距最后一条输入快照的剩余时间。</p>",
        "<h2>2. 数据与处理</h2>",
        table(["项目", "数量/范围"], [
            ["数据文件", Path(d['file']).name], ["原始记录 / Lot", f"{d['raw_rows']:,} / {d['raw_lots']:,}"],
            ["缺失 stage → NULL", f"{d['raw_missing_stage']:,} 行"],
            ["标准化记录 / Lot", f"{d['normalized_rows']:,} / {d['normalized_lots']:,}"],
            ["标准化数据时间范围", f"{d['normalized_time_min']} — {d['normalized_time_max']}"],
            ["原规则认定可构造标签的 Lot", f"{d['completed_label_lots']:,}"],
            ["训练输入快照行数", f"{d['preprocessor_fit_rows']:,}"],
            ["训练窗口 / Lot / 商品", f"{tr['windows']:,} / {tr['lots']:,} / {tr['products']:,}"],
            ["验证窗口 / Lot / 商品", f"{va['windows']:,} / {va['lots']:,} / {va['products']:,}"],
            ["训练和验证标签 Lot 重叠", va['train_val_labelled_lot_overlap']],
        ]),
        "<p>处理采用原代码：缺失 stage 填 NULL、时间四舍五入到 10 分钟、Lot/时间去重、商品及父批次一致性检查。"
        "DK 保留不规则快照，不按天补齐，不额外删除单 stage 批次。完成标签取 root Lot 的最后观测时间，并排除数据起止边界附近批次。</p>",
        "<h2>3. 模型与训练设置</h2>", table(["参数", "设置"], config_rows),
        f"<p>模型参数量 {summary['model_parameters']:,}。关闭逐轮绘图，训练后统一生成图表；源配置文件哈希与开跑时一致。</p>",
        "<h2>4. 验证结果</h2>", metric_table({"Transformer": m, "训练商品平均周期基线": baseline}),
        comparison_section,
        f"<p>基线仅使用训练标签 Lot 的商品平均周期，减去当前已观测时长，结果下限为零；新商品使用训练总体平均周期。"
        f"它仅作为事后比较。模型预测未裁剪：负值 {m['negative_predictions']} 个；有符号偏差 {m['bias_days']:.3f} 天"
        "（负数表示偏早预测）。</p>",
        f"<p>每个 Lot 内先平均、再对 Lot 平均的 MAE：{summary['lot_macro_mae_days']:.3f} 天。"
        f"仅保留每个验证 Lot 最晚评估窗口的 MAE：{summary['latest_per_lot']['mae_days']:.3f} 天。{scale_note}</p>",
        metric_table(summary["by_product_seen"]),
        "<p>“已见/未见商品”以是否出现在带完工标签的训练 Lot 中区分；类别编码器可能已见过该商品在截止日前的快照。</p>",
        metric_table(summary["by_actual_remaining_days"]),
        "<p>长剩余时间窗口的偏早预测较明显，应结合最大的误差明细核对。上面的分组按原始标签分段，"
        "并不代表统一预测日的在制品分布。</p>",
        f"<img alt='Training history, validation predictions and error distribution' src='data:image/png;base64,{graphic}'>",
        "<h2>5. 结果解释与适用范围</h2><ul>",
        "<li>完工标签是企业现有快照规则推定的最后出现时间，未与独立的真实出货/完工事件逐条核验。</li>",
        "<li>原验证窗口围绕商品平均周期约三分之一处构造；不是在所有生产阶段均匀抽取，也不是统一某日的全量在制品快照。</li>",
        f"<li>原代码按完工时间划分标签 Lot，验证窗口的最后输入时间可早于训练截止。本次共有 "
        f"{va['reference_time_on_or_before_train_cut_windows']:,} 个验证窗口、"
        f"{va['reference_time_on_or_before_train_cut_lots']:,} 个 Lot 的参考时间不晚于训练截止，"
        "因此整体结果不能直接解释为严格的截止日后滚动预测测试。</li>",
        "<li>原代码先从全体可标注 Lot 计算商品 LT，再用 LT 选择验证窗口；该抽样依赖后验完整历史。"
        "本次保留原定义，若后续评价真实上线泛化，应另做严格时间回测并让企业确认评估口径。</li>",
        f"<li>数据实际截至 {html.escape(d['normalized_time_max'])}，配置验证截止仍为 {training['data']['cuts']['val_end']}；"
        "配置日期不会补出缺失的后续数据。</li></ul>",
        "<h2>6. 预测明细与复现</h2><p>validation_predictions.csv 为每个验证窗口的商品、Lot、最后输入快照、"
        "预测剩余天数、预测完工时刻及误差；validation_latest_per_lot.csv 为每个验证 Lot 的最晚评估窗口。"
        "预测完工时刻 = 最后输入快照时间 + 预测剩余天数。largest_errors.csv 列出误差最大的 50 个窗口。"
        "这些文件是历史验证结果，不能当作当前实时在制清单。</p>",
        table(["产物", "本地路径"], [
            ["最佳模型与预处理器", summary['checkpoint']], ["逐轮日志", str(Path(summary['experiment']) / 'training.log')],
            ["完整评估指标", str(run_dir / 'evaluation_summary.json')], ["预测明细", str(run_dir / 'validation_predictions.csv')],
            ["数据 SHA-256", d['sha256']], ["起始代码提交", manifest['git_head']],
        ]),
        "<p>本次包含未提交的本地环境与辅助脚本；具体状态见 run_manifest.json，源码哈希见 source_hashes.json。"
        "数据、模型和训练输出保留在本地。未执行 Git commit 或 push。</p>",
        "<pre>&amp; .\\.venv\\Scripts\\python.exe run.py train --dataset dk --no-plot</pre>",
        "<p>复现时设置 OMP_NUM_THREADS=4、MKL_NUM_THREADS=4，并使用本项目 .venv。" + next_step + "</p>",
    ]
    css = """body{font-family:'Microsoft YaHei','Segoe UI',sans-serif;color:#172b3a;max-width:1120px;margin:40px auto;padding:0 26px;line-height:1.8}h1{font-size:30px}h2{font-size:21px;margin-top:34px;color:#164a67}p,li{font-size:15px}.muted{color:#607283}.callout{background:#edf5fb;border-left:5px solid #247ba0;padding:18px 22px}.attention{background:#fff5df;border-left:5px solid #b67515;padding:18px 22px;margin:20px 0;font-size:15px}table{width:100%;border-collapse:collapse;font-size:14px;margin:15px 0}th,td{text-align:left;padding:9px 11px;border-bottom:1px solid #d8e1e7;overflow-wrap:anywhere}th{background:#f0f4f7}img{width:100%;margin:20px 0}pre{background:#f4f6f8;padding:15px;overflow:auto}@media print{body{margin:0;max-width:none;font-size:12px}h2{break-after:avoid}table,img,.callout,.attention{break-inside:avoid}}"""
    (run_dir / "experiment_report.html").write_text(
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>" + html.escape(title) + "</title><style>" + css + "</style><body>" + "".join(sections) + "</body></html>",
        encoding="utf-8")


if __name__ == "__main__":
    main()
