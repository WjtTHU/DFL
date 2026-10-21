"""Window/Lot metrics and honest summaries of an open-ended categorical tail."""
import numpy as np
import pandas as pd


def point_metrics(index, predicted):
    actual = index.remaining_days.to_numpy(dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if not len(actual) or not np.isfinite(predicted).all():
        raise ValueError('Metrics require nonempty finite predictions.')
    error = predicted-actual
    ae = np.abs(error)
    by_lot = pd.Series(ae).groupby(index.lot_id.to_numpy()).mean()
    return {'windows': len(actual), 'lots': len(by_lot), 'mae_days': float(ae.mean()),
            'rmse_days': float(np.sqrt(np.square(error).mean())),
            'lot_macro_mae_days': float(by_lot.mean()), 'bias_days': float(error.mean()),
            'median_ae_days': float(np.median(ae)), 'p90_ae_days': float(np.quantile(ae, .9)),
            'within_3_days': float((ae <= 3).mean())}


def distribution_summary(probabilities, horizon):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != horizon+1 or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-6):
        raise ValueError('Probabilities must be finite, nonnegative and sum to one over days and tail.')
    result = {'tail_probability': p[:, -1],
              'mean_days_lower_bound': p @ np.arange(1, horizon+2)}
    cdf = p.cumsum(1)
    cdf[:, -1] = 1.0
    for name, quantile in [('q05', .05), ('median', .5), ('q95', .95)]:
        category = (cdf < quantile).sum(1)
        result[f'{name}_lower_days'] = category+1
        result[f'{name}_upper_days'] = np.where(category == horizon, np.nan, category+1)
        result[f'{name}_in_tail'] = category == horizon
    return result


def distribution_metrics(index, probabilities, horizon, nll_values):
    p = np.asarray(probabilities, dtype=float)
    summaries = distribution_summary(p, horizon)
    actual = index.remaining_days.to_numpy(dtype=int)
    categories = np.where(actual > horizon, horizon, actual-1)
    losses = np.asarray(nll_values, dtype=float)
    if not len(losses) or not np.isfinite(losses).all():
        raise ValueError('Non-finite NLL.')
    q05 = summaries['q05_lower_days']-1
    q95 = summaries['q95_lower_days']-1
    coverage = (categories >= q05) & (categories <= q95)
    calibration = {}
    thresholds = sorted({1, 3, 7, 14, 21, 28, 30, horizon} & set(range(1, horizon+1)))
    for day in thresholds:
        forecast = p[:, :day].sum(1)
        truth = (actual <= day).astype(float)
        bins = np.minimum((forecast*10).astype(int), 9)
        groups = []
        ece = 0.0
        for b in range(10):
            mask = bins == b
            if mask.any():
                fp, tp = float(forecast[mask].mean()), float(truth[mask].mean())
                ece += mask.mean()*abs(fp-tp)
                groups.append({'bin': b, 'n': int(mask.sum()), 'predicted_cdf': fp, 'observed_fraction': tp})
        calibration[str(day)] = {'brier_score': float(np.square(forecast-truth).mean()),
                                 'ece_10_bins': float(ece), 'bins': groups}
    return {'windows': len(actual), 'lots': int(index.lot_id.nunique()),
            'coarsened_nll': float(losses.mean()),
            'lot_macro_coarsened_nll': float(pd.Series(losses).groupby(index.lot_id.to_numpy()).mean().mean()),
            'actual_tail_fraction': float((actual > horizon).mean()),
            'mean_predicted_tail_probability': float(p[:, -1].mean()),
            'central_90_class_interval_coverage': float(coverage.mean()),
            'central_90_finite_upper_fraction': float((q95 < horizon).mean()),
            'calibration_by_horizon_days': calibration,
            'note': 'NLL scores the day/tail category. Tail quantiles have open upper bounds; an exact mean is not identified.'}


def evaluate_predictions(index, outputs, mode, spec, losses=None):
    frame = index.drop(columns=['row_start', 'row_stop']).reset_index(drop=True).copy()
    if mode == 'point':
        raw = outputs.reshape(-1)*spec['scale']+spec['min']
        pred = np.maximum(raw, 0)
        frame['raw_predicted_remaining_days'] = raw
        frame['predicted_remaining_days'] = pred
        # This fractional value is a point estimate of integer-date difference, not an intraday timestamp.
        frame['predicted_completion_date_ceil'] = (frame.forecast_date + pd.to_timedelta(np.ceil(pred), unit='D')).dt.date
        result = point_metrics(index, pred) if index.remaining_days.notna().all() else None
        if result is not None:
            result['negative_raw_predictions'] = int((raw < 0).sum())
            frame['error_days'] = pred-index.remaining_days.to_numpy(dtype=float)
    else:
        # Compute softmax in float64 for stable exported probabilities.
        logits = outputs.astype(float)-outputs.max(1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(1, keepdims=True)
        for name, values in distribution_summary(probabilities, spec['horizon']).items():
            frame[name] = values
        probability_frame = pd.DataFrame(probabilities, columns=[*[f'p_day_{i}' for i in range(1, spec['horizon']+1)], 'p_tail'])
        frame = pd.concat([frame, probability_frame], axis=1)
        result = distribution_metrics(index, probabilities, spec['horizon'], losses) if losses is not None else None
    if result is not None:
        result['by_remaining_days'] = {}
        groups = [('1–3', 1, 3), ('4–7', 4, 7), ('8–14', 8, 14)]
        if spec['horizon'] > 14:
            groups += [(f"15–{spec['horizon']}", 15, spec['horizon']),
                       (f">{spec['horizon']}", spec['horizon']+1, float('inf'))]
        else:
            groups += [('>14', 15, float('inf'))]
        for name, low, high in groups:
            mask = (index.remaining_days >= low) & (index.remaining_days <= high)
            if mask.any():
                if mode == 'point':
                    values = point_metrics(index.loc[mask], pred[mask])
                else:
                    values = {'windows': int(mask.sum()), 'coarsened_nll': float(np.asarray(losses)[mask].mean())}
                result['by_remaining_days'][name] = values
    # Inference artifacts never carry placeholder truth/label fields.
    if index.remaining_days.isna().all():
        frame = frame.drop(columns=['remaining_days', 'completion_at', 'label_available_at'])
    return frame, result
