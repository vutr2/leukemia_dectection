"""Outcome-free temporal validation for LeukemiPrediag clone kinetics.

This module evaluates whether longitudinal VAF kinetics contain genuine forward
predictive signal by holding out the last measurement of each trajectory and
predicting it from earlier measurements.  It does not use leukemia outcomes.

Models
------
- carry_forward: last observed training VAF (strong no-growth baseline)
- raw_linear: linear VAF trend
- log_linear: exponential-growth approximation via linear log(VAF)

Research use only. Not a diagnostic device.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalValidationConfig:
    patient_col: str = "patient_id"
    cohort_col: str = "cohort"
    variant_col: str = "variant_id"
    time_col: str = "time_years"
    vaf_col: str = "vaf"
    min_total_timepoints: int = 3
    min_train_timepoints: int = 2
    log_eps: float = 1e-5
    max_vaf_clip: float = 1.0


def _prepare(g: pd.DataFrame, cfg: TemporalValidationConfig) -> pd.DataFrame:
    return (
        g[[cfg.time_col, cfg.vaf_col]]
        .dropna()
        .groupby(cfg.time_col, as_index=False)[cfg.vaf_col]
        .mean()
        .sort_values(cfg.time_col)
    )


def leave_last_timepoint_forecasts(
    serial: pd.DataFrame,
    cfg: TemporalValidationConfig = TemporalValidationConfig(),
) -> pd.DataFrame:
    """One-step-ahead forecast table, one row per eligible trajectory."""
    req = [cfg.patient_col, cfg.cohort_col, cfg.variant_col, cfg.time_col, cfg.vaf_col]
    missing = [c for c in req if c not in serial.columns]
    if missing:
        raise KeyError(f"serial missing required columns: {missing}")

    rows = []
    for (pid, cohort, vid), g in serial.groupby(
        [cfg.patient_col, cfg.cohort_col, cfg.variant_col], sort=False
    ):
        z = _prepare(g, cfg)
        if len(z) < cfg.min_total_timepoints:
            continue
        train = z.iloc[:-1].copy()
        test = z.iloc[-1]
        if len(train) < cfg.min_train_timepoints:
            continue

        t = train[cfg.time_col].to_numpy(float)
        v = np.clip(train[cfg.vaf_col].to_numpy(float), cfg.log_eps, cfg.max_vaf_clip)
        t_test = float(test[cfg.time_col])
        y_true = float(np.clip(test[cfg.vaf_col], cfg.log_eps, cfg.max_vaf_clip))
        if np.ptp(t) <= 0 or t_test <= t.max():
            continue

        # No-growth / persistence baseline.
        pred_cf = float(v[-1])

        # Raw VAF linear forecast.
        b_raw, a_raw = np.polyfit(t, v, 1)
        pred_raw = float(np.clip(a_raw + b_raw * t_test, cfg.log_eps, cfg.max_vaf_clip))

        # Log-linear / exponential forecast.
        lv = np.log(v)
        b_log, a_log = np.polyfit(t, lv, 1)
        pred_log = float(np.clip(np.exp(a_log + b_log * t_test), cfg.log_eps, cfg.max_vaf_clip))

        row = {
            cfg.patient_col: pid,
            cfg.cohort_col: cohort,
            cfg.variant_col: vid,
            "n_train_timepoints": int(len(train)),
            "train_start_time": float(t.min()),
            "train_end_time": float(t.max()),
            "forecast_time": t_test,
            "forecast_horizon_years": float(t_test - t.max()),
            "last_train_vaf": float(v[-1]),
            "true_vaf": y_true,
            "pred_carry_forward": pred_cf,
            "pred_raw_linear": pred_raw,
            "pred_log_linear": pred_log,
            "train_raw_slope": float(b_raw),
            "train_log_slope": float(b_log),
        }
        for name, pred in [
            ("carry_forward", pred_cf),
            ("raw_linear", pred_raw),
            ("log_linear", pred_log),
        ]:
            row[f"abs_error_{name}"] = abs(pred - y_true)
            row[f"sq_error_{name}"] = (pred - y_true) ** 2
            row[f"abs_log_error_{name}"] = abs(
                np.log(pred + cfg.log_eps) - np.log(y_true + cfg.log_eps)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _metrics_for(df: pd.DataFrame, model: str) -> Dict[str, float]:
    if df.empty:
        return {"n": 0}
    ae = pd.to_numeric(df[f"abs_error_{model}"], errors="coerce")
    se = pd.to_numeric(df[f"sq_error_{model}"], errors="coerce")
    ale = pd.to_numeric(df[f"abs_log_error_{model}"], errors="coerce")
    true = pd.to_numeric(df["true_vaf"], errors="coerce")
    pred = pd.to_numeric(df[f"pred_{model}"], errors="coerce")
    ok = np.isfinite(ae) & np.isfinite(se) & np.isfinite(ale) & np.isfinite(true) & np.isfinite(pred)
    if not ok.any():
        return {"n": 0}
    ae, se, ale, true, pred = ae[ok], se[ok], ale[ok], true[ok], pred[ok]
    return {
        "n": int(ok.sum()),
        "mae_vaf": float(ae.mean()),
        "median_ae_vaf": float(ae.median()),
        "rmse_vaf": float(np.sqrt(se.mean())),
        "mae_log_vaf": float(ale.mean()),
        "median_ae_log_vaf": float(ale.median()),
        "mean_bias_vaf": float((pred - true).mean()),
    }


def summarize_forecasts(forecasts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Overall and cohort-specific forecast metrics."""
    models = ["carry_forward", "raw_linear", "log_linear"]
    overall = []
    for m in models:
        r = _metrics_for(forecasts, m)
        r["model"] = m
        overall.append(r)
    overall = pd.DataFrame(overall).set_index("model")

    if not forecasts.empty and "abs_log_error_carry_forward" in forecasts:
        base = forecasts["abs_log_error_carry_forward"]
        for m in ["raw_linear", "log_linear"]:
            overall.loc[m, "fraction_better_than_carry_forward"] = float(
                (forecasts[f"abs_log_error_{m}"] < base).mean()
            )
            overall.loc[m, "mean_delta_abs_log_error_vs_carry_forward"] = float(
                (forecasts[f"abs_log_error_{m}"] - base).mean()
            )

    cohort_rows = []
    if not forecasts.empty:
        for cohort, g in forecasts.groupby("cohort", sort=False):
            for m in models:
                r = _metrics_for(g, m)
                r.update({"cohort": cohort, "model": m})
                cohort_rows.append(r)
    cohort = pd.DataFrame(cohort_rows)
    return overall, cohort


def temporal_validation_audit(
    serial: pd.DataFrame,
    cfg: TemporalValidationConfig = TemporalValidationConfig(),
) -> Dict[str, object]:
    f = leave_last_timepoint_forecasts(serial, cfg)
    overall, cohort = summarize_forecasts(f)
    return {
        "config": asdict(cfg),
        "n_eligible_trajectories": int(len(f)),
        "n_patients": int(f[cfg.patient_col].nunique()) if len(f) else 0,
        "overall": overall.reset_index().to_dict(orient="records"),
        "by_cohort": cohort.to_dict(orient="records"),
    }


def _trajectory_forecast_table(
    serial: pd.DataFrame,
    cfg: TemporalValidationConfig,
) -> pd.DataFrame:
    """Alias kept private so cross-fitting can build train/test forecast tables."""
    return leave_last_timepoint_forecasts(serial, cfg)


def _tune_log_shrinkage_alpha(
    forecasts: pd.DataFrame,
    alpha_grid=None,
) -> float:
    """Tune one global shrinkage weight on training-patient trajectories only.

    alpha=0 is carry-forward; alpha=1 is the unshrunk log-linear forecast.
    Blending in log space is equivalent to shrinking the estimated growth slope
    toward zero while anchoring the forecast at the last observed VAF.
    """
    if alpha_grid is None:
        alpha_grid = np.linspace(0.0, 1.0, 21)
    if forecasts.empty:
        return 0.0
    true = np.clip(forecasts["true_vaf"].to_numpy(float), 1e-12, None)
    cf = np.clip(forecasts["pred_carry_forward"].to_numpy(float), 1e-12, None)
    ll = np.clip(forecasts["pred_log_linear"].to_numpy(float), 1e-12, None)
    best_alpha, best_loss = 0.0, np.inf
    for a in alpha_grid:
        pred = np.exp((1.0 - float(a)) * np.log(cf) + float(a) * np.log(ll))
        loss = float(np.mean(np.abs(np.log(pred) - np.log(true))))
        if loss < best_loss - 1e-12:
            best_alpha, best_loss = float(a), loss
    return best_alpha


def cross_fitted_shrinkage_forecasts(
    serial: pd.DataFrame,
    cfg: TemporalValidationConfig = TemporalValidationConfig(),
    *,
    n_splits: int = 5,
    alpha_grid=None,
    seed: int = 17,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Patient-held-out validation of a conservatively shrunk growth forecast.

    The shrinkage weight is selected only on training patients in each outer
    fold, then applied to completely held-out patients.  This avoids selecting
    the growth weight on the same patients used for evaluation.
    """
    from sklearn.model_selection import KFold

    patients = np.asarray(sorted(serial[cfg.patient_col].astype(str).unique()))
    if len(patients) < n_splits:
        raise ValueError("n_splits exceeds number of patients")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    fold_rows = []

    for fold, (tr_idx, te_idx) in enumerate(kf.split(patients), 1):
        train_ids = set(patients[tr_idx])
        test_ids = set(patients[te_idx])
        train_serial = serial[serial[cfg.patient_col].astype(str).isin(train_ids)]
        test_serial = serial[serial[cfg.patient_col].astype(str).isin(test_ids)]

        train_f = _trajectory_forecast_table(train_serial, cfg)
        test_f = _trajectory_forecast_table(test_serial, cfg)
        alpha = _tune_log_shrinkage_alpha(train_f, alpha_grid=alpha_grid)

        if len(test_f):
            cf = np.clip(test_f["pred_carry_forward"].to_numpy(float), cfg.log_eps, None)
            ll = np.clip(test_f["pred_log_linear"].to_numpy(float), cfg.log_eps, None)
            pred = np.exp((1.0 - alpha) * np.log(cf) + alpha * np.log(ll))
            pred = np.clip(pred, cfg.log_eps, cfg.max_vaf_clip)
            test_f = test_f.copy()
            test_f["outer_fold"] = fold
            test_f["shrinkage_alpha"] = alpha
            test_f["pred_shrunk_log_linear"] = pred
            test_f["abs_error_shrunk_log_linear"] = np.abs(pred - test_f["true_vaf"].to_numpy(float))
            test_f["sq_error_shrunk_log_linear"] = (pred - test_f["true_vaf"].to_numpy(float)) ** 2
            test_f["abs_log_error_shrunk_log_linear"] = np.abs(
                np.log(pred + cfg.log_eps) - np.log(test_f["true_vaf"].to_numpy(float) + cfg.log_eps)
            )
            rows.append(test_f)

        fold_rows.append({
            "fold": fold,
            "n_train_patients": len(train_ids),
            "n_test_patients": len(test_ids),
            "n_train_trajectories": int(len(train_f)),
            "n_test_trajectories": int(len(test_f)),
            "shrinkage_alpha": float(alpha),
        })

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out, pd.DataFrame(fold_rows)


def summarize_cross_fitted_shrinkage(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Compare carry-forward, unshrunk growth, and cross-fitted shrunk growth."""
    models = ["carry_forward", "log_linear", "shrunk_log_linear"]
    rows = []
    for model in models:
        if f"pred_{model}" not in forecasts.columns:
            continue
        r = _metrics_for(forecasts, model)
        r["model"] = model
        rows.append(r)
    out = pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()
    if len(out) and "abs_log_error_carry_forward" in forecasts:
        base = forecasts["abs_log_error_carry_forward"]
        for m in ["log_linear", "shrunk_log_linear"]:
            if f"abs_log_error_{m}" in forecasts:
                out.loc[m, "fraction_better_than_carry_forward"] = float(
                    (forecasts[f"abs_log_error_{m}"] < base).mean()
                )
                out.loc[m, "mean_delta_abs_log_error_vs_carry_forward"] = float(
                    (forecasts[f"abs_log_error_{m}"] - base).mean()
                )
    return out
