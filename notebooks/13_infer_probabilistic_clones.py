#!/usr/bin/env python
"""Run LeukemiPrediag v2.1 uncertainty-aware clone inference on real serial data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).parent.name == 'notebooks' else Path.cwd()
for candidate in [ROOT / 'src', Path('/content/blood-cancer-early-detection/src'), Path('/mnt/data')]:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from clone_inference import CloneInferenceConfig, run_clone_inference


def uncertainty_comparison(result: dict) -> dict:
    tr = result['trajectory_posteriors']
    assn = result['assignments']
    clones = result['clone_posteriors']
    multi = clones[clones['n_variants'] > 1]
    if multi.empty:
        return {'n_multivariant_clones': 0}
    x = (multi.merge(assn[['patient_id','variant_id','clone_id']], on=['patient_id','clone_id'])
              .merge(tr[['patient_id','variant_id','growth_sd']], on=['patient_id','variant_id']))
    q = (x.groupby(['patient_id','clone_id'])
          .agg(clone_sd=('clone_growth_sd','first'), mean_member_sd=('growth_sd','mean'))
          .reset_index())
    ratio = q['clone_sd'] / q['mean_member_sd']
    return {
        'n_multivariant_clones': int(len(q)),
        'median_clone_to_member_sd_ratio': float(ratio.median()),
        'fraction_with_lower_clone_uncertainty': float((q['clone_sd'] < q['mean_member_sd']).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--serial', default='data/processed/v2/real_serial_filtered.csv')
    ap.add_argument('--patient-features', default='data/processed/v2/real_patient_features.csv')
    ap.add_argument('--outdir', default='data/processed/v2/clone_inference')
    ap.add_argument('--draws', type=int, default=1000)
    ap.add_argument('--support', type=float, default=0.80)
    args = ap.parse_args()

    serial = pd.read_csv(args.serial)
    cfg = CloneInferenceConfig(n_draws=args.draws, pair_support_threshold=args.support)
    result = run_clone_inference(serial, cfg)
    result['audit']['uncertainty_comparison'] = uncertainty_comparison(result)

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    result['trajectory_posteriors'].to_csv(out/'trajectory_posteriors.csv', index=False)
    result['pair_support'].to_csv(out/'pair_support.csv', index=False)
    result['assignments'].to_csv(out/'clone_assignments.csv', index=False)
    result['clone_posteriors'].to_csv(out/'clone_posteriors.csv', index=False)
    result['patient_features'].to_csv(out/'patient_clone_features.csv', index=False)

    if Path(args.patient_features).exists():
        base = pd.read_csv(args.patient_features)
        merged = base.merge(result['patient_features'].drop(columns=['cohort'], errors='ignore'),
                            on='patient_id', how='left', validate='one_to_one')
        merged.to_csv(out/'real_patient_features_cloneaware.csv', index=False)

    with open(out/'clone_inference_audit.json','w') as f:
        json.dump(result['audit'], f, indent=2)
    print(json.dumps(result['audit'], indent=2))


if __name__ == '__main__':
    main()
