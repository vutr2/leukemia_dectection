"""Minimal unit checks for the v2.7 reproducibility and anti-leakage rules."""
import unittest
import numpy as np
import pandas as pd
from feature_robustness_v27 import prepare_versions,correct_time_axis,comparisons

class V27Test(unittest.TestCase):
    def mk(self):
        rec=[]
        for pid,vals in [('p1',[.02,.026,.03,.04]),('p2',[0,.001,.002,.025])]:
            for w,v in enumerate(vals,1):
                rec.append({'patient_id':pid,'cohort':'GSE178936','variant_id':'v1',
                            'time_years':float(w-1),'wave':float(w),'vaf':v,
                            'depth':500,'gene':'DNMT3A','detection_limit':.015 if not(pid=='p1' and w==2) else np.nan,
                            'detection_limit_source':'X95MDAF'})
        raw=pd.DataFrame(rec)
        filtered=raw[raw.vaf>=.01].copy()
        return filtered,raw

    def test_missing_limit_never_imputed(self):
        a,b=self.mk();v=prepare_versions(a,b,approx_time=False)
        d=v['V1_LOD_FLOORED_AUDIT_ONLY']
        row=d[(d.patient_id=='p1')&(d.wave==2)].iloc[0]
        self.assertTrue(np.isnan(row.vaf_lod_floored))
        self.assertFalse(row.quantifiable)
        self.assertEqual(v['V1_RELIABLE'].query("patient_id=='p1'").shape[0],3)

    def test_no_last_wave_ascertainment(self):
        a,b=self.mk();v=prepare_versions(a,b,approx_time=False)
        self.assertFalse(v['V1_LOD_FLOORED_AUDIT_ONLY'].query("patient_id=='p2'").known_pre_index.any())
        self.assertNotIn('p2',set(v['V2_PREINDEX_RELIABLE'].patient_id))

    def test_time_axis_is_versioned(self):
        a,b=self.mk();z=correct_time_axis(a,True)
        self.assertTrue(np.array_equal(z.time_years_v26_original,a.time_years))
        self.assertTrue(np.array_equal(z.time_years,a.time_years))

    def test_tie_never_fake_perfect_top_quintile(self):
        a=pd.DataFrame({'patient_id':[str(i) for i in range(20)],'cohort':['A']*20,
                        'n_variants_crossed_vaf_ge_20pct':[0]*19+[1]})
        b=a.copy()
        from model_spec import CORE_DYNAMICS_V23
        for f in CORE_DYNAMICS_V23:
            if f not in a:
                a[f]=np.arange(20)
                b[f]=np.arange(20)
        c,_=comparisons({'V0':a,'V1_RELIABLE':b,'V2_PREINDEX_RELIABLE':b})
        z=c.query("left=='V0' and right=='V2_PREINDEX_RELIABLE' and cohort=='A' and feature=='n_variants_crossed_vaf_ge_20pct'")
        self.assertTrue(bool(z.iloc[0].top_quintile_tie_ambiguous))
        self.assertTrue(pd.isna(z.iloc[0].top_quintile_agreement))

if __name__=='__main__':
    unittest.main()
