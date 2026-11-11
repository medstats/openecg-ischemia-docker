import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json
import numpy as np
import torch
from ecg_adapters import (InputError, validate_waveform, longest_finite_interval,
                          resample_observed, median_input, gustafsson_input)
from model_inference import load_ischemia_models, predict_ischemia
from run_all_ecg import layout_definition, short_canonical_from_rows


def synthetic(n=5000, fs=500):
    t = np.arange(n) / fs
    row = np.zeros(n)
    for r in np.arange(.6, t[-1] - .3, .8):
        row += np.exp(-((t - r) / .014) ** 2) - .2 * np.exp(-((t-r-.035)/.015)**2)
        row += .25 * np.exp(-((t-r-.22)/.055)**2)
    return np.stack([row * (.6 + i * .06) for i in range(12)])


class ShapeCheckedModel(torch.nn.Module):
    def __init__(self, expected, fail=False):
        super().__init__();self.expected=expected;self.fail=fail;self.calls=0
    def forward(self, x):
        self.calls += 1
        if tuple(x.shape) != self.expected:raise AssertionError(f'Wrong shape {x.shape}')
        if not torch.isfinite(x).all():raise AssertionError('NaNs reached model')
        return torch.tensor([[float('nan'), 1.]]) if self.fail else torch.tensor([[0., 1.]])


class InputRegressionTests(unittest.TestCase):
    def test_units_explicit_and_no_guessing(self):
        a=synthetic();np.testing.assert_allclose(validate_waveform(a*1000,500,'uV'),a)
        with self.assertRaises(InputError):validate_waveform(a,500,'unknown')
        with self.assertRaises(InputError):validate_waveform(a,float('nan'),'mV')

    def test_missing_runs_not_concatenated(self):
        a=synthetic();a[:,1000:1500]=np.nan
        lo,hi=longest_finite_interval(a)
        self.assertEqual((lo,hi),(1500,5000))
        self.assertEqual(hi-lo,3500)

    def test_resampling_preserves_time_and_landmark(self):
        a=np.zeros((12,1000));a[:,500]=1
        b=resample_observed(a,500,400)
        self.assertEqual(b.shape,(12,800));self.assertEqual(int(b[0].argmax()),400)

    def test_canonical_missing_slots_yield_medians_not_zeros(self):
        a=synthetic();layout=layout_definition('standard_3x4')
        canonical=np.full_like(a,np.nan)
        for row in layout['leads']:
            for col,label in enumerate(row):
                from ecg_adapters import LEADS
                i=LEADS.index(label);canonical[i,col*1250:(col+1)*1250]=a[i,:1250]
        med,info=median_input(canonical,500,'canonical',layout)
        self.assertEqual(med.shape,(12,500));self.assertTrue(np.isfinite(med).all())
        self.assertTrue(all(c['complete_beats']>=2 for c in info['columns']))
        # R must remain near the training-time anchor, and retain millivolt amplitude.
        self.assertTrue(np.all(np.abs(med.argmax(axis=1)-240)<=2))
        self.assertGreater(float(med.max()),.8);self.assertLess(float(med.max()),2.)

    def test_flat_missing_and_short_columns_rejected(self):
        a=synthetic();a[3]=0
        with self.assertRaises(InputError):median_input(a,500,'native',simultaneous=True)
        a=synthetic(550)
        with self.assertRaises(InputError):median_input(a,500,'native',simultaneous=True)
        a=synthetic();a[1,1500:]=np.nan
        with self.assertRaises(InputError):median_input(a,500,'native',simultaneous=True)

    def test_vendor_median_not_resized(self):
        a=synthetic(1000)[:,150:650]
        b,info=median_input(a,500,'median');np.testing.assert_allclose(a,b,atol=1e-6)
        with self.assertRaises(InputError):median_input(synthetic(),500,'median')

    def test_gustafsson_native_and_short_guard(self):
        a=synthetic();b,info=gustafsson_input(a,500,'native',simultaneous=True)
        self.assertEqual(b.shape,(8,4096));self.assertEqual(info['observed_samples_400hz'],4000)
        self.assertFalse(info['experimental']);self.assertTrue(np.all(b[:,:48]==0));self.assertTrue(np.all(b[:,-48:]==0))
        with self.assertRaises(InputError):gustafsson_input(a[:,:1250],500,'native',simultaneous=True)
        with self.assertRaises(InputError):gustafsson_input(a,500,'native',simultaneous=False)

    def test_exact_model_shapes_and_isolated_failures(self):
        good=ShapeCheckedModel((1,12,500));bad=ShapeCheckedModel((1,1,12,500),fail=True)
        r=predict_ischemia({'chongqing':good,'smartnet':bad},synthetic(),sample_rate_hz=500,units='mV',
                           input_kind='native',simultaneous=True,requested=['chongqing','smartnet'])
        self.assertEqual(r['status'],'PARTIAL');self.assertEqual(r['models']['chongqing']['status'],'SUCCESS')
        self.assertEqual(r['models']['smartnet']['status'],'FAILED_INFERENCE')

    def test_missing_weights_cannot_be_success(self):
        with tempfile.TemporaryDirectory() as folder:
            models,errors=load_ischemia_models(folder,requested=['chongqing'])
        self.assertIn('chongqing',errors)
        r=predict_ischemia(models,synthetic(),sample_rate_hz=500,units='mV',input_kind='native',simultaneous=True,
                           requested=['chongqing'],load_errors=errors)
        self.assertEqual(r['status'],'FAILED');self.assertEqual(r['models']['chongqing']['status'],'UNAVAILABLE')

    def test_rhythm_rows_never_overwrite_short_leads(self):
        layout=layout_definition('standard_3x4_with_r1');rows=np.stack([np.full(5000,v) for v in [1.,2.,3.,99.]])
        a=short_canonical_from_rows(rows,layout)
        self.assertEqual(a[1,0],2.);self.assertTrue(np.isnan(a[1,1250:]).all())
        self.assertNotIn(99.,a[np.isfinite(a)])

class CLIRegressionTests(unittest.TestCase):
    def test_failure_clears_stale_predictions_and_checkpoints_manifest(self):
        from run_all_ecg import main
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'record.npy';np.save(source,synthetic())
            output=root/'results';case=output/'record';case.mkdir(parents=True)
            (case/'ischemia_predictions.json').write_text('{"status":"SUCCESS"}')
            with patch('run_all_ecg.load_ischemia_models',return_value=({},{})):
                code=main(['--input',str(source),'--output',str(output),'--models','chongqing'])
            self.assertEqual(code,1)
            self.assertFalse((case/'ischemia_predictions.json').exists())
            self.assertEqual(json.loads((case/'qc.json').read_text())['status'],'FAILED')
            self.assertIn('FAILED',(output/'master_ecg_ischemia_manifest.csv').read_text())

    def test_native_cli_preserves_metadata_and_saves_actual_model_array(self):
        from run_all_ecg import main
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'record.npy';np.save(source,synthetic())
            source.with_suffix('.json').write_text(json.dumps({'input_kind':'native','units':'mV',
                                                               'sample_rate_hz':500,'simultaneous':True}))
            output=root/'results';model=ShapeCheckedModel((1,12,500))
            with patch('run_all_ecg.load_ischemia_models',return_value=({'chongqing':model},{})):
                code=main(['--input',str(source),'--output',str(output),'--models','chongqing'])
            self.assertEqual(code,0)
            self.assertEqual(np.load(output/'record'/'median_500hz_mv.npy').shape,(12,500))
            self.assertEqual(json.loads((output/'record'/'ischemia_predictions.json').read_text())['status'],'SUCCESS')

if __name__=='__main__':unittest.main()
