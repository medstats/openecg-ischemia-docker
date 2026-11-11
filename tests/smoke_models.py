"""Smoke-test all real checkpoints, correct shapes, and finite outputs on CPU."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import torch
from test_inputs import synthetic
from model_inference import load_ischemia_models,predict_ischemia,MODEL_NAMES

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models-dir',required=True);args=p.parse_args()
    torch.set_num_threads(2)
    models,errors=load_ischemia_models(args.models_dir,'cpu')
    if errors:raise RuntimeError(errors)
    r=predict_ischemia(models,synthetic(),sample_rate_hz=500,units='mV',input_kind='native',simultaneous=True,age=60,sex='M')
    if r['status']!='SUCCESS':raise RuntimeError(r)
    for name in MODEL_NAMES:
        score=r['models'][name]['score_omi'];assert 0<=score<=1
        print(name,score,r['models'][name]['input']['shape'])
