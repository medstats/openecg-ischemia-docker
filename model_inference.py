"""Load requested checkpoints explicitly and report per-model outcomes."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ecg_adapters import InputError, validate_waveform, median_input, gustafsson_input
from models.chongqing.model import ChongqingDenseNet
from models.smartnet.model import ECGSMARTNET

MODEL_NAMES = ('chongqing', 'smartnet', 'gustafsson')


def load_ischemia_models(models_dir, device='cpu', requested=MODEL_NAMES):
    root = Path(models_dir)
    loaded, errors = {}, {}
    for name in requested:
        try:
            if name in ('chongqing', 'smartnet'):
                filename = 'best_chongqing_densenet.pth' if name == 'chongqing' else 'best_ecg_smartnet_full.pth'
                cls = ChongqingDenseNet if name == 'chongqing' else ECGSMARTNET
                model = cls(num_classes=2)
                model.load_state_dict(torch.load(root / name / filename, map_location='cpu', weights_only=True), strict=True)
                loaded[name] = model.to(device).eval()
            elif name == 'gustafsson':
                from models.gustafsson.model import ECGModel
                folder = root / name
                cfg = argparse.Namespace(**json.loads((folder / 'model_config.json').read_text()))
                cfg.device = device
                folds = []
                for i in range(1, 6):
                    model = ECGModel(cfg)
                    # Published training checkpoints contain NumPy scalars as well as weights.
                    # Only load trusted checkpoints; Docker verifies the release archive SHA256.
                    checkpoint = torch.load(folder / f'model_{i}.pth', map_location='cpu', weights_only=False)
                    model.load_state_dict(checkpoint['model'], strict=True)
                    folds.append(model.to(device).eval())
                loaded[name] = {'folds': folds, 'config': cfg}
            else:
                raise ValueError(f'Unknown model {name}')
        except Exception as exc:
            errors[name] = f'{type(exc).__name__}: {exc}'
    return loaded, errors


def predict_ischemia(models, waveform, device='cpu', *, sample_rate_hz, units,
                     input_kind, layout=None, simultaneous=False, age=None, sex=None,
                     requested=MODEL_NAMES, load_errors=None, short_mode='reject', artifact_dir=None):
    """Never fill canonical NaNs; no whole-record resizing into median-beat models."""
    result = {'models': {}, 'status': 'FAILED'}
    load_errors = load_errors or {}
    try:
        x = validate_waveform(waveform, sample_rate_hz, units)
    except (ValueError, TypeError) as exc:
        result['error'] = str(exc)
        for name in requested:
            result['models'][name] = {'status': 'FAILED_INPUT', 'error': str(exc)}
        return result
    med = med_info = None
    med_error = None
    for name in requested:
        try:
            if name not in models:
                result['models'][name] = {'status': 'UNAVAILABLE', 'error': load_errors.get(name, 'Requested model is not loaded')}
                continue
            if name in ('chongqing', 'smartnet'):
                if med is None and med_error is None:
                    try:
                        med, med_info = median_input(x, sample_rate_hz, input_kind, layout, simultaneous)
                    except InputError as exc:
                        med_error = str(exc)
                if med_error:
                    raise InputError(med_error)
                arr = med[None] if name == 'chongqing' else med[None, None]
                tensor = torch.from_numpy(arr).to(device)
                with torch.inference_mode():
                    logits = models[name](tensor)
                    if logits.shape != (1, 2) or not torch.isfinite(logits).all():
                        raise ValueError('Nonfinite or unexpected binary classifier logits')
                    score = float(torch.softmax(logits, dim=1)[0, 1])
                output = {'status': 'SUCCESS', 'score_omi': score, 'input': med_info}
                if artifact_dir:
                    np.save(Path(artifact_dir) / 'median_500hz_mv.npy', med)
            else:
                if age is None or sex not in ('M', 'F') or not np.isfinite(age) or not 0 < age <= 120:
                    raise InputError('Gustafsson requires explicit age (0–120) and sex M/F; no silent demographic imputation')
                arr, info = gustafsson_input(x, sample_rate_hz, input_kind, layout, simultaneous, short_mode)
                tensor = torch.from_numpy(arr[None]).to(device)
                cfg = models[name]['config']
                tabular = torch.tensor([[float(sex == 'M'), (age - cfg.age_mean) / cfg.age_sd]], dtype=torch.float32, device=device)
                with torch.inference_mode():
                    fold_logits = torch.stack([m((tabular, tensor)) for m in models[name]['folds']])
                    if fold_logits.shape != (5, 1, 11) or not torch.isfinite(fold_logits).all():
                        raise ValueError('Nonfinite or unexpected Gustafsson ensemble logits')
                    logits = fold_logits.mean(dim=0)
                    probs = torch.softmax(logits[0, :10], dim=0)
                    ids = [i for i, label in enumerate(cfg.outcomes_cat) if '_omi_' in label]
                    if len(ids) != 6:
                        raise ValueError('Unexpected Gustafsson outcome labels')
                    fold_scores = torch.softmax(fold_logits[:, 0, :10], dim=1)[:, ids].sum(dim=1)
                output = {'status': 'SUCCESS', 'score_omi': float(probs[ids].sum()),
                          'fold_omi_scores': fold_scores.cpu().tolist(),
                          'fold_score_std': float(fold_scores.std(unbiased=False)), 'input': info}
                if artifact_dir:
                    np.save(Path(artifact_dir) / 'gustafsson_input_400hz_mv.npy', arr)
            result['models'][name] = output
        except InputError as exc:
            result['models'][name] = {'status': 'FAILED_INPUT', 'error': str(exc)}
        except Exception as exc:
            result['models'][name] = {'status': 'FAILED_INFERENCE', 'error': f'{type(exc).__name__}: {exc}'}
    successes = sum(r['status'] == 'SUCCESS' for r in result['models'].values())
    result['status'] = 'SUCCESS' if successes == len(requested) and successes else ('PARTIAL' if successes else 'FAILED')
    return result
