#!/usr/bin/env python3
"""OpenECG extraction with explicit QC and model-specific input adapters."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image, ImageOps
import torch
import yaml
from ecg_adapters import LEADS, InputError
from model_inference import MODEL_NAMES, load_ischemia_models, predict_ischemia
from audit import render_comparison_audit, render_waveform_overlay

ROOT = Path(__file__).resolve().parent
DEFAULT_DIGITIZER = Path('/workspace/Open-ECG-Digitizer')
if not DEFAULT_DIGITIZER.exists():
    DEFAULT_DIGITIZER = Path('/data/data/com.termux/files/home/ecg-ai/Electrocardiogram-Digitization')


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def layout_definition(name):
    if name.startswith('standard_3x4'):
        rows = [['I','aVR','V1','V4'], ['II','aVL','V2','V5'], ['III','aVF','V3','V6']]
    elif name.startswith('standard_6x2'):
        rows = [[LEADS[i], LEADS[i + 6]] for i in range(6)]
    elif name == 'standard_12x1':
        rows = LEADS.copy()
    else:
        raise InputError('Unsupported layout name; supply an explicit layout definition in metadata')
    rhythm = int(name.rsplit('_r', 1)[1]) if '_r' in name else 0
    return {'layout': {'rows': len(rows), 'cols': len(rows[0]) if isinstance(rows[0], list) else 1},
            'leads': rows, 'rhythm_leads': ['Any'] * rhythm}


def short_canonical_from_rows(rows_uv, layout):
    if rows_uv.ndim != 2 or rows_uv.shape[0] != len(layout['leads']) + len(layout.get('rhythm_leads', [])):
        raise InputError('Extracted row count disagrees with layout')
    width = rows_uv.shape[1];cols = int(layout['layout']['cols'])
    result = np.full((12, width), np.nan)
    for row_index, row in enumerate(layout['leads']):
        for col, label in enumerate(row if isinstance(row, list) else [row]):
            name = label.lstrip('-')
            if name not in LEADS:continue
            lo = col * (width // cols);hi = (col + 1) * (width // cols) if col < cols - 1 else width
            result[LEADS.index(name), lo:hi] = (-1 if label.startswith('-') else 1) * rows_uv[row_index, lo:hi]
    # Do not overwrite short channels using unverified automatic rhythm-strip matching.
    return result


def load_digitizer(repo, device, resample_size):
    sys.path.insert(0, str(repo))
    from src.config.default import get_cfg
    from src.model.inference_wrapper import InferenceWrapper
    cfg = get_cfg(str(repo / 'src/config/inference_wrapper.yml'))
    cfg.MODEL.KWARGS.device = device
    cfg.MODEL.KWARGS.resample_size = resample_size
    nested = cfg.MODEL.KWARGS.config
    nested.SEGMENTATION_MODEL.weight_path = str(repo / 'weights/unet_weights_07072025.pt')
    nested.LAYOUT_IDENTIFIER.config_path = str(repo / 'src/config/lead_layouts_all.yml')
    nested.LAYOUT_IDENTIFIER.unet_config_path = str(repo / 'src/config/lead_name_unet.yml')
    nested.LAYOUT_IDENTIFIER.unet_weight_path = str(repo / 'weights/lead_name_unet_weights_07072025.pt')
    nested.LAYOUT_IDENTIFIER.KWARGS.device = device
    nested.LAYOUT_IDENTIFIER.KWARGS.possibly_flipped = False
    layouts = yaml.safe_load(Path(nested.LAYOUT_IDENTIFIER.config_path).read_text())
    return InferenceWrapper(**cfg.MODEL.KWARGS).eval(), layouts


def digitize(path, digitizer, layouts, args, output):
    with Image.open(path) as im:
        arr = np.array(ImageOps.exif_transpose(im).convert('RGB'), copy=True)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)[None].float().to(args.device)
    with torch.inference_mode():
        got = digitizer(tensor, args.layout_filter)
    # An overlay is saved even when layout/coverage QC subsequently fails.
    raw_rows = got['signal']['raw_lines'].detach().cpu().numpy()
    x_offset = getattr(digitizer.signal_extractor, 'last_x_offset', None)
    if x_offset is None:
        raise InputError('Digitizer is missing trace coordinate patch; rebuild the updated image')
    render_waveform_overlay(got['aligned']['image'].numpy(), raw_rows, x_offset, output / 'waveform_overlay.png')
    cost = float(got['signal']['layout_matching_cost'])
    if not np.isfinite(cost) or cost > args.max_cost:
        raise InputError(f'Layout matching failed: cost={cost}')
    sx, sy = float(got['pixel_spacing_mm']['x']), float(got['pixel_spacing_mm']['y'])
    if not np.isfinite([sx, sy]).all() or not (.005 < sx < 1 and .005 < sy < 1 and .5 < sx / sy < 2):
        raise InputError('Invalid physical grid calibration')
    span = float(got.get('horizontal_span_mm', float('nan')))
    if not np.isfinite(span) or span <= 0:
        raise InputError('Missing measured paper span; refusing to assume a ten-second page')
    layout_name = str(got['layout_name']);layout = layouts[layout_name]
    rows = got['signal']['lines'].detach().cpu().numpy().astype(np.float64) * (10 / args.gain)
    # Upstream normalization uses average grid spacing; restore the vertical calibration.
    rows *= float(got['pixel_spacing_mm']['average_pixel_per_mm']) * sy
    canonical = short_canonical_from_rows(rows, layout)
    fs = (canonical.shape[1] - 1) / (span / args.paper_speed)
    meta = {'input_kind': 'canonical', 'units': 'uV', 'sample_rate_hz': fs, 'layout_name': layout_name,
            'layout': layout, 'layout_cost': cost, 'tier': 'easy' if cost <= .25 else 'moderately_difficult',
            'pixel_spacing_mm': {'x': sx, 'y': sy}, 'paper_speed_mm_s': args.paper_speed,
            'gain_mm_mv': args.gain, 'measured_span_mm': span, 'simultaneous': False,
            'rhythm_assignment': 'unverified rhythm rows excluded from classifier short channels'}
    np.save(output / 'canonical_uv.npy', canonical);np.save(output / 'extracted_rows_uv.npy', rows)
    render_comparison_audit(path, canonical, output / 'comparison_audit.png', fs=fs, layout=layout,
                            rows_uv=rows, paper_speed=args.paper_speed, gain=args.gain)
    return canonical, meta


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', type=Path, default=Path('results/ecg_digitized'))
    p.add_argument('--models', choices=MODEL_NAMES, nargs='+', default=list(MODEL_NAMES))
    p.add_argument('--models-dir', type=Path, default=ROOT / 'models')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--digitizer-repo', type=Path, default=DEFAULT_DIGITIZER)
    p.add_argument('--paper-speed', type=float, help='Verified image paper speed in mm/s')
    p.add_argument('--gain', type=float, help='Verified image gain in mm/mV')
    p.add_argument('--max-cost', type=float, default=.35)
    p.add_argument('--layout-filter')
    p.add_argument('--resample-size', type=int, default=2000)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--metadata', type=Path, help='JSON metadata for a single NPY input; default: same stem.json')
    p.add_argument('--age', type=float, help='Optional explicit override of Gustafsson age')
    p.add_argument('--sex', choices=['M', 'F'])
    p.add_argument('--gustafsson-short-mode', choices=['reject', 'pad', 'cycle-repeat'], default='reject')
    p.add_argument('--digitize-only', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.threads < 1 or not np.isfinite(args.max_cost) or not 0 < args.max_cost <= 1:
        raise SystemExit('Invalid thread count or maximum layout cost')
    torch.set_num_threads(args.threads)
    images = {'.png', '.jpg', '.jpeg', '.webp'}
    paths = [args.input] if args.input.is_file() else sorted(p for p in args.input.iterdir() if p.suffix.lower() in images | {'.npy'})
    if not paths:raise SystemExit('No supported image/NPY inputs found')
    if args.metadata and len(paths) != 1:raise SystemExit('--metadata is only valid with one input')
    if any(p.suffix.lower() in images for p in paths):
        if args.paper_speed is None or args.gain is None or not np.isfinite([args.paper_speed, args.gain]).all() or min(args.paper_speed, args.gain) <= 0:
            raise SystemExit('Images require explicit positive --paper-speed and --gain')
    args.output.mkdir(parents=True, exist_ok=True)
    loaded, load_errors = ({}, {}) if args.digitize_only else load_ischemia_models(args.models_dir, args.device, args.models)
    json_write(args.output / 'model_loading.json', {'requested': [] if args.digitize_only else args.models, 'loaded': list(loaded), 'errors': load_errors})
    digitizer = None;layouts = None;records = []
    for path in paths:
        started = time.perf_counter();output = args.output / path.stem;output.mkdir(parents=True, exist_ok=True)
        # Reruns must never expose a previous prediction as the result of a failed case.
        for artifact in ('ischemia_predictions.json', 'median_500hz_mv.npy', 'gustafsson_input_400hz_mv.npy',
                         'canonical_uv.npy', 'extracted_rows_uv.npy', 'input_metadata.json',
                         'waveform_overlay.png', 'comparison_audit.png', 'qc.json'):
            (output / artifact).unlink(missing_ok=True)
        record = {'case_id': path.stem, 'source_input': str(path), 'status': 'FAILED', 'output_dir': str(output)}
        meta = {};result = None
        try:
            if path.suffix.lower() in images:
                if digitizer is None:digitizer, layouts = load_digitizer(args.digitizer_repo, args.device, args.resample_size)
                waveform, meta = digitize(path, digitizer, layouts, args, output)
            elif path.suffix.lower() == '.npy':
                metadata_path = args.metadata or path.with_suffix('.json')
                meta = json.loads(metadata_path.read_text())
                if meta.get('lead_order', LEADS) != LEADS:
                    raise InputError('Input lead order differs from canonical order')
                if 'layout' not in meta and meta.get('layout_name'):
                    meta['layout'] = layout_definition(meta['layout_name'])
                waveform = np.load(path, allow_pickle=False)
            else:raise InputError('Only PNG/JPEG/WEBP and NPY inputs are implemented')
            json_write(output / 'input_metadata.json', meta)
            if args.digitize_only:
                if path.suffix.lower() not in images:raise InputError('--digitize-only requires image input')
                record['status'] = 'DIGITIZED_ONLY'
            else:
                result = predict_ischemia(loaded, waveform, args.device, sample_rate_hz=meta['sample_rate_hz'],
                     units=meta['units'], input_kind=meta['input_kind'], layout=meta.get('layout'),
                     simultaneous=meta.get('simultaneous') is True,
                     age=args.age if args.age is not None else meta.get('age'),
                     sex=args.sex if args.sex is not None else meta.get('sex'), requested=args.models,
                     load_errors=load_errors, short_mode=args.gustafsson_short_mode, artifact_dir=output)
                json_write(output / 'ischemia_predictions.json', result)
                record['status'] = result['status']
                for name, r in result['models'].items():
                    record[f'{name}_status'] = r['status'];record[f'{name}_score_omi'] = r.get('score_omi')
                    record[f'{name}_error'] = r.get('error', '')
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
        record['seconds'] = time.perf_counter() - started
        json_write(output / 'qc.json', {**record, 'input_metadata': meta})
        records.append(record)
        print(f"{path.name}: {record['status']}" + (f" ({record['error']})" if 'error' in record else ''), flush=True)
        # Checkpoint the manifest after every case, including partial/failed cases.
        fields = list(dict.fromkeys(k for r in records for k in r))
        with (args.output / 'master_ecg_ischemia_manifest.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields);writer.writeheader();writer.writerows(records)
    return 0 if all(r['status'] in ('SUCCESS', 'DIGITIZED_ONLY') for r in records) else 1

if __name__ == '__main__':
    sys.exit(main())
