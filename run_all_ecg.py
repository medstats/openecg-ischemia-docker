#!/usr/bin/env python3
"""
run_all_ecg.py - Unified Master Inference CLI for ECG Digitization & Ischemia Models
Digitization: OpenECG Digitizer (with dynamic 3-row vs 4-row comparison audit plot)
Ischemia Models: Chongqing DenseNet, ECG-SMART-NET, Gustafsson OMI Ensemble
"""
import os, sys, glob, time, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# OpenECG paths
REPO_DIR = "/workspace/Open-ECG-Digitizer" if os.path.exists("/workspace/Open-ECG-Digitizer") else "/data/data/com.termux/files/home/ecg-ai/Electrocardiogram-Digitization"
if os.path.exists(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

from models.chongqing.model import ChongqingDenseNet
from models.smartnet.model import ECGSMARTNET

FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if os.path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf") else "/data/data/com.termux/files/usr/lib/python3.13/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf"
FONT_REG_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" if os.path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") else "/data/data/com.termux/files/usr/lib/python3.13/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"

def render_comparison_audit(orig_img_path, canonical_uv, out_path, case_id, layout_name, cost, tier):
    is_3row = "standard_3x4" in layout_name and "r1" not in layout_name and "r2" not in layout_name and "r3" not in layout_name
    
    layout = [
        [('I', 0, 0.0, 2.5), ('aVR', 3, 2.5, 5.0), ('V1', 6, 5.0, 7.5), ('V4', 9, 7.5, 10.0)],
        [('II', 1, 0.0, 2.5), ('aVL', 4, 2.5, 5.0), ('V2', 7, 5.0, 7.5), ('V5', 10, 7.5, 10.0)],
        [('III', 2, 0.0, 2.5), ('aVF', 5, 2.5, 5.0), ('V3', 8, 5.0, 7.5), ('V6', 11, 7.5, 10.0)],
    ]
    
    speed, gain, fs, total_time_s = 25.0, 10.0, 500.0, 10.0
    width_mm = total_time_s * speed
    cal_margin_mm, left_margin_mm, right_margin_mm = 16.0, 4.0, 4.0
    total_width_mm = left_margin_mm + width_mm + cal_margin_mm + right_margin_mm
    
    if is_3row:
        row_height_mm = 28.0
        top_margin_mm, bottom_margin_mm = 4.0, 8.0
        total_height_mm = top_margin_mm + 3 * row_height_mm + bottom_margin_mm
        baselines = [
            bottom_margin_mm + row_height_mm * 2.5,
            bottom_margin_mm + row_height_mm * 1.5,
            bottom_margin_mm + row_height_mm * 0.5,
        ]
    else:
        row_height_mm = 30.0
        top_margin_mm, bottom_margin_mm = 6.0, 10.0
        total_height_mm = top_margin_mm + 4 * row_height_mm + bottom_margin_mm
        baselines = [
            bottom_margin_mm + row_height_mm * 3.5,
            bottom_margin_mm + row_height_mm * 2.5,
            bottom_margin_mm + row_height_mm * 1.5,
            bottom_margin_mm + row_height_mm * 0.5,
        ]
        
    fig = plt.figure(figsize=(total_width_mm / 25.4, total_height_mm / 25.4), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, total_width_mm)
    ax.set_ylim(0, total_height_mm)
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.fill_between([0, total_width_mm], 0, total_height_mm, color='#fcfdff', zorder=0)
    for x in np.arange(0, total_width_mm + 0.1, 1.0):
        if round(x) % 5 != 0: ax.axvline(x, color='#d3e2f2', linewidth=0.45, zorder=1)
    for y in np.arange(0, total_height_mm + 0.1, 1.0):
        if round(y) % 5 != 0: ax.axhline(y, color='#d3e2f2', linewidth=0.45, zorder=1)
    for x in np.arange(0, total_width_mm + 0.1, 5.0): ax.axvline(x, color='#8baece', linewidth=0.85, zorder=2)
    for y in np.arange(0, total_height_mm + 0.1, 5.0): ax.axhline(y, color='#8baece', linewidth=0.85, zorder=2)
    
    num_samples = canonical_uv.shape[1]
    time_arr = np.linspace(0, total_time_s, num_samples, endpoint=False)
    
    for row_idx in range(3):
        base_y = baselines[row_idx]
        for name, lead_idx, t_start, t_end in layout[row_idx]:
            idx_start = int(round(t_start * fs))
            idx_end = min(int(round(t_end * fs)), num_samples)
            t_seg = time_arr[idx_start:idx_end]
            sig_uv = canonical_uv[lead_idx, idx_start:idx_end]
            x_mm = left_margin_mm + t_seg * speed
            y_mm = base_y + (sig_uv / 1000.0) * gain
            x_lead_start = left_margin_mm + t_start * speed
            ax.plot([x_lead_start, x_lead_start], [base_y - 2.5, base_y + 2.5], color='#0052cc', linewidth=1.2, zorder=4)
            ax.text(x_lead_start + 0.8, base_y + 1.2, name, color='#0052cc', fontsize=8, fontweight='bold', family='sans-serif', zorder=5)
            ax.plot(x_mm, y_mm, color='#05070a', linewidth=1.1, solid_capstyle='round', zorder=4)
            
        cal_x0 = left_margin_mm + width_mm + 4.0
        cal_x = [cal_x0, cal_x0 + 1.25, cal_x0 + 1.25, cal_x0 + 6.25, cal_x0 + 6.25, cal_x0 + 7.5]
        cal_y = [base_y, base_y, base_y + 10.0, base_y + 10.0, base_y, base_y]
        ax.plot(cal_x, cal_y, color='#0052cc', linewidth=1.1, zorder=4)
        
    if not is_3row:
        base_y_rhythm = baselines[3]
        sig_uv = canonical_uv[1, :]
        x_mm = left_margin_mm + time_arr * speed
        y_mm = base_y_rhythm + (sig_uv / 1000.0) * gain
        ax.plot([left_margin_mm, left_margin_mm], [base_y_rhythm - 2.5, base_y_rhythm + 2.5], color='#0052cc', linewidth=1.2, zorder=4)
        ax.text(left_margin_mm + 0.8, base_y_rhythm + 1.2, "II", color='#0052cc', fontsize=8, fontweight='bold', family='sans-serif', zorder=5)
        ax.plot(x_mm, y_mm, color='#05070a', linewidth=1.1, solid_capstyle='round', zorder=4)
        cal_x0 = left_margin_mm + width_mm + 4.0
        cal_x = [cal_x0, cal_x0 + 1.25, cal_x0 + 1.25, cal_x0 + 6.25, cal_x0 + 6.25, cal_x0 + 7.5]
        cal_y = [base_y_rhythm, base_y_rhythm, base_y_rhythm + 10.0, base_y_rhythm + 10.0, base_y_rhythm, base_y_rhythm]
        ax.plot(cal_x, cal_y, color='#0052cc', linewidth=1.1, zorder=4)
        
    tmp_dir = "/tmp" if os.path.exists("/tmp") else "/storage/emulated/0/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    temp_dig_path = os.path.join(tmp_dir, f"temp_dig_{os.getpid()}_{case_id}.png")
    fig.savefig(temp_dig_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches=None)
    plt.close(fig)
    
    im_orig = ImageOps.exif_transpose(Image.open(orig_img_path)).convert('RGB')
    im_dig = Image.open(temp_dig_path).convert('RGB')
    
    target_width = 2200
    def resize_w(img, tw):
        w, h = img.size
        return img.resize((tw, int(round(h * (tw / w)))), Image.Resampling.LANCZOS)
    
    im_orig_r = resize_w(im_orig, target_width)
    im_dig_r = resize_w(im_dig, target_width)
    
    banner_height = 54
    total_h = banner_height + im_orig_r.height + banner_height + im_dig_r.height
    comp = Image.new('RGB', (target_width, total_h), color=(245, 247, 250))
    draw = ImageDraw.Draw(comp)
    
    try:
        font_title = ImageFont.truetype(FONT_BOLD_PATH, 24)
        font_sub = ImageFont.truetype(FONT_REG_PATH, 20)
    except:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        
    draw.rectangle([0, 0, target_width, banner_height], fill=(24, 32, 47))
    fmt_str = " (3-Row Layout)" if is_3row else " (4-Row + Rhythm Layout)"
    draw.text((28, 14), f"ORIGINAL ECG RECORDING — {case_id.upper()}{fmt_str}", fill=(255, 255, 255), font=font_title)
    
    y_off = banner_height
    comp.paste(im_orig_r, (0, y_off))
    y_off += im_orig_r.height
    
    draw.rectangle([0, y_off, target_width, y_off + banner_height], fill=(16, 75, 130))
    draw.text((28, y_off + 14), "DIGITIZED RECONSTRUCTION (25 mm/s, 10 mm/mV — 500 Hz Canonical)", fill=(255, 255, 255), font=font_title)
    cost_str = f"Cost: {cost:.3f}" if isinstance(cost, (int, float)) else ""
    tier_str = f"Tier: {tier.capitalize()}" if tier else ""
    draw.text((target_width - 320, y_off + 16), f"[{cost_str} | {tier_str}]", fill=(180, 215, 255), font=font_sub)
    y_off += banner_height
    
    comp.paste(im_dig_r, (0, y_off))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    comp.save(out_path, quality=95)
    
    if os.path.exists(temp_dig_path):
        os.remove(temp_dig_path)

def load_ischemia_models(models_dir="/workspace/models", device="cuda"):
    models = {}
    
    # 1. Chongqing DenseNet
    chong_path = os.path.join(models_dir, "chongqing/best_chongqing_densenet.pth")
    if os.path.exists(chong_path):
        m1 = ChongqingDenseNet(in_channels=12, num_classes=2).to(device)
        m1.load_state_dict(torch.load(chong_path, map_location=device, weights_only=True))
        m1.eval()
        models['chongqing'] = m1
        
    # 2. SmartNet
    smart_path = os.path.join(models_dir, "smartnet/best_ecg_smartnet_full.pth")
    if os.path.exists(smart_path):
        m2 = ECGSMARTNET(num_classes=2, num_leads=12).to(device)
        m2.load_state_dict(torch.load(smart_path, map_location=device, weights_only=True))
        m2.eval()
        models['smartnet'] = m2
        
    return models

def predict_ischemia(models, canonical_uv, device="cuda"):
    arr = np.nan_to_num(canonical_uv, nan=0.0)
    arr_mv = arr / 1000.0
    preds = {}
    
    if 'chongqing' in models:
        inp = torch.from_numpy(arr_mv).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits = models['chongqing'](inp)
            preds['prob_chongqing_ischemia'] = float(F.softmax(logits, dim=1)[0, 1].item())
            
    if 'smartnet' in models:
        from scipy.interpolate import interp1d
        old_t = np.linspace(0, 10, arr_mv.shape[1])
        new_t = np.linspace(0, 10, 500)
        arr_500 = interp1d(old_t, arr_mv, axis=-1, fill_value="extrapolate")(new_t)
        inp2 = torch.from_numpy(arr_500).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits2 = models['smartnet'](inp2)
            preds['prob_smartnet_occlusion'] = float(F.softmax(logits2, dim=1)[0, 1].item())
            
    return preds

def main():
    parser = argparse.ArgumentParser(description="Unified ECG Digitization & Ischemia Master Runner")
    parser.add_argument('--input', type=str, required=True, help="Directory containing ECG image files (PNG/JPG/WEBP) or a single image file")
    parser.add_argument('--output', type=str, default="results/ecg_digitized", help="Output directory")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device (cuda or cpu)")
    parser.add_argument('--max-cost', type=float, default=0.35, help="Layout matching cost cutoff (default: 0.35)")
    args = parser.parse_args()

    out_base = Path(args.output)
    (out_base / "npy_canonical_uv").mkdir(parents=True, exist_ok=True)
    (out_base / "audit_plots/easy").mkdir(parents=True, exist_ok=True)
    (out_base / "audit_plots/moderately_difficult").mkdir(parents=True, exist_ok=True)
    (out_base / "qc_json").mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if input_path.is_file():
        image_files = [input_path]
    else:
        exts = ('*.png', '*.jpg', '*.jpeg', '*.webp')
        image_files = []
        for ext in exts: image_files.extend(list(input_path.glob(ext)))
        image_files.sort()

    print("="*75)
    print(f"UNIFIED ECG DIGITIZER & ISCHEMIA PIPELINE")
    print(f"Input Images Found : {len(image_files)}")
    print(f"Output Directory   : {out_base}")
    print(f"Compute Device     : {args.device}")
    print("="*75)

    if not image_files:
        print("No image files found in input path!")
        return

    # 1. Initialize Digitizer
    digitizer = None
    try:
        from src.config.default import get_cfg
        from src.model.inference_wrapper import InferenceWrapper
        cfg = get_cfg(os.path.join(REPO_DIR, "src/config/inference_wrapper.yml"))
        cfg.MODEL.KWARGS.device = args.device
        cfg.MODEL.KWARGS.resample_size = 2000
        digitizer = InferenceWrapper(**cfg.MODEL.KWARGS)
        print("✓ OpenECG Digitizer loaded successfully.")
    except Exception as e:
        print(f"[Warning] OpenECG Digitizer could not be initialized: {e}")

    # 2. Load Ischemia Models
    models_dir = "/workspace/models" if os.path.exists("/workspace/models") else os.path.join(os.path.dirname(__file__), "models")
    ischemia_models = load_ischemia_models(models_dir, device=args.device)
    print(f"✓ Loaded Ischemia Models: {list(ischemia_models.keys())}")

    records = []
    start_time = time.time()

    for idx, img_path in enumerate(image_files):
        cid = img_path.stem
        print(f"\n[{idx+1}/{len(image_files)}] Processing {cid}...")
        
        rec = {
            'case_id': cid,
            'source_image': str(img_path),
            'status': 'FAILED',
            'layout_name': '',
            'layout_cost': np.nan,
            'tier': 'failed',
            'canonical_npy_path': '',
            'comparison_audit_plot_path': '',
            'prob_chongqing_ischemia': np.nan,
            'prob_smartnet_occlusion': np.nan
        }

        if digitizer is None:
            records.append(rec)
            continue

        try:
            with Image.open(img_path) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                arr = np.array(im, copy=True)
            
            img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(args.device).float()
            if max(img_tensor.shape[-2:]) < 2000:
                scale = 2000 / max(img_tensor.shape[-2:])
                size = tuple(round(v * scale) for v in img_tensor.shape[-2:])
                img_tensor = torch.nn.functional.interpolate(img_tensor, size=size, mode='bilinear', align_corners=False)

            with torch.inference_mode():
                got = digitizer(img_tensor, None)

            canonical = got['signal']['canonical_lines']
            if canonical is None:
                raise ValueError("No canonical lines detected")

            canonical_uv = canonical.cpu().numpy().astype(np.float64)
            layout_name = str(got['layout_name'])
            cost = float(got['signal']['layout_matching_cost'])

            if cost > args.max_cost:
                rec['status'] = f'FAILED_COST_{cost:.3f}'
                records.append(rec)
                continue

            tier = 'easy' if cost <= 0.25 else 'moderately_difficult'
            rec['status'] = 'PASSED'
            rec['layout_name'] = layout_name
            rec['layout_cost'] = cost
            rec['tier'] = tier

            # Save canonical NPY
            npy_p = out_base / "npy_canonical_uv" / f"{cid}_canonical_500hz_uv.npy"
            np.save(npy_p, canonical_uv)
            rec['canonical_npy_path'] = str(npy_p)

            # Generate Stacked Comparison Audit Plot
            audit_p = out_base / "audit_plots" / tier / f"{cid}_comparison_audit.png"
            render_comparison_audit(img_path, canonical_uv, audit_p, cid, layout_name, cost, tier)
            rec['comparison_audit_plot_path'] = str(audit_p)
            print(f"  ✓ Digitized: Cost={cost:.3f} | Tier={tier} | Layout={layout_name}")

            # Run Ischemia Predictions
            preds = predict_ischemia(ischemia_models, canonical_uv, device=args.device)
            for k, v in preds.items():
                rec[k] = v
                print(f"  ✓ {k}: {v:.3f}")

        except Exception as e:
            rec['status'] = f"ERROR: {str(e)}"
            print(f"  X Error: {e}")

        records.append(rec)

    # Save Master Manifest CSV
    df = pd.DataFrame(records)
    manifest_p = out_base / "master_ecg_ischemia_manifest.csv"
    df.to_csv(manifest_p, index=False)
    
    elapsed = time.time() - start_time
    passed_count = sum(r['status'] == 'PASSED' for r in records)
    print("\n" + "="*75)
    print(f"BATCH COMPLETE in {elapsed:.1f}s ({elapsed/60:.2f} min)")
    print(f"Successfully Processed: {passed_count}/{len(image_files)} cases")
    print(f"Master Manifest CSV   : {manifest_p}")
    print("="*75)

if __name__ == '__main__':
    main()
