import os, sys, glob, time, json, argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor, as_completed

from models.chongqing.model import ChongqingDenseNet
from models.smartnet.model import ECGSMARTNET

FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

def render_comparison_audit(orig_img_path, canonical_uv, out_path, case_id, layout_name, cost, tier):
    is_3row = "standard_3x4" in layout_name and "r1" not in layout_name and "r2" not in layout_name and "r3" not in layout_name
    
    layout = [
        [('I', 0, 0.0, 2.5), ('aVR', 3, 2.5, 5.0), ('V1', 6, 5.0, 7.5), ('V4', 9, 7.5, 10.0)],
        [('II', 1, 0.0, 2.5), ('aVL', 4, 2.5, 5.0), ('V2', 7, 5.0, 7.5), ('V5', 10, 7.5, 10.0)],
        [('III', 2, 0.0, 2.5), ('aVF', 5, 2.5, 5.0), ('V3', 8, 5.0, 7.5), ('V6', 11, 7.5, 10.0)],
    ]
    
    speed = 25.0
    gain = 10.0
    fs = 500.0
    total_time_s = 10.0
    width_mm = total_time_s * speed
    cal_margin_mm = 16.0
    left_margin_mm = 4.0
    right_margin_mm = 4.0
    total_width_mm = left_margin_mm + width_mm + cal_margin_mm + right_margin_mm
    
    if is_3row:
        row_height_mm = 28.0
        top_margin_mm = 4.0
        bottom_margin_mm = 8.0
        total_height_mm = top_margin_mm + 3 * row_height_mm + bottom_margin_mm
        baselines = [
            bottom_margin_mm + row_height_mm * 2.5,
            bottom_margin_mm + row_height_mm * 1.5,
            bottom_margin_mm + row_height_mm * 0.5,
        ]
    else:
        row_height_mm = 30.0
        top_margin_mm = 6.0
        bottom_margin_mm = 10.0
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
        
    temp_dig_path = f"/tmp/dig_{os.getpid()}_{case_id}.png"
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

def load_ischemia_models(device="cuda"):
    models = {}
    
    # 1. Chongqing DenseNet
    chong_path = "/workspace/models/chongqing/best_chongqing_densenet.pth"
    if os.path.exists(chong_path):
        m1 = ChongqingDenseNet(in_channels=12, num_classes=2).to(device)
        m1.load_state_dict(torch.load(chong_path, map_location=device, weights_only=True))
        m1.eval()
        models['chongqing'] = m1
        
    # 2. SmartNet
    smart_path = "/workspace/models/smartnet/best_ecg_smartnet_full.pth"
    if os.path.exists(smart_path):
        m2 = ECGSMARTNET(num_classes=2, num_leads=12).to(device)
        m2.load_state_dict(torch.load(smart_path, map_location=device, weights_only=True))
        m2.eval()
        models['smartnet'] = m2
        
    return models

def predict_ischemia(models, canonical_uv, device="cuda"):
    # canonical_uv: (12, 5000) in microvolts
    # Replace NaNs with 0
    arr = np.nan_to_num(canonical_uv, nan=0.0)
    # Convert to mV
    arr_mv = arr / 1000.0
    
    preds = {}
    
    if 'chongqing' in models:
        # Expected: (batch, 12, seq_len)
        inp = torch.from_numpy(arr_mv).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits = models['chongqing'](inp)
            prob = F.softmax(logits, dim=1)[0, 1].item()
            preds['prob_chongqing_ischemia'] = prob
            
    if 'smartnet' in models:
        # Expected: (batch, 1, 12, 500)
        # Resample 5000 -> 500
        from scipy.interpolate import interp1d
        old_t = np.linspace(0, 10, arr_mv.shape[1])
        new_t = np.linspace(0, 10, 500)
        arr_500 = interp1d(old_t, arr_mv, axis=-1, fill_value="extrapolate")(new_t)
        inp2 = torch.from_numpy(arr_500).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits2 = models['smartnet'](inp2)
            prob2 = F.softmax(logits2, dim=1)[0, 1].item()
            preds['prob_smartnet_occlusion'] = prob2
            
    return preds
