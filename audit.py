"""Audit actual extracted rows; never invent a rhythm label or recording duration."""
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ecg_adapters import LEADS


def render_comparison_audit(original, canonical_uv, out_path, *, fs, layout=None,
                            rows_uv=None, paper_speed=25, gain=10):
    count = len(layout['leads']) + len(layout.get('rhythm_leads', [])) if rows_uv is not None else 12
    fig, axes = plt.subplots(count, 1, figsize=(12, max(4, count * 1.15)), squeeze=False)
    duration = canonical_uv.shape[1] / fs
    for i, axis in enumerate(axes[:, 0]):
        if rows_uv is not None:
            y = rows_uv[i]
            label = ' / '.join(layout['leads'][i]) if i < len(layout['leads']) and isinstance(layout['leads'][i], list) else (
                layout['leads'][i] if i < len(layout['leads']) else f'Rhythm row {i-len(layout["leads"])+1}: label unverified')
        else:
            y, label = canonical_uv[i], LEADS[i]
        axis.plot(np.arange(len(y)) / fs, y / 1000, lw=.8, color='black')
        axis.set_ylabel(label, rotation=0, ha='right', fontsize=8)
        axis.set_xlim(0, duration)
        axis.grid(alpha=.25)
    axes[-1, 0].set_xlabel('Observed time (s); amplitudes in mV')
    fig.suptitle(f'Extracted rows | measured {fs:.2f} Hz | paper {paper_speed:g} mm/s, {gain:g} mm/mV')
    fig.tight_layout()
    stream = BytesIO()
    fig.savefig(stream, format='png', dpi=130)
    plt.close(fig)
    stream.seek(0)
    reconstructed = Image.open(stream).convert('RGB')
    with Image.open(original) as im:
        source = ImageOps.exif_transpose(im).convert('RGB')
    width = 1800
    def resize(im):
        return im.resize((width, round(im.height * width / im.width)), Image.Resampling.LANCZOS)
    source, reconstructed = resize(source), resize(reconstructed)
    canvas = Image.new('RGB', (width, source.height + reconstructed.height + 60), 'white')
    canvas.paste(source, (0, 30));canvas.paste(reconstructed, (0, source.height + 60))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), 'ORIGINAL ECG', fill='black')
    draw.text((12, source.height + 38), 'EXTRACTED WAVEFORMS - verify against source before interpreting scores', fill='black')
    canvas.save(out_path)


def render_waveform_overlay(aligned_image, raw_rows, x_offset, out_path):
    """Overlay actual extracted pixel coordinates on perspective-aligned source paper."""
    arr = np.asarray(aligned_image)
    if arr.ndim == 4:arr = arr[0]
    if arr.shape[0] == 3:arr = arr.transpose(1, 2, 0)
    arr = np.clip(arr, 0, 1)
    fig, ax = plt.subplots(figsize=(14, max(3, 14 * arr.shape[0] / arr.shape[1])))
    ax.imshow(arr)
    for i, row in enumerate(raw_rows):
        ax.plot(np.arange(len(row)) + x_offset, row, lw=.7, color=plt.get_cmap('tab10')(i % 10), alpha=.9)
    ax.set_title('Actual extracted trace over aligned source paper; colors identify rows')
    ax.axis('off');fig.tight_layout();fig.savefig(out_path, dpi=150);plt.close(fig)
