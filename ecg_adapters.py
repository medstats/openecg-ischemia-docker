"""Explicit units, observed time, and checkpoint-specific ECG input construction."""
from fractions import Fraction
import math
import numpy as np
from scipy import signal

LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
PAPER_WARNING = ('Experimental reconstruction from short printed leads; this is not a native '
                 '10-second simultaneous ECG. Performance of this transformation is not validated.')

class InputError(ValueError):
    pass


def validate_waveform(waveform, sample_rate_hz, units):
    x = np.asarray(waveform, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != 12 or x.shape[1] < 250:
        raise InputError('Expected 12 leads in I,II,III,aVR,aVL,aVF,V1..V6 order and >=250 samples')
    if not np.isfinite(sample_rate_hz) or not 100 <= sample_rate_hz <= 5000:
        raise InputError('Explicit sample_rate_hz between 100 and 5000 is required')
    if units not in ('uV', 'mV'):
        raise InputError('Explicit units must be uV or mV; amplitude-based unit guessing is disabled')
    if np.isinf(x).any():
        raise InputError('Infinite samples are invalid')
    return x / 1000.0 if units == 'uV' else x.copy()


def longest_finite_interval(x):
    indices = np.flatnonzero(np.isfinite(x).all(axis=0))
    if not len(indices):
        raise InputError('No common finite interval in printed column')
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    run = max(runs, key=len)
    return int(run[0]), int(run[-1] + 1)


def resample_observed(x, fs, target_fs):
    """Change sample frequency while preserving elapsed time; never join missing runs."""
    ratio = Fraction(float(target_fs) / float(fs)).limit_denominator(100000)
    y = signal.resample_poly(x, ratio.numerator, ratio.denominator, axis=-1)
    n = int(math.floor((x.shape[1] - 1) * target_fs / fs)) + 1
    return y[:, :n]


def checked_segment(x):
    if not np.isfinite(x).all():
        raise InputError('Nonfinite samples inside selected interval')
    for i, row in enumerate(x):
        if np.std(row) < .001:
            raise InputError(f'Flat lead in column (local lead index {i})')
        if np.ptp(row) > 20:
            raise InputError(f'Implausible amplitude >20 mV in column (local lead index {i})')
        if max(np.mean(row == row.min()), np.mean(row == row.max())) > .02:
            raise InputError('Possible clipping: repeated extrema in selected interval')


def observed_groups(x_mv, fs, kind, layout=None, simultaneous=False, min_coverage=.75):
    if kind == 'native':
        if not simultaneous or not np.isfinite(x_mv).all():
            raise InputError('Native input requires finite simultaneous data and explicit simultaneous=true')
        groups = [(list(range(12)), 0, x_mv.shape[1])]
    elif kind == 'canonical':
        if not isinstance(layout, dict) or 'leads' not in layout:
            raise InputError('Canonical input needs its printed layout definition')
        cols = int(layout['layout']['cols'])
        if cols <= 0:
            raise InputError('Invalid printed layout')
        groups_by_col = {}
        seen = set()
        for row in layout['leads']:
            for col, label in enumerate(row if isinstance(row, list) else [row]):
                # Canonical channels already have polarity restored, including -aVR layouts.
                name = label.lstrip('-')
                if name in LEADS:
                    if name in seen:
                        raise InputError('Repeated short lead in layout')
                    seen.add(name)
                    groups_by_col.setdefault(col, []).append(LEADS.index(name))
        if seen != set(LEADS):
            raise InputError('All 12 printed short leads are required; no lead synthesis')
        width = x_mv.shape[1]
        groups = [(ids, col * (width // cols), (col + 1) * (width // cols) if col < cols - 1 else width)
                  for col, ids in sorted(groups_by_col.items())]
    else:
        raise InputError('Input kind must be canonical, native, or median')
    out = []
    for ids, lo, hi in groups:
        block = x_mv[ids, lo:hi]
        if (np.isfinite(block).mean(axis=1) < min_coverage).any():
            raise InputError('Less than 75% of a printed lead interval is observed')
        start, end = longest_finite_interval(block)
        if (end - start) / fs < 1.0:
            raise InputError('Common continuous printed interval is shorter than one second')
        segment = block[:, start:end]
        checked_segment(segment)
        out.append((ids, resample_observed(segment, fs, 500),
                    {'leads': [LEADS[i] for i in ids], 'source_start': lo + start,
                     'source_end_exclusive': lo + end, 'duration_sec': (end - start) / fs}))
    return out


def qrs_indices(x, fs=500):
    q = signal.sosfiltfilt(signal.butter(2, [8, 35], fs=fs, btype='bandpass', output='sos'), x, axis=1)
    scale = np.maximum(np.percentile(np.abs(q), 95, axis=1), .02)
    energy = np.mean((q / scale[:, None]) ** 2, axis=0)
    energy = np.convolve(energy, np.ones(35) / 35, mode='same')
    peaks, _ = signal.find_peaks(energy, distance=150, prominence=max(.05, float(energy.max()) * .12))
    v = np.sum((x - np.median(x, axis=1, keepdims=True)) ** 2, axis=0)
    return np.array(sorted(set(max(0, p - 25) + int(np.argmax(v[max(0, p - 25):min(x.shape[1], p + 26)]))
                               for p in peaks)), dtype=int)


def median_input(x_mv, fs, kind, layout=None, simultaneous=False, min_beats=2):
    if kind == 'median':
        if fs != 500 or x_mv.shape != (12, 500):
            raise InputError('A supplied vendor median must be (12,500) at 500 Hz')
        checked_segment(x_mv)
        return x_mv.astype(np.float32), {'method': 'supplied_vendor_median', 'shape': [12, 500],
                                        'sample_rate_hz': 500, 'units': 'mV'}
    groups = observed_groups(x_mv, fs, kind, layout, simultaneous)
    med = np.empty((12, 500), np.float32)
    details = []
    for ids, x, info in groups:
        x = signal.sosfiltfilt(signal.butter(3, [.5, 45], fs=500, btype='bandpass', output='sos'), x, axis=1)
        peaks = qrs_indices(x)
        beats = []
        for p in peaks:
            if p >= 240 and p + 260 <= x.shape[1]:
                beat = x[:, p - 240:p + 260].copy()
                beat -= np.median(beat[:, 180:220], axis=1, keepdims=True)
                beats.append(beat)
        if len(beats) < min_beats:
            raise InputError(f"{info['leads']}: only {len(beats)} complete beats; need {min_beats}")
        med[ids] = np.median(beats, axis=0)
        details.append({**info, 'qrs_indices': peaks.tolist(), 'complete_beats': len(beats)})
    return med, {'method': 'columnwise_QRS_aligned_median', 'shape': [12, 500], 'units': 'mV',
                 'sample_rate_hz': 500, 'qrs_anchor_sample': 240, 'columns': details,
                 'experimental': True, 'warning': PAPER_WARNING if kind == 'canonical' else
                 'Locally constructed median algorithm is not the original vendor algorithm.'}


def gustafsson_input(x_mv, fs, kind, layout=None, simultaneous=False, short_mode='reject'):
    if kind == 'median':
        raise InputError('Gustafsson needs a continuous recording, not a median beat')
    native = kind == 'native' and simultaneous and np.isfinite(x_mv).all() and x_mv.shape[1] / fs >= 10
    if not native and short_mode == 'reject':
        raise InputError('Gustafsson needs >=10 s simultaneous finite data; short-paper inference is disabled by default')
    if native:
        checked_segment(x_mv)
        x500 = resample_observed(x_mv[:, :int(round(10 * fs))], fs, 500)
    else:
        groups = observed_groups(x_mv, fs, kind, layout, simultaneous)
        n = min(x.shape[1] for _, x, _ in groups)
        x500 = np.empty((12, n))
        for ids, x, _ in groups:
            x500[ids] = x[:, :n]
    x = x500[[0, 1, 6, 7, 8, 9, 10, 11]]
    order, _ = signal.ellipord(.8 / 250, .2 / 250, .5, 40)
    sos = signal.iirfilter(order, .8 / 250, .5, 40, btype='high', ftype='ellip', output='sos')
    x = signal.sosfiltfilt(sos, x, padtype='constant', axis=-1)
    x400 = signal.resample_poly(x, 4, 5, axis=-1)
    observed_n = x400.shape[1]
    if not native and short_mode == 'cycle-repeat':
        peaks = qrs_indices(x500[[0, 1, 2]])
        if len(peaks) < 2:
            raise InputError('At least two QRS peaks required for cycle repetition')
        lo, hi = int(round(peaks[0] * .8)), int(round(peaks[-1] * .8))
        period = x400[:, lo:hi]
        if period.shape[1] < 100:
            raise InputError('Invalid repetition interval')
        x400 = np.tile(period, (1, int(np.ceil(4000 / period.shape[1]))))[:, :4000]
    if x400.shape[1] > 4096:
        raise InputError('Input exceeds 4096 samples; refusing temporal compression')
    padded = np.zeros((8, 4096), np.float32)
    start = (4096 - x400.shape[1]) // 2
    padded[:, start:start + x400.shape[1]] = x400
    return padded, {'method': 'native' if native else short_mode, 'shape': [8, 4096],
                    'units': 'mV', 'sample_rate_hz': 400, 'observed_samples_400hz': observed_n,
                    'experimental': not native, 'warning': None if native else PAPER_WARNING}
