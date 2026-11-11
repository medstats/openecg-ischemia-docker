# OpenECG digitization and ischemia model inference

Research inference container: `ghcr.io/medstats/openecg-ischemia-docker:latest`.
It includes the Open-ECG Digitizer, Chongqing DenseNet, ECG-SMART-NET and the five-fold Gustafsson ensemble. These checkpoints and the paper-to-model transformations have not been clinically validated on this benchmark. Model scores are not interchangeable diagnostic probabilities.

## Corrected model inputs

| Model | Input passed to checkpoint | Preparation |
| --- | --- | --- |
| Chongqing | `(1,12,500)` in mV, 500 Hz | One-second QRS-aligned median, R at sample 240 |
| ECG-SMART-NET | `(1,1,12,500)` in mV, 500 Hz | The same median waveform |
| Gustafsson | `(1,8,4096)` in mV, 400 Hz | I, II, V1–V6; 0.8 Hz high-pass; ten seconds with centered zero padding; explicit age/sex |

The pipeline preserves observed duration when resampling. It never substitutes zeros for unobserved canonical lead slots or compresses an entire recording into one beat. Constructed medians require at least two complete beats per printed column, at least 75% observed coverage per lead, and a common continuous interval of at least one second. Flat leads, possible clipping and implausible amplitudes are rejected. Local median construction is experimental and differs from the vendor median algorithm used for training.

Gustafsson requires a finite native simultaneous recording of at least ten seconds by default. Ordinary 3×4 paper leads do not satisfy this requirement. Explicit research overrides `--gustafsson-short-mode pad` or `cycle-repeat` return an experimental-input warning. They do not recreate a native recording. Age and sex are never silently imputed.

## Run on paper images

Supported inputs are PNG, JPEG and WEBP, or a directory containing those images and NPY files (nonrecursive). Verify paper speed and gain from the source ECG and supply both explicitly:

```bash
docker pull ghcr.io/medstats/openecg-ischemia-docker:latest
docker run --rm -v "$PWD/data:/data" -v "$PWD/results:/results" \
  ghcr.io/medstats/openecg-ischemia-docker:latest \
  python /workspace/run_all_ecg.py --input /data --output /results \
  --paper-speed 25 --gain 10 --models chongqing smartnet --device cpu
```

Use `--gpus all` and `--device cuda` for a compatible NVIDIA host. Omit `--models` to request all three models; short paper recordings will then report Gustafsson as `FAILED_INPUT` and the case as `PARTIAL` if the other models succeed. For digitization alone add `--digitize-only`. `pipeline.py` is an alias for the same CLI.

OpenECG source is pinned and patched to retain physical span and trace coordinates. The measured paper width and supplied speed determine the actual sample frequency; the 5000-sample canonical array is not assumed to be 500 Hz. Amplitudes use detected vertical spacing and supplied gain. Layouts and extra rhythm rows are plotted as extracted. If automatic matching chooses an incomplete layout, inference fails. For a visually verified layout, pass a restriction such as `--layout-filter standard_3x4` and inspect the resulting overlay. Automatically assigned rhythm labels are unverified and never overwrite the short classifier channels.

## Run on existing arrays

An NPY must have shape `(12,N)`, ordered I, II, III, aVR, aVL, aVF, V1–V6. Provide a same-stem JSON file, or `--metadata path.json` for a single input. Native example `record.json`:

```json
{
  "input_kind": "native",
  "units": "mV",
  "sample_rate_hz": 500,
  "simultaneous": true,
  "age": 60,
  "sex": "F"
}
```

```bash
python /workspace/run_all_ecg.py --input /data/record.npy --output /results
```

Canonical paper arrays use `"input_kind":"canonical"`, explicit units (`uV` or `mV`), measured sample frequency and a `layout_name` such as `standard_3x4`, `standard_3x4_with_r1`, `standard_6x2` or `standard_12x1`. Other layouts need the full OpenECG layout dictionary in `layout`. Canonical slots retain NaNs outside their observed printed columns. Existing aligned arrays made by joining disjoint finite intervals are unsuitable; use their canonical originals. Vendor medians use `"input_kind":"median"` and must already be `(12,500)` at 500 Hz; they bypass median construction and cannot be used for Gustafsson.

PDF, CSV and MAT inputs are not implemented. No unit guessing or lead synthesis is performed.

## Outputs and failures

Each case has `input_metadata.json`, `qc.json` and, when model inference is attempted, `ischemia_predictions.json`. Image cases also save `canonical_uv.npy`, `extracted_rows_uv.npy`, `waveform_overlay.png` (actual traces on aligned source paper), and `comparison_audit.png`. Inspect these overlays before interpreting scores. Actual model arrays are saved as `median_500hz_mv.npy` and/or `gustafsson_input_400hz_mv.npy` when applicable.

The root contains `model_loading.json` and an incrementally checkpointed `master_ecg_ischemia_manifest.csv`. Per-model statuses are `SUCCESS`, `FAILED_INPUT`, `FAILED_INFERENCE` or `UNAVAILABLE`; successful entries include `score_omi` and input construction details. Case status is `SUCCESS`, `PARTIAL`, `FAILED`, or `DIGITIZED_ONLY`. A missing checkpoint is never reported as successful. Exit code is 1 if any case fails or is partial, while successful results remain available. No unvalidated voting threshold or consensus diagnosis is produced.

## Build and verification

```bash
docker build -t openecg-ischemia .
python -m unittest discover -s tests -v
python tests/smoke_models.py --models-dir /workspace/models
```

The Docker build pins dependencies, verifies downloaded weight checksums, runs regression tests and all three real-checkpoint CPU smoke tests, and initializes the digitizer before publishing. Gustafsson's original training checkpoints contain NumPy metadata and require trusted pickle loading; the bundled release archive is checksum verified. Only use trusted replacement checkpoints. GitHub Actions publishes `latest` and the full source commit tag after the build checks pass.

Upstream digitizer: [Ahus-AIM/Open-ECG-Digitizer](https://github.com/Ahus-AIM/Open-ECG-Digitizer). Gustafsson architecture and configuration are included under `models/gustafsson` with their original author information.
