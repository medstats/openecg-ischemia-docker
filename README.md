# OpenECG Digitization & Myocardial Ischemia Deep Learning Suite

A production-grade, GPU-accelerated Docker container combining **Open-ECG Paper Digitization** with the leading **clinical deep learning foundation models for Myocardial Ischemia and Occlusion Myocardial Infarction (OMI)**.

* Container Registry: `ghcr.io/medstats/openecg-ischemia-docker:latest`
* Vast.ai Template: **`OpenECG-Ischemia-GPU`** (Template ID: `798702`)
* GitHub Repository: [`medstats/openecg-ischemia-docker`](https://github.com/medstats/openecg-ischemia-docker)

---

## 1. Integrated Models & Capabilities

| Component / Model | Type / Architecture | Input | Primary Outputs |
| :--- | :--- | :--- | :--- |
| **Open-ECG Digitizer** | Dual U-Net (Lead detection + Signal segmentation) | Paper ECG photo / scan (PNG, JPG, PDF) | Calibrated 12-lead waveforms at $500\,\text{Hz}$ ($\mu\text{V}$), shape $[12, 5000]$, plus dynamic 3-row / 4-row layout matching |
| **Gustafsson OMI Ensemble** | 5-Model Deep 1D CNN Ensemble | 12-lead voltage $[12, 5000]$ or $[12, T]$ | **Occlusion Myocardial Infarction (OMI)** probability $[0.0, 1.0]$, ensemble mean, standard deviation / model uncertainty |
| **Chongqing DenseNet** | 12-Lead 1D DenseNet-121 | 12-lead voltage $[12, 5000]$ or $[12, T]$ | **Acute Myocardial Ischemia / ACS** probability $[0.0, 1.0]$ |
| **ECG-SMART-NET** | Multi-Branch Residual Spatial-Temporal Net | 12-lead voltage $[12, 5000]$ or $[12, T]$ | **Ischemic ST-T & Infarction Pattern** probability $[0.0, 1.0]$ |
| **Visual Audit Generator** | Grid & Waveform Alignment Renderer | Original image + Digitized traces | Stacked comparison audit plot (`*_comparison_audit.png`) for visual verification |

---

## 2. Input Data Types

The pipeline accepts:
1. **Paper ECG Images / Scans:** `.png`, `.jpg`, `.jpeg`, `.pdf` (photographs, scanned records, digital exports).
2. **Pre-Digitized Raw Arrays:** `.npy`, `.csv`, `.mat` containing $[12, N]$ voltage traces. When raw signals are provided, the digitization stage is safely bypassed and fed directly into the ischemia ensemble.

Directory or single-file inputs are supported:
```text
ecg_inputs/
├── patient_001.png
├── patient_002.jpg
├── patient_003_raw.npy
└── folder_scans/
    └── patient_004.pdf
```

---

## 3. How to Elicit Inference

### Option A: The Master 1-Command Runner (`run_all_ecg.py`)
Run the entire end-to-end pipeline (digitization + audit plot + all ischemia models + manifest CSV):

```bash
python /workspace/run_all_ecg.py --input /workspace/ecg_inputs --output /workspace/results
```

For a single file:
```bash
python /workspace/run_all_ecg.py --input /workspace/ecg_inputs/patient_001.png --output /workspace/results
```

Select specific ischemia models:
```bash
python /workspace/run_all_ecg.py \
    --input /workspace/ecg_inputs \
    --output /workspace/results \
    --models gustafsson chongqing smartnet \
    --audit-dpi 200
```

---

### Option B: Standalone Model Execution

#### 1. Paper ECG Digitization Only (`pipeline.py`)
```bash
python /workspace/pipeline.py \
    --image /workspace/ecg_inputs/patient_001.png \
    --output-dir /workspace/results/patient_001
```
* Generates `signals.npy` ($[12, 5000]$ at $500\,\text{Hz}$) and layout metadata.

#### 2. Running Ischemia Evaluation on Existing `.npy` Signals
```bash
python /workspace/pipeline.py \
    --npy /workspace/signals.npy \
    --output-dir /workspace/results/ischemia_eval
```
* Evaluates Chongqing, SmartNet, and Gustafsson OMI models on the raw waveform and outputs `predictions.json`.

---

## 4. Stacked Comparison Audit Plots & Layout Geometry

A critical challenge in ECG digitization is paper layout fidelity:
* Standard $3\times 4$ ECG recordings without an extended rhythm strip (`standard_3x4`) contain **3 physical rows** of lead tracings on the paper:
  * Row 1: I, aVR, V1, V4 (2.5 seconds each)
  * Row 2: II, aVL, V2, V5 (2.5 seconds each)
  * Row 3: III, aVF, V3, V6 (2.5 seconds each)
* Forcing a 4th row when no rhythm strip exists causes artificial 2.5s truncated rows and visual distortion.
* `run_all_ecg.py` dynamically inspects the layout metadata:
  * **3-Row Layout:** Automatically renders a 3-row comparison grid matching the exact dimensions and order of the original paper sheet.
  * **4-Row Layout (`standard_3x4_with_rhythm`):** Renders the 3 short rows plus the continuous 10-second lead II rhythm strip below.
* The resulting audit file (`*_comparison_audit.png`) stacks the original scanned paper ECG directly above the digitized reconstruction, allowing immediate visual quality control before trusting deep learning predictions.

---

## 5. Output Manifest & Structure

Running `run_all_ecg.py` creates a structured output folder:

```text
results/
├── master_ecg_ischemia_manifest.csv    # Master summary table across all input files
├── patient_001/
│   ├── canonical_500hz_uv.npy          # Calibrated [12, 5000] array at 500 Hz (in uV)
│   ├── patient_001_comparison_audit.png# High-resolution visual audit image
│   ├── ischemia_predictions.json       # Detailed probabilities & model uncertainties
│   └── raw_digitization_meta.json      # Bounding boxes & lead detection confidences
└── patient_002/
    ├── canonical_500hz_uv.npy
    ├── patient_002_comparison_audit.png
    └── ischemia_predictions.json
```

### Manifest Schema (`master_ecg_ischemia_manifest.csv`)

| Column | Description | Example |
| :--- | :--- | :--- |
| `file_name` | Source ECG image or array filename | `patient_001.png` |
| `status` | Processing status (`SUCCESS` / `FAILED`) | `SUCCESS` |
| `detected_layout` | Paper geometry identified by OpenECG | `standard_3x4` |
| `chongqing_ischemia_prob` | Acute ischemia probability (Chongqing DenseNet) | `0.842` |
| `smartnet_ischemia_prob` | Ischemia / MI probability (ECG-SMART-NET) | `0.791` |
| `gustafsson_omi_prob` | Occlusion MI probability (5-model ensemble mean) | `0.885` |
| `gustafsson_model_std` | Inter-model standard deviation / uncertainty | `0.034` |
| `consensus_ischemia_flag` | High-confidence alert flag ($\ge 2$ models $>0.50$) | `TRUE` |
| `audit_plot_path` | Absolute path to visual comparison audit | `.../patient_001_comparison_audit.png` |
| `waveform_npy_path` | Absolute path to digitized $500\,\text{Hz}$ array | `.../canonical_500hz_uv.npy` |

---

## 6. Cloud Execution & Environments

### A. Vast.ai (Recommended for High-Throughput Batch Processing)
Launch an on-demand GPU instance using the pre-configured Vast.ai template:
1. Open Vast.ai and select Template ID: **`798702`** (**`OpenECG-Ischemia-GPU`**).
2. Or launch via the Vast.ai CLI:
   ```bash
   vastai create instance <MACHINE_ID> \
       --image ghcr.io/medstats/openecg-ischemia-docker:latest \
       --disk 30 \
       --args "python /workspace/run_all_ecg.py --input /workspace/data --output /workspace/results"
   ```

### B. Google Colab / Kaggle (Single-Cell Run)
To run directly inside a Colab or Kaggle GPU notebook without installing Docker:

```python
# 1. Clone repo & setup dependencies
!git clone https://github.com/medstats/openecg-ischemia-docker.git /content/ecg_suite
%cd /content/ecg_suite
!pip install -q timm scikit-image

# 2. Download model checkpoints from GitHub Release
!mkdir -p /content/ecg_suite/models/gustafsson
!curl -L -s -o /tmp/gustafsson.tar.gz https://github.com/medstats/openecg-ischemia-docker/releases/download/v1.0.0/gustafsson_models.tar.gz
!tar -xzf /tmp/gustafsson.tar.gz -C /content/ecg_suite/models/gustafsson && rm -f /tmp/gustafsson.tar.gz

# 3. Clone OpenECG Digitizer weights
!git clone https://github.com/Ahus-AIM/Open-ECG-Digitizer.git /content/ecg_suite/Open-ECG-Digitizer
!mkdir -p /content/ecg_suite/Open-ECG-Digitizer/weights
!curl -L -s -o /content/ecg_suite/Open-ECG-Digitizer/weights/lead_name_unet_weights_07072025.pt https://github.com/medstats/openecg-ischemia-docker/releases/download/v1.0.0/lead_name_unet_weights_07072025.pt
!curl -L -s -o /content/ecg_suite/Open-ECG-Digitizer/weights/unet_weights_07072025.pt https://github.com/medstats/openecg-ischemia-docker/releases/download/v1.0.0/unet_weights_07072025.pt

# 4. Run full inference
!python /content/ecg_suite/run_all_ecg.py --input /content/my_ecgs --output /content/results
```

---

## 7. Model Citations & Acknowledgments

* **Open-ECG Digitizer**: Ahus-AIM / CWK (`https://github.com/Ahus-AIM/Open-ECG-Digitizer`).
* **Gustafsson OMI Ensemble**: Gustafsson et al., *Artificial intelligence for the detection of occlusion myocardial infarction on 12-lead electrocardiograms*.
* **Chongqing DenseNet & ECG-SMART-NET**: Deep learning architectures for automated myocardial ischemia and ACS detection from standard 12-lead recordings.
