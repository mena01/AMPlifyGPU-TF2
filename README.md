# AMPlifyGPU-TF2

An unofficial TensorFlow 2 and GPU-compatible inference port of the original [AMPlify](https://github.com/BirolLab/AMPlify) model.

This project modernizes the original AMPlify inference implementation for TensorFlow 2.x and modern NVIDIA GPUs while preserving the original pretrained models, architecture, preprocessing, ensemble strategy, and prediction workflow.

AMPlifyGPU-TF2 supports **GPU-accelerated and batched inference**, for faster large-scale peptide screening. In practical large-scale runs, the GPU-enabled workflow processed approximately **35 million peptides in ~34 hours**, compared with a previous CPU workflow requiring approximately **160 hours for ~8 million peptides**, corresponding to an observed throughput improvement of about **20×**.

**No retraining of the AMPlify models was performed.**

- **Original AMPlify:** Chenkai Li and the Birol Lab
- **TensorFlow 2 / GPU port:** Mena Khalaf
## Large-Scale GPU Performance

AMPlifyGPU-TF2 was also used for large-scale peptide screening.

In a previous CPU-based workflow, processing approximately **8 million peptides** required about **160 hours**, despite using **64 CPU threads** and a high-memory system with approximately **500 GB RAM**.

Using the TensorFlow 2 GPU-enabled implementation, approximately **35 million peptide sequences** were processed in about **34 hours**.

| Workflow | Sequences | Runtime | Approx. throughput |
|---|---:|---:|---:|
| Previous CPU workflow | ~8 million | ~160 h | ~50,000 peptides/h |
| AMPlifyGPU-TF2 GPU workflow | ~35 million | ~34 h | ~1.03 million peptides/h |

This corresponds to an observed throughput increase of approximately **20×** for the large-scale workflow.

The comparison reflects practical end-to-end runs rather than a controlled hardware-normalized benchmark, because the CPU and GPU workloads and computing environments were different.

## Validation Results

### Validation Against Original AMPlify

The TensorFlow 2 implementation was independently validated against the original **TensorFlow 1.12 / Keras 2.2.4** implementation using **835 peptides from the original AMPlify test dataset**.

Both implementations were executed on CPU using:

- the same 835 peptide sequences;
- the same original pretrained balanced ensemble weights;
- the same preprocessing workflow;
- the same model architecture;
- no retraining.

| Test | Result |
|---|---:|
| Peptides tested | 835 |
| Sequence ID agreement | 835/835 |
| Sequence agreement | 835/835 |
| Classification agreement | 835/835 (100%) |
| Classification mismatches | 0 |
| Ensemble probability MAE | 3.88 × 10⁻⁸ |
| Median absolute ensemble difference | 0 |
| Maximum ensemble probability difference | 7.60 × 10⁻⁷ |
| Ensemble predictions within 1 × 10⁻⁶ | 835/835 |
| Retraining performed | No |

All five individual ensemble members also reproduced the original predictions closely.

| Sub-model | Maximum absolute difference |
|---|---:|
| Model 1 | 2.43 × 10⁻⁶ |
| Model 2 | 1.35 × 10⁻⁶ |
| Model 3 | 1.76 × 10⁻⁶ |
| Model 4 | 1.77 × 10⁻⁶ |
| Model 5 | 2.13 × 10⁻⁶ |
### GPU Validation

The TensorFlow 2 implementation was also validated using an **NVIDIA GeForce RTX 3070 Ti**.

The same 835-peptide validation dataset was evaluated using GPU inference and compared with both the original TensorFlow 1 CPU implementation and the TensorFlow 2 CPU implementation.

| Comparison | Mean absolute difference | Maximum absolute difference | Classification agreement |
|---|---:|---:|---:|
| TF1 CPU vs TF2 CPU | 3.88 × 10⁻⁸ | 7.60 × 10⁻⁷ | **835/835 (100%)** |
| TF1 CPU vs TF2 GPU | 8.51 × 10⁻⁶ | 2.94 × 10⁻⁴ | **835/835 (100%)** |
| TF2 CPU vs TF2 GPU | 8.51 × 10⁻⁶ | 2.94 × 10⁻⁴ | **835/835 (100%)** |

GPU execution therefore preserved **100% of the AMP/non-AMP classifications** produced by the original AMPlify implementation on the validation dataset.

The very small numerical differences between CPU and GPU probability scores are consistent with expected floating-point differences between hardware execution backends and did not alter any classification.

The GPU-enabled implementation also supports batched inference, allowing substantially larger peptide datasets to be processed efficiently on modern NVIDIA GPUs.

These results show that the TensorFlow 2 implementation reproduces the original AMPlify inference results within floating-point tolerance on CPU, while GPU execution **preserves the same AMP/non-AMP classifications** with only minor numerical differences in probability scores.


Validation outputs are available in:

```text
validation_results/final/TF1_CPU_reference.tsv
validation_results/final/TF2_CPU_port.tsv
validation_results/final/TF2_GPU_port.tsv
validation_results/validation_report.md
```

Modern TensorFlow/Keras uses a different default recurrent activation. Therefore, the TensorFlow 2 port explicitly sets `recurrent_activation='hard_sigmoid'` to preserve the behavior expected by the original pretrained AMPlify weights.

This compatibility setting was essential for reproducing the original TensorFlow 1 predictions within floating-point tolerance.

## What This Port Changes

Compared with the original AMPlify implementation, this repository:
- ports inference from TensorFlow 1.x / legacy Keras to TensorFlow 2.x
- enables GPU-accelerated inference on modern NVIDIA GPUs
- supports batched inference for faster large-scale peptide screening
- preserves the original pretrained AMPlify weights
- preserves the original model architecture
- preserves the original preprocessing and prediction workflow
- does not retrain or redesign the AMPlify model
- provides direct TF1-vs-TF2 numerical validation

This repository does **not** introduce a new antimicrobial peptide prediction model. It is a modernization of the original AMPlify inference implementation.

## Architecture

The original AMPlify architecture is preserved:

```text
Input (200 × 20)
        ↓
     Masking
        ↓
BiLSTM (512 × 2)
        ↓
Multi-Head Attention
   (32 heads)
        ↓
     Dropout
        ↓
    Attention
        ↓
      Dense
        ↓
      Output
```

## Repository Structure

```text
AMPlifyGPU-TF2/
├── src/
│   ├── amplify_original/
│   │   ├── AMPlify.py
│   │   ├── layers.py
│   │   └── models/
│   └── amplify_tf2/
│       ├── AMPlify_tf2.py
│       └── layers.py
├── validation_results/
│   ├── final/
│   │   ├── TF1_CPU_reference.tsv
│   │   ├── TF2_CPU_port.tsv
│   │   └── TF2_GPU_port.tsv
│   └── validation_report.md
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

A Python 3.10 environment is recommended for the TensorFlow 2 implementation.

```bash
conda create -n amplify_tf2_gpu python=3.10
conda activate amplify_tf2_gpu
```

Install the required packages:

```bash
pip install -r requirements.txt
```

The tested TensorFlow/Keras versions are:

```text
TensorFlow 2.15.0
Keras 2.15.0
```

## Running AMPlifyGPU-TF2

Example:

```bash
python src/amplify_tf2/AMPlify_tf2.py \
  -s examples/sequences.fa \
  -m balanced \
  -od examples/test_run \
  -of tsv \
  -sub on \
  -v
```

### Main Options

```text
-s       Input FASTA file
-m       Model type: balanced or imbalanced
-od      Output directory
-of      Output format: txt or tsv
-sub     Output individual sub-model predictions: on/off
-att     Output attention scores: on/off
-v       Verbose output
```

## GPU Usage

When a supported NVIDIA GPU is available, TensorFlow 2 can use it automatically.

Check GPU detection with:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

For CPU-only execution:

```bash
CUDA_VISIBLE_DEVICES="" python src/amplify_tf2/AMPlify_tf2.py \
  -s examples/sequences.fa \
  -m balanced \
  -od examples/test_run \
  -of tsv \
  -sub on \
  -v
```

## Tested GPU Environment

The TensorFlow 2 implementation has been tested with:

- Python 3.10
- TensorFlow 2.15
- Keras 2.15
- NVIDIA GeForce RTX 3070 Ti
- NVIDIA driver 535.309.01
- CUDA 12.2
- cuDNN 8.9.7
- Linux

GPU compatibility may depend on the installed TensorFlow, NVIDIA driver, CUDA, and cuDNN versions.

## Original AMPlify

This project is derived from the original AMPlify software.

**Li C, Sutherland D, Hammond SA, et al.**  
AMPlify: attentive deep learning model for discovery of novel antimicrobial peptides effective against WHO priority pathogens.  
*BMC Genomics*. 2022;23:77.  
https://doi.org/10.1186/s12864-022-08310-4

**Li C, Warren RL, Birol I.**  
Models and data of AMPlify: a deep learning tool for antimicrobial peptide prediction.  
*BMC Research Notes*. 2023;16:11.  
https://doi.org/10.1186/s13104-023-06279-1

Original repository:

https://github.com/BirolLab/AMPlify

## Attribution

AMPlify was originally developed by **Chenkai Li and collaborators in the Birol Lab**.

This repository contains a TensorFlow 2 / GPU-compatible inference port developed by **Mena Khalaf**.

The purpose of this project is to improve compatibility with modern TensorFlow and NVIDIA GPU environments while preserving the original AMPlify model and prediction behavior.

This repository is currently an **unofficial derivative implementation** and is not an official release of the Birol Lab.

## License

This project is derived from AMPlify and follows the licensing terms of the original AMPlify project.

The original AMPlify software is distributed under the **GNU General Public License v3.0 (GPL-3.0)**.

This modified implementation is distributed under the same license.

See the `LICENSE` file for the full license terms.
