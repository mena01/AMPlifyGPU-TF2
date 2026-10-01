# AMPlifyGPU-TF2

An unofficial TensorFlow 2 GPU-compatible inference port of the original [AMPlify](https://github.com/BirolLab/AMPlify) model.

This project modernizes the original AMPlify inference implementation for TensorFlow 2.x and modern NVIDIA GPUs while preserving the original pretrained models and prediction workflow.

**No retraining of the AMPlify models was performed.**

- **Original AMPlify:** Chenkai Li and the Birol Lab
- **TensorFlow 2 / GPU port:** Mena Khalaf

## Validation Results

### Validation Against Original AMPlify

The TensorFlow 2 implementation was compared directly with the original TensorFlow 1 implementation using **835 peptides from the original AMPlify test dataset**.

| Test | Result |
|------|--------|
| Peptides tested | 835 |
| Classification agreement | 835/835 (100%) |
| Ensemble probability MAE | 0.0 |
| Maximum observed ensemble probability difference | 0.0 |
| Retraining performed | No |

These results demonstrate numerical agreement between the original TensorFlow 1 implementation and the TensorFlow 2 port on the tested AMPlify dataset and software/hardware environment.

Validation outputs are available in:

```text
validation_results/
```

### Example Output

```text
$ python AMPlify_tf2.py -s examples/sequences.fa -m balanced

Loading balanced models...
  Model 1: 100%
  Model 2: 100%
  Model 3: 100%
  Model 4: 100%
  Model 5: 100%

Predicting...
Sequence ID: teAMP0001
Sequence: QLPICGETCVLGGCYTPNCRCQYPICVR
Length: 28
Charge: 1
Probability score: 0.99987632
AMPlify_log_scaled_score: 31.6625
Prediction: AMP
```

## What This Port Changes

Compared with the original AMPlify implementation, this repository:

- ports inference from TensorFlow 1.x / legacy Keras to TensorFlow 2.x;
- enables inference on modern NVIDIA GPU environments;
- preserves the original pretrained AMPlify weights;
- preserves the original preprocessing and prediction workflow;
- does not retrain or redesign the AMPlify model;
- provides direct TF1-vs-TF2 numerical validation;
- supports efficient large-scale peptide screening.

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

## Tested Environment

The TensorFlow 2 implementation has been tested on:

- Python 3.10
- TensorFlow 2.15
- Keras 2.15
- NVIDIA RTX 3070 Ti
- CUDA 12.2
- cuDNN 8.9.7
- Linux

## GPU Requirements

A CUDA-compatible NVIDIA GPU is recommended for GPU inference.

The validated environment used:

- NVIDIA RTX 3070 Ti
- CUDA 12.2
- cuDNN 8.9.7
- TensorFlow 2.15

GPU compatibility may depend on the installed TensorFlow, CUDA, NVIDIA driver, and cuDNN versions.

## Installation

Create a Conda environment:

```bash
conda create -n amplify_tf2_gpu python=3.10
conda activate amplify_tf2_gpu
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install TensorFlow and Keras:

```bash
pip install tensorflow==2.15.0 keras==2.15.0
```

Verify GPU detection:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If the GPU environment is configured correctly, an NVIDIA GPU device should appear in the returned list.

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

This repository contains a TensorFlow 2 / GPU-compatible port developed by **Mena Khalaf**.

The purpose of this project is to improve compatibility with modern TensorFlow and NVIDIA GPU environments while preserving the original AMPlify model and prediction behavior.

This repository is currently an **unofficial derivative implementation** and is not an official release of the Birol Lab.

## License

This project is derived from AMPlify and follows the licensing terms of the original AMPlify project.

The original AMPlify software is distributed under the **GNU General Public License v3.0 (GPL-3.0)**.

This modified implementation is distributed under the same license.

See the `LICENSE` file for the full license terms.
