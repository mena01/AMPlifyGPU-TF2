## Validation Results

### Validation Against Original AMPlify

The TensorFlow 2 implementation was independently validated against the original **TensorFlow 1.12 / Keras 2.2.4** implementation using **835 peptides from the original AMPlify test dataset**.

The comparison used:

- the same 835 peptide sequences;
- the same original pretrained AMPlify model weights;
- the same preprocessing pipeline;
- the same five-model ensemble architecture;
- the same classification threshold.

### TF1 CPU vs TF2 CPU

The TensorFlow 2 CPU implementation reproduces the original TensorFlow 1 predictions to near machine precision.

| Metric | Result |
|---|---:|
| Number of peptides | 835 |
| Mean absolute probability difference | 3.88 × 10⁻⁸ |
| Median absolute probability difference | 0 |
| Maximum absolute probability difference | 7.60 × 10⁻⁷ |
| Predictions within 1 × 10⁻⁶ | 835 / 835 |
| Classification agreement | **835 / 835 (100%)** |
| Classification mismatches | **0** |

These results demonstrate that the TensorFlow 2 CPU implementation is numerically equivalent to the original TensorFlow 1 AMPlify inference implementation for the validated dataset.

### GPU Validation

The TensorFlow 2 implementation was also validated on an **NVIDIA GeForce RTX 3070 Ti** using TensorFlow 2.15.

GPU inference produced the same AMP/non-AMP classifications as both the original TF1 implementation and the TF2 CPU implementation.

| Comparison | Mean Absolute Difference | Maximum Absolute Difference | Classification Agreement |
|---|---:|---:|---:|
| TF1 CPU vs TF2 CPU | 3.88 × 10⁻⁸ | 7.60 × 10⁻⁷ | **835 / 835 (100%)** |
| TF1 CPU vs TF2 GPU | 8.51 × 10⁻⁶ | 2.94 × 10⁻⁴ | **835 / 835 (100%)** |
| TF2 CPU vs TF2 GPU | 8.51 × 10⁻⁶ | 2.94 × 10⁻⁴ | **835 / 835 (100%)** |

The small numerical differences observed during GPU execution are consistent with expected floating-point differences between CPU and GPU computation and did **not** change any classification in the 835-peptide validation set.

Therefore, the GPU-enabled TensorFlow 2 implementation preserves the predictive behavior of the original AMPlify model while enabling inference on modern NVIDIA GPUs.

### Important Notes

- No AMPlify models were retrained.
- The original pretrained weights are used directly.
- The five AMPlify ensemble sub-models remain unchanged.
- The TF2 port preserves the original model architecture, preprocessing, ensemble calculation, and prediction threshold.
- GPU execution may produce very small floating-point differences compared with CPU execution.
- In the validation dataset, these numerical differences resulted in **no classification changes**.

### Validation Hardware and Software

**Original implementation**

- TensorFlow 1.12
- Keras 2.2.4
- CPU inference

**TensorFlow 2 implementation**

- TensorFlow 2.15
- Keras 2.15
- Python 3.10
- NVIDIA GeForce RTX 3070 Ti
- GPU compute capability 8.6

The complete validation results are available in:

```text
validation_results/final/TF1_CPU_reference.tsv
validation_results/final/TF2_CPU_port.tsv
validation_results/final/TF2_GPU_port.tsv
