# Example Predictions

## 835-Peptide Validation Results

The TensorFlow 2 implementation was validated against the original TensorFlow 1 implementation using **835 peptides from the original AMPlify test dataset**.

Validation results are available in:

```text
validation_results/compare_tf1_tf2.csv
```

Example prediction outputs are available in:

```text
examples/predictions_835.csv
```

## Validation Summary

- Total peptides tested: 835
- Classification agreement: 835/835 (100%)
- Ensemble probability MAE: 0.0
- Maximum observed ensemble probability difference: 0.0
- No retraining was performed
- Original pretrained AMPlify weights were used

These results demonstrate numerical agreement between the original TensorFlow 1 implementation and the TensorFlow 2 port on the tested dataset and environment.

## Example Sequences

| Sequence ID | Sequence | Probability | Class |
|-------------|----------|-------------|-------|
| teAMP0001 | QLPICGETCVLGGCYTPNCRCQYPICVR | 0.999+ | AMP |
| teAMP0002 | MSGRGKGGKGLGKGGAKRHRKVLRDNIQGITKPAIRRRGGVKRISGLIYEETRGVLKVFLENVIRDAVTYTEHAKRKTVTAMDVVYALKRQGRTLYGFGG | 0.152 | non-AMP |

## Ensemble Model Behavior

AMPlify uses five pretrained sub-models in the balanced ensemble.

The five sub-models produce distinct probability scores for the same peptide, and their outputs are combined to produce the final AMPlify probability score.

For example:

```text
Sequence: KLLKLLKKLLKLLK

Sub-model probability scores:
0.9999993
0.99957126
0.9997376
0.9999981
1.0

Final probability score:
0.9998613
```

This confirms that the TensorFlow 2 implementation loads and evaluates the five pretrained ensemble members independently.

## Reproducibility

Repeated inference with the same pretrained models and input data produced the same final predictions in the tested environment.

The TF1-vs-TF2 validation showed:

- Classification agreement: 100%
- Ensemble probability MAE: 0.0
- Maximum observed ensemble probability difference: 0.0

## Running the Example

An example FASTA file is provided at:

```text
examples/sequences.fa
```

Run:

```bash
python src/amplify_tf2/AMPlify_tf2.py \
  -s examples/sequences.fa \
  -m balanced \
  -od examples/test_run \
  -of tsv \
  -sub on \
  -v
```

Generated outputs are written to:

```text
examples/test_run/
```

This directory is excluded from Git tracking through `.gitignore`.
