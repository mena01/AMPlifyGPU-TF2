# AMPlify TF1 vs TF2 Validation

The TensorFlow 2 implementation was validated against the original
TensorFlow 1.12 / Keras 2.2.4 implementation using 835 peptides from
the original AMPlify test dataset.

Both implementations were executed on CPU using the same pretrained
balanced ensemble weights and identical input sequences.

## Results

- Sequences compared: 835
- Sequence ID agreement: 835 / 835
- Sequence agreement: 835 / 835
- Classification agreement: 835 / 835 (100%)
- Classification mismatches: 0
- Ensemble probability MAE: 3.88 × 10^-8
- Median absolute ensemble difference: 0
- Maximum ensemble probability difference: 7.60 × 10^-7
- All 835 ensemble predictions agreed within 1 × 10^-6
- No retraining was performed
- Original pretrained AMPlify weights were used

## Sub-model agreement

Maximum absolute sub-model differences:

- Model 1: 2.43 × 10^-6
- Model 2: 1.35 × 10^-6
- Model 3: 1.76 × 10^-6
- Model 4: 1.77 × 10^-6
- Model 5: 2.13 × 10^-6

## TensorFlow 2 compatibility

The original AMPlify implementation uses legacy Keras 2.2.4 LSTM
behavior.

To preserve compatibility with the original pretrained weights, the
TensorFlow 2 implementation explicitly uses:

    recurrent_activation='hard_sigmoid'

Without this setting, modern TensorFlow/Keras uses a different LSTM
recurrent activation default and produces different inference results.

## Conclusion

The TensorFlow 2 implementation reproduces the original AMPlify
inference results within floating-point tolerance while preserving the
original pretrained weights and model architecture.

Validation files:

- final/TF1_CPU_reference.tsv
- final/TF2_CPU_port.tsv
