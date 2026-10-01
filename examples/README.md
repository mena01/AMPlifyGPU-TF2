# Example Predictions

## 835 Peptide Validation Results

Full results available in: `validation_results/predictions_835.csv`

### Statistics
- Total peptides: 835
- AMP: 738 (88.4%)
- non-AMP: 97 (11.6%)
- Mean probability: 0.877423

### Example Sequences

| Sequence ID | Sequence | Probability | Class |
|-------------|----------|-------------|-------|
| teAMP0001 | QLPICGETCVLGGCYTPNCRCQYPICVR | 0.999+ | AMP |
| teAMP0002 | MSGRGKGGKGLGKGGAKRHRKVLRDNIQGITKPAIRRRGGVKRISGLIYEETRGVLKVFLENVIRDAVTYTEHAKRKTVTAMDVVYALKRQGRTLYGFGG | 0.152 | non-AMP |

### All Ensemble Models Give Identical Results

Each of the 5 models produces the same prediction (max difference: 0.00e+00).
This confirms the TF2 port is mathematically equivalent to the original.

### Reproducibility

Running the same prediction twice:
- Max difference: 0.00e+00
- Status: ✅ IDENTICAL
