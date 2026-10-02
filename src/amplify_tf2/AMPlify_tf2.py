#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMPlify TF2 GPU Inference Port
Modern TensorFlow 2.x implementation preserving original architecture and weights
"""

import os
import argparse
from textwrap import dedent
import time
from Bio import SeqIO
import numpy as np
import pandas as pd
import tensorflow as tf
from layers import Attention, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Masking, Dense, LSTM, Bidirectional, Input, Dropout

MAX_LEN = 200


def one_hot_padding(seq_list, padding):
    """
    Generate features for amino acid sequences [one-hot encoding with zero padding].
    Input: seq_list: list of sequences, 
           padding: padding length, >= max sequence length.
    Output: one-hot encoding of sequences.
    """
    feat_list = []
    one_hot = {}
    aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    for i in range(len(aa)):
        one_hot[aa[i]] = [0] * 20
        one_hot[aa[i]][i] = 1
    for i in range(len(seq_list)):
        feat = []
        for j in range(len(seq_list[i])):
            feat.append(one_hot[seq_list[i][j]])
        feat = feat + [[0] * 20] * (padding - len(seq_list[i]))
        feat_list.append(feat)
    return np.array(feat_list, dtype=np.float32)


def build_amplify():
    """
    Build the complete model architecture matching original
    """
    inputs = Input(shape=(MAX_LEN, 20), name='Input')
    masking = Masking(mask_value=0.0, input_shape=(MAX_LEN, 20), name='Masking')(inputs)
    hidden = Bidirectional(LSTM(512, use_bias=True, dropout=0.5, return_sequences=True, recurrent_activation='hard_sigmoid'), name='Bidirectional-LSTM')(masking)
    hidden = MultiHeadAttention(head_num=32, activation='relu', use_bias=True,
                                return_multi_attention=False, name='Multi-Head-Attention')(hidden)
    hidden = Dropout(0.2, name='Dropout_1')(hidden)
    hidden = Attention(name='Attention')(hidden)
    prediction = Dense(1, activation='sigmoid', name='Output')(hidden)
    model = Model(inputs=inputs, outputs=prediction)
    return model


def build_attention():
    """
    Build the model architecture for attention output
    """
    inputs = Input(shape=(MAX_LEN, 20), name='Input')
    masking = Masking(mask_value=0.0, input_shape=(MAX_LEN, 20), name='Masking')(inputs)
    hidden = Bidirectional(LSTM(512, use_bias=True, dropout=0.5, return_sequences=True, recurrent_activation='hard_sigmoid'), name='Bidirectional-LSTM')(masking)
    hidden = MultiHeadAttention(head_num=32, activation='relu', use_bias=True,
                                return_multi_attention=False, name='Multi-Head-Attention')(hidden)
    hidden = Dropout(0.2, name='Dropout_1')(hidden)
    hidden = Attention(return_attention=True, name='Attention')(hidden)
    model = Model(inputs=inputs, outputs=hidden)
    return model


def load_multi_model(model_dir_list, architecture):
    """
    Load multiple models with the same architecture
    """
    model_list = []
    for i in range(len(model_dir_list)):
        model = architecture()
        model.load_weights(model_dir_list[i], by_name=True)
        model_list.append(model)
    return model_list


def ensemble(model_list, X, verbose=0, batch_size=256):
    """
    Ensemble prediction using manual host-side batching and a compiled TF graph.

    The model architecture and weights are unchanged.  tf.function removes the
    per-batch eager/Python execution overhead while keeping only one inference
    batch on the GPU at a time.
    """
    indv_pred = []
    n = len(X)

    input_spec = tf.TensorSpec(shape=(None, MAX_LEN, 20), dtype=tf.float32)

    for model_idx, model in enumerate(model_list):
        if verbose:
            print(f"Predicting model {model_idx + 1}/{len(model_list)}...")

        # Compile once per ensemble member.  The None batch dimension prevents
        # retracing for the final, shorter batch.  Do not enable XLA here: the
        # goal is a conservative execution optimization, not a model change.
        @tf.function(input_signature=[input_spec], reduce_retracing=True)
        def infer_batch(x):
            return model(x, training=False)

        model_predictions = []

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X_batch = tf.convert_to_tensor(X[start:end], dtype=tf.float32)
            pred_batch = infer_batch(X_batch).numpy().reshape(-1)
            model_predictions.append(pred_batch)

            if verbose and (start == 0 or end == n or end % 50000 == 0):
                print(f"  {end:,}/{n:,}")

        pred = np.concatenate(model_predictions)
        indv_pred.append(pred)

    indv_pred = np.array(indv_pred)
    ens_pred = np.mean(indv_pred, axis=0)
    return ens_pred, indv_pred

def get_attention_scores(indv_pred_list, attention_model_list, seq_list, X):
    """
    Get attention scores from the most confident model
    """
    attention_scores_list = []
    for i in range(len(attention_model_list)):
        attention_with_padding = attention_model_list[i].predict(X)
        attention_without_padding = []
        for j in range(len(seq_list)):
            attention_without_padding.append(attention_with_padding[j][0:len(seq_list[j])].flatten())
        attention_scores_list.append(attention_without_padding)
    
    ens_pred = np.mean(np.array(indv_pred_list), axis=0)
    confident_attention = []
    for i in range(len(ens_pred)):
        if ens_pred[i] > 0.5:
            indices = list(indv_pred_list[:, i])
            max_idx = indices.index(max(indices))
            confident_attention.append(attention_scores_list[max_idx][i])
        else:
            indices = list(indv_pred_list[:, i])
            min_idx = indices.index(min(indices))
            confident_attention.append(attention_scores_list[min_idx][i])
    
    return confident_attention


def proba_to_class_name(scores):
    """
    Turn prediction scores into real class names
    If score > 0.5, label the sample as AMP; else non-AMP.
    """
    classes = []
    for i in range(len(scores)):
        if scores[i] > 0.5:
            classes.append('AMP')
        else:
            classes.append('non-AMP')
    return np.array(classes)


def main():
    parser = argparse.ArgumentParser(description=dedent('''
        AMPlify TF2 GPU Inference Port
        ------------------------------------------------------
        Predict whether a sequence is AMP or not using TensorFlow 2.x
        '''),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-m', '--model', help="Balanced or imbalanced model (balanced by default, optional)",
                        choices=['balanced', 'imbalanced'], default='balanced', required=False)
    parser.add_argument('-s', '--seqs', help="Sequences for prediction, fasta file", required=True)
    parser.add_argument('-od', '--out_dir', help="Output directory (optional)", default=os.getcwd(), required=False)
    parser.add_argument('-of', '--out_format', help="Output format, txt or tsv (tsv by default, optional)",
                        choices=['txt', 'tsv'], default='tsv', required=False)
    parser.add_argument('-sub', '--sub_model',
                        help="Whether to output sub-model results, on or off (off by default, optional)",
                        choices=['on', 'off'], default='off', required=False)
    parser.add_argument('-att', '--attention',
                        help="Whether to output attention scores, on or off (off by default, optional)",
                        choices=['on', 'off'], default='off', required=False)
    parser.add_argument('-v', '--verbose', help="Verbose output (optional)", action='store_true')
    parser.add_argument('-bs', '--batch_size', type=int, default=256,
                        help="GPU inference batch size (default: 256)")

    args = parser.parse_args()

    # Get paths - use absolute paths from AMPlify_TF2_clean directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(os.path.dirname(base_dir), 'amplify_original', 'models', args.model)
    models = [os.path.join(model_dir, 'AMPlify_' + args.model + '_model_weights_' + str(i + 1) + '.h5')
              for i in range(5)]

    if args.verbose:
        print('\nLoading %s models...' % args.model)
        for n in range(5):
            print(models[n])

    # Load models
    out_model = load_multi_model(models, build_amplify)
    
    if args.attention == 'on':
        att_model = load_multi_model(models, build_attention)

    # Read input sequences
    aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    seq_id = []
    peptide = []
    for seq_record in SeqIO.parse(args.seqs, 'fasta'):
        seq_id.append(str(seq_record.id))
        peptide.append(str(seq_record.seq))

    # Filter valid sequences
    valid_ix = []
    for i in range(len(peptide)):
        if len(peptide[i]) <= 200 and len(peptide[i]) >= 2:
            if set(peptide[i]) - set(aa) == set() or (set(peptide[i][:-1]) - set(aa) == set() and peptide[i][-1] == '*'):
                valid_ix.append(i)

    # Select valid sequences
    peptide_valid = [peptide[i] if peptide[i][-1] != '*' else peptide[i][:-1] for i in valid_ix]

    # Generate input
    X_seq_valid = one_hot_padding(peptide_valid, MAX_LEN)

    # Ensemble prediction
    if args.verbose:
        print('\nPredicting...')
    
    y_score_valid, y_indv_list_valid = ensemble(out_model, X_seq_valid, verbose=args.verbose, batch_size=args.batch_size)
    y_class_valid = proba_to_class_name(y_score_valid)

    # Initialize result arrays
    y_score = []
    y_indv_list = [[] for n in range(5)]
    y_log_score = []
    y_class = []
    y_length = []
    y_charge = []

    if args.attention == 'on':
        attention_valid = get_attention_scores(y_indv_list_valid, att_model, peptide_valid, X_seq_valid)
        attention = []

    # Assemble full results
    valid_ix_set = set(valid_ix)
    ix = 0
    for i in range(len(peptide)):
        if i in valid_ix_set:
            y_score.append(str(round(y_score_valid[ix], 8)))
            if y_score_valid[ix] < 0.99999999:
                y_log_score.append(str(round(-10 * np.log10(1 - y_score_valid[ix]), 4)))
            else:
                y_log_score.append(str(round(-10 * np.log10(1 - 0.99999999), 4)))
            y_class.append(y_class_valid[ix])
            y_length.append(len(peptide[i]))
            y_charge.append(peptide[i].count('K') + peptide[i].count('R') - peptide[i].count('D') - peptide[i].count('E'))
            if args.sub_model == 'on':
                for n in range(5):
                    y_indv_list[n].append(str(round(y_indv_list_valid[n][ix], 8)))
            if args.attention == 'on':
                attention.append(list(attention_valid[ix]))
            ix = ix + 1
        else:
            y_score.append('NA')
            y_log_score.append('NA')
            y_class.append('NA')
            y_length.append('NA')
            y_charge.append('NA')
            if args.sub_model == 'on':
                for n in range(5):
                    y_indv_list[n].append('NA')
            if args.attention == 'on':
                attention.append('NA')

    # Build the human-readable text only when TXT output is requested.
    # For large TSV runs, avoiding per-sequence printing/string concatenation
    # removes substantial CPU and I/O overhead without changing predictions.
    out_txt = ''
    if args.out_format == 'txt':
        txt_parts = []
        for i in range(len(seq_id)):
            temp_txt = 'Sequence ID: ' + seq_id[i] + '\n' + 'Sequence: ' + peptide[i] + '\n' \
                       + 'Length: ' + str(y_length[i]) + '\n' + 'Charge: ' + str(y_charge[i]) + '\n'
            if args.sub_model == 'on':
                temp_txt += 'Sub-model probability scores: ' \
                            + ', '.join([y_indv_list[n][i] for n in range(5)]) + '\n'
            temp_txt += 'Probability score: ' + y_score[i] + '\n' \
                        + 'AMPlify_log_scaled_score: ' + y_log_score[i] + '\n' \
                        + 'Prediction: ' + y_class[i] + '\n'
            if args.attention == 'on':
                temp_txt += 'Attention: ' + str(attention[i]) + '\n'
            txt_parts.append(temp_txt + '\n')
        out_txt = ''.join(txt_parts)
        if args.verbose:
            print(out_txt)

    # Save results
    if args.out_format is not None:
        print('\nSaving results...')
        out_name = 'AMPlify_' + args.model + '_TF2_results_' + time.strftime('%Y%m%d%H%M%S', time.localtime())
        if (args.out_format).lower() == 'txt':
            out_name = out_name + '.txt'
            out_path = os.path.join(args.out_dir, out_name)
            if os.path.isfile(out_path):
                print('\nUnable to save! File already existed!')
            else:
                out = open(out_path, 'w')
                out.write(out_txt)
                out.close()
                print('\nResults saved as: ' + out_path)
        else:
            out_name = out_name + '.tsv'
            out_path = os.path.join(args.out_dir, out_name)
            if os.path.isfile(out_path):
                print('\nUnable to save! File already existed!')
            else:
                out = pd.DataFrame({'Sequence_ID': seq_id,
                                    'Sequence': peptide,
                                    'Length': y_length,
                                    'Charge': y_charge,
                                    'Probability_score': y_score,
                                    'AMPlify_log_scaled_score': y_log_score,
                                    'Prediction': y_class})
                if args.sub_model == 'on':
                    for n in range(5):
                        out.insert(loc=n + 4, column='Sub_model_%d_probability_score' % (n + 1), value=y_indv_list[n])
                if args.attention == 'on':
                    out.insert(loc=len(out.columns), column='Attention', value=attention)
                out.to_csv(out_path, sep='\t', index=False)
                print('\nResults saved as: ' + out_path)


if __name__ == "__main__":
    main()
