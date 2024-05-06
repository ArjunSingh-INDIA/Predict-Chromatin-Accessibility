#!/usr/bin/env python3   

# Question: Infer cell-type specific chromatin activity solely from DNA sequence motifs using XGBoost. 
# Provided sequence motifs to featurize the whole genome and train a gradient boosted tree ensemble to predict ATAC-seq peaks. 
# Files used: reference genome, TF DNA binding motifs, Genomic subsequences (training and prediction)
 
import Bio.motifs as motifs
from Bio import SeqIO
import numpy as np
import xgboost as xgb
import argparse
from multiprocessing import Pool
from sklearn.model_selection import cross_val_score, KFold
from itertools import product


if __name__ == '__main__': 

    parser = argparse.ArgumentParser(description='Predict ATAC-seq data from sequence motifs')
    parser.add_argument('-g','--genome',help='reference genome FASTA file',required=True)
    parser.add_argument('-t','--train',help='training bed file with chromosome, start, end and label',required=True)
    parser.add_argument('-m','--motifs',help='file of motifs to use as features',required=True)
    parser.add_argument('-p','--predict',help='training bed file for prediction with chromosome, start, end', required=True)
    parser.add_argument('-o','--output_file',help='Output predictions',required=True)

    args = parser.parse_args()
    
    # Reading the reference genome
    chromos = {}
    for seqr in SeqIO.parse(args.genome,'fasta'):
        chromos[seqr.name] = seqr.seq

    # Reading TF DNA motifs to create a list of PSSMs (Position Specific Scoring Matrix)
    def read_motifs(motif_file):
        motifss = []
        with open(motif_file) as handle:
            # Reading motifs in the pfm-four-columns format
            for m in motifs.parse(handle, "pfm-four-columns"):
                psssm = m.pssm
                motifss.append(psssm)         
        return motifss
    
    # Reading the motifs
    motif_object  =  read_motifs(args.motifs)

    # Featurize the sequence using the motifs
    def featurize_sequence(sequence):
        motifss = motif_object
        motifss_len = len(motifss)
        features = np.zeros((1, motifss_len))
        for i in range(motifss_len):
            n = len(sequence)
            m = motifss[i].length
            # Define the scoring array for each motif
            scores = np.empty(n - m + 1, np.float32)
            c = motifss[i]
            #  log-odds score is the likelihood of finding a specific DNA base at a particular motif position relative to its general frequency.
            # https://github.com/biopython/biopython/blob/master/Bio/motifs/matrix.py 
            logodds = np.array(
                [[c[letter][j] for letter in "ACGT"] for j in range(m)], float
            )
            # The sequence and logodds is used to calcuate the score for each motif
            motifs._pwm.calculate(bytes(sequence), logodds, scores)
            max_score = np.max(scores)
            mean_score = np.mean(scores)
            if np.isinf(scores).any():
                features[0, i] = max_score
            else:
                features[0, i] = mean_score
        return features
    
    # Running the featurize_sequence in parallel
    def featurize_in_parallel(sequences,num_cores): 
        return Pool(num_cores).map(featurize_sequence, sequences)
    
    # Read the Genomic subsequences for training and testing from the genome
    def read_bed_file(bed_file, genome):
        sequences = []
        with open(bed_file, 'r') as file:
            for line in file:
                vals = line.split()
                # reading the chromosome, start and end positions of the genomic subsequence
                c = vals[0]
                start = int(vals[1])
                end = int(vals[2])
                sequence = genome[c][start:end]
                sequences.append(sequence)
        return sequences

    # Read the Genomic subsequences for training from the genome
    train_sequences = read_bed_file(args.train, chromos)
    # Reading the labels for training
    train_labels = [line.split()[3] for line in open(args.train, 'r')]
     # Read the Genomic subsequences for testing from the genome
    predict_sequences = read_bed_file(args.predict, chromos)

    def prepare_features(sequences, n_cores):
        features = featurize_in_parallel(sequences, n_cores)
        # Concatenate features from parallel processing
        features = np.concatenate(features, axis=0)
        # Normalize and clean data types
        features = np.nan_to_num(features.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return features
 
    # Running the featurize_sequence in parallel on 64-core virtual machine for the training sequences
    train_features = prepare_features(train_sequences, 64)
    train_labels = np.array(train_labels, dtype=np.float64)
    dtrain = xgb.DMatrix(train_features, label=train_labels)
    # These 3 lines of code can be removed for other objective functions
    threshold = np.median(dtrain.get_label())  
    binary_labels = (dtrain.get_label() > threshold).astype(int)
    dtrain.set_label(binary_labels)
    
    # Running the featurize_sequence in parallel on 64-core virtual machine for the training sequences for the predicting sequences
    predict_features = prepare_features(predict_sequences, 64)
    dtest = xgb.DMatrix(predict_features)

    # Function to find the hyperparameters
    def grid_search_xgboost(dtrain, param_grid):
        best_score = float('-inf')
        best_params = {}
        for params in [dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())]:
            cv_results = xgb.cv(params, dtrain, num_boost_round=90, nfold=5, metrics='auc', early_stopping_rounds=10, as_pandas=True)
            mean_auc = cv_results['test-auc-mean'].max()
            if mean_auc > best_score:
                best_score = mean_auc
                best_params = params
        return best_params
        
    # Define parameter grid
    param_grid = {
        'objective': ['binary:logistic'],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'eval_metric': ['auc']
    }
    
    
    # Getting parameters
    best_parameters = grid_search_xgboost(dtrain, param_grid)
    
    # Use the best parameters found to train the final model
    final_model = xgb.train(best_parameters, dtrain, num_boost_round=90)

    # Predicting the ATAC-seq peaks using the trained model
    predictions = final_model.predict(dtest)

    # Saving one-dimensional output array of predicted labels as a numpy array
    np.save(args.output_file, predictions)

    
