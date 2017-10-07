#! /usr/bin/python
'''
This script loads a pre-trained GRU RNN model and then use it to generate (sample) sentence(s).
'''

import sys
from gru_rnn_theano import GRUTheano
from utils import *

# Dataset related configuration
vocabulary_size = 500
dataset = 'data/nl-feat-links-5k.txt'

# Model related configuration
pre_trained_model_path = 'trained_model_param/model_params.npz'

# Generator related configuration
num_sentences = 30
#sentence_min_length = 5
sentence_min_length = 7

# Load data
X_train, y_train, word_to_index, index_to_word = load_data(dataset, vocabulary_size)

# Load model
model = load_model_parameters(pre_trained_model_path, modelClass=GRUTheano)

# Generate some sample sentences
print("\n\nGenerating sample sentences from pre-trained model\n")
generate_sentences(model, num_sentences, index_to_word, word_to_index, sentence_min_length)
