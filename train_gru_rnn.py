import sys
import os
import time
import numpy as np
from utils import *
from datetime import datetime
from gru_rnn_theano import GRUTheano

# Dataset related configuration
vocabulary_size = 500
dataset = 'data/nl-feat-links-5k.txt'

# Model related configuration
train_model_from_scratch = False
model_param_input_file = './trained_model_param/model_params.npz' #set this variable if not training from scratch
model_param_output_file = './trained_model_param/model_params_%s.npz' % datetime.now().strftime('%Y-%m-%d %H:%M:%S')
word_embedding_dim = 500 
hidden_dim = 400

# sgd training related configuration
#lr = 0.001
#lr = 0.0008
#lr = 0.0004
#lr = 0.0002
#lr = 0.0001
#lr = 0.00008
#lr = 0.00005
#lr = 0.00003
#lr = 0.00001
lr = 0.000001
epochs = 10

# Load data
X_train, y_train, word_to_index, index_to_word = load_data(dataset, vocabulary_size)

# Build GRU model
if not train_model_from_scratch:
    print("\nTraining model from a pre-trained model")
    print("Loading pre-trained model parameter...\n")

    model = load_model_parameters(model_param_input_file)
    print("Model parameter loading completed...")
else:
    model = GRUTheano(vocabulary_size, embedding_dim=word_embedding_dim, hidden_dim=hidden_dim, bptt_truncate=-1)
    print("\nTraining model from scratch...")

# Print SGD step time
t1 = time.time()
model.sgd_step(X_train[20], y_train[20], lr)
t2 = time.time()
print("\nSGD Step time: %f milliseconds" % ((t2 - t1) * 1000.))
sys.stdout.flush()

# We do this every few examples to understand what's going on
def sgd_callback(model, num_examples_seen):
    dt = datetime.now().isoformat()
    #loss = model.calculate_loss(X_train[:9000], y_train[:9000])
    loss = model.calculate_loss(X_train, y_train)
    print("\n%s (%d)" % (dt, num_examples_seen))
    print("-------------------------------------------------")
    print("Loss: %f" % loss)
    print("-------------------------------------------------")
    print("Generating sample sentences")
    generate_sentences(model, 10, index_to_word, word_to_index)
    #save_model_parameters(model, model_param_output_file)
    print("\n")
    sys.stdout.flush()

print("Training model with SGD...\n")
train_with_sgd(model, X_train, y_train, learning_rate=lr, nepoch=epochs, decay=0.9, callback_every=5000, callback=sgd_callback)

# Save model
print("\nTraining completed...")
print("Saving model...")
save_model_parameters(model, model_param_output_file)

# Generate some sample sentences
print("\n\nGenerating sample sentences from trained model\n")
num_sentences = 20
sentence_min_length = 5

generate_sentences(model, num_sentences, index_to_word, word_to_index, sentence_min_length)
