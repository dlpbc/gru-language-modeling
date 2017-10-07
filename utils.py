import itertools
import numpy as np
import nltk
import time
import sys
import operator
import io
import array
from datetime import datetime
from gru_rnn_theano import GRUTheano
import pprint

UNKNOWN_TOKEN = "UNKNOWN_TOKEN"
SENTENCE_START_TOKEN = "SENTENCE_START"
SENTENCE_END_TOKEN = "SENTENCE_END"

def load_data(filename='data/nl-feat-links-dataset.txt', vocabulary_size=2000, min_sent_characters=0):
    word_to_index = []
    index_to_word = []

    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print("\nLoading dataset...")
    with open(filename) as f:
        # Read sentences
        sentences = f.readlines()
        sentences = [sentence.lower() for sentence in sentences]
        # Filter sentences
        sentences = [s for s in sentences if len(s) >= min_sent_characters]
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (SENTENCE_START_TOKEN, x, SENTENCE_END_TOKEN) for x in sentences]
    print("Parsed %d featured links." % (len(sentences)))

    # Tokenise the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique words token." % len(word_freq.items()))
        
    # Get the most common words and build inex_to_word and word_to_index vectors
    #vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse=True)[:vocabulary_size-2]
    vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse=True)
    # removing parenthesis and some style of quotation marks from vocabulary
    # they occur too many times in this vocabulary set that the neural network learns to misuse them :-)
    # In essence, I suspect there are not enough words in the vocabulary to enable the network learn better
    # associations with the parentheses and quotation marks.
    vocab = [x for x in vocab if x[0] not in ['(', ')', '``', "''"]]
    vocab = vocab[:vocabulary_size - 2]
    print("Using vocabulary size %d." % vocabulary_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

    sorted_vocab = sorted(vocab, key=operator.itemgetter(1))
    index_to_word = ["<MASK/>", UNKNOWN_TOKEN] + [x[0] for x in sorted_vocab]
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]

    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
    return X_train, y_train, word_to_index, index_to_word

def save_model_parameters(model, outfile):
    # Get model parameters
    np.savez(outfile,
        E=model.E.get_value(),
        U=model.U.get_value(),
        U2=model.U2.get_value(),
        W=model.W.get_value(),
        V=model.V.get_value(),
        b=model.b.get_value(),
        c=model.c.get_value())
    print("\nSaved model parameters to %s." % outfile)

def load_model_parameters(path, modelClass=GRUTheano):
    npzfile = np.load(path)
    E, U, U2, W, V, b, c = npzfile["E"], npzfile["U"], npzfile["U2"], npzfile["W"], npzfile["V"], npzfile["b"], npzfile["c"]
    embedding_dim, word_dim = E.shape[0], E.shape[1]
    hidden_dim = U.shape[1]
    print("\nBuildng model from %s with word_dim=%d, embedding_dim=%d and hidden_dim=%d\n" % (path, word_dim, embedding_dim, hidden_dim))
    sys.stdout.flush()
    model = modelClass(word_dim, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    model.E.set_value(E)
    model.U.set_value(U)
    model.U2.set_value(U2)
    model.W.set_value(W)
    model.V.set_value(V)
    model.b.set_value(b)
    model.c.set_value(c)
    return model
    
# Outer SGD Loop
# - model: The RNN model instancde
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - decay
# - callback_every
# - callback
def train_with_sgd(model, X_train, y_train, learning_rate=0.001, nepoch=20, decay=0.9, callback_every=10000, callback=None):
    num_examples_seen = 0
    for epoch in np.arange(nepoch):
        # For each training example...
        for i in np.random.permutation(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate, decay)
            num_examples_seen += 1
            # Optionally do callback
            if (callback and callback_every and num_examples_seen % callback_every == 0):
                callback(model, num_examples_seen)
    return model

# Generate a sentence using the trained RNN model
# model - the trained RNN model used to generate sentences
def generate_sentence(model, index_to_word, word_to_index, min_length=5):
    # We start the sentence with the start token
    new_sentence = [word_to_index[SENTENCE_START_TOKEN]]
    # Repeat unitl we get an end token
    while not new_sentence[-1] == word_to_index[SENTENCE_END_TOKEN]:
        # Get the predicted next word in the sequence

        next_word_probs = model.predict(new_sentence)[-1]
        #print("\nsum = %f" % np.sum(next_word_probs[:-1]))
        #print("sumall = %f" % np.sum(next_word_probs))
        if sum(next_word_probs[:-1]) > 1.0:
            #print("Culprit found...")
            #print(next_word_probs)
            #print("stop generating. about to crash")
            #sys.exit(0)
            return None
        samples = np.random.multinomial(1, next_word_probs)
        sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)

        # Sometimes we get stuck if the sentence becomes too long
        # And we don't want sentences with UNKNOWN_TOKEN's
        if len(new_sentence) > 100 or sampled_word == word_to_index[UNKNOWN_TOKEN]:
            return None

    if len(new_sentence) < min_length:
        return None
    return new_sentence

def generate_sentences(model, n, index_to_word, word_to_index, sentence_min_length=5):
    for i in range(n):
        sent = None
        while not sent:
            sent = generate_sentence(model, index_to_word, word_to_index, sentence_min_length)
        print_sentence(sent, index_to_word)

def print_sentence(s, index_to_word):
    sentence_str = [index_to_word[x] for x in s[1:-1]]
    print(" ".join(sentence_str))
    sys.stdout.flush()

def gradient_check_theano(model, x, y, h=0.001, error_threshold=0.01):
    # Overwrite the bptt attribute. We need to backpropagate all the way to get the correct gradient
    model.bptt_truncate = 1000
    # Calculate the gradients using backprop
    bptt_gradients = model.bptt(x, y)
    # List of all parameters we want to check.
    model_parameters = ['E', 'U', 'W', 'b', 'V', 'c']
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # Get the actual parameter value from model e.g. model.U
        parameter_T = operator.attrgetter(pname)(model)
        parameter = parameter_T.get_value()
        print("Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape)))
        # Iterate over each element of the parameter matrix, e.g. (0, 0), (0, 1), ...
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # Save the original value so we can reset it later
            original_value = parameter[ix]
            # Estimate the gradient using (f(x+h) - f(x-h)) / (2*h)
            parameter[ix] = original_value + h
            parameter_T.set_value(parameter)
            gradplus = model.calculate_total_loss([x], [y])
            parameter[ix] = original_value - h
            parameter_T.set_value(parameter)
            gradminus = model.calculate_total_loss([x], [y])
            estimated_gradient = (gradplus - gradminus) / (2*h)
            parameter[ix] = original_value
            parameter_T.set_value(parameter)
            # The gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]
            # Calculate the relative error: (|x - y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient) / (np.abs(backprop_gradient) + np.abs(estimated_gradient))
            # If the error is too large fail the gradient check
            if relative_error > error_threshold:
                print("Gradient Check Error: parameter=%s ix=%s" % (pname, ix))
                print("+h Loss: %f" % gradplus)
                print("-h Loss: %f" % gradminus)
                print("Estimated_gradient: %f" % estimated_gradient)
                print("Backpropagation gradient: ", backprop_gradient)
                print("Relative Error: %f" % relative_error)
                return
            it.iternext()
        print("Gradient check for paramter %s passed." % (pname))

