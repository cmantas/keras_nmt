#!/usr/bin/env python
# coding: utf-8

import collections

import helper
import numpy as np
#import project_tests as tests

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import LSTM, GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy

# As a toy dataset we will use french-english parallel corpus, created by Udacity.
# see here: https://www.floydhub.com/udacity/datasets/language-translation-en-fr.
# This is a small but high quality dataset similar to those one can find here:
# http://www.manythings.org/anki/ (based on the Tatoeba project)

# Load English data
english_sentences = helper.load_data('data/small_vocab_en')
# Load French data
french_sentences = helper.load_data('data/small_vocab_fr')

print('Dataset Loaded')


# In[4]:


for sample_i in range(2):
    print('small_vocab_en Line {}:  {}'.format(sample_i + 1, english_sentences[sample_i]))
    print('small_vocab_fr Line {}:  {}'.format(sample_i + 1, french_sentences[sample_i]))


# ## Pre-processing
# 
# ### Vocabulary
# The complexity of the problem is determined by the complexity of the vocabulary.  A more complex vocabulary is a more complex problem.  Let's look at the complexity of the dataset we'll be working with.

# In[5]:


english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split()])
french_words_counter = collections.Counter([word for sentence in french_sentences for word in sentence.split()])

print('{} English words.'.format(len([word for sentence in english_sentences for word in sentence.split()])))
print('{} unique English words.'.format(len(english_words_counter)))
print('10 Most common words in the English dataset:')
print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')
print()
print('{} French words.'.format(len([word for sentence in french_sentences for word in sentence.split()])))
print('{} unique French words.'.format(len(french_words_counter)))
print('10 Most common words in the French dataset:')
print('"' + '" "'.join(list(zip(*french_words_counter.most_common(10)))[0]) + '"')


# ## Preprocess
# We need to convert our textual data into numbers, so that we can feed it into our neural network model. We will be using the following preprocess methods to do that;
# 1. Tokenize the words into ids
# 2. Add padding to make all the sequences the same length.
# 
# ### Tokenization
# We can turn each character into a number or each word into a number. These
# are called character and word ids, respectively. Character ids are used for
# character level models that generate text predictions for each character.
# A word level model uses word ids that generate text predictions for each word.
# Word level models tend to learn better, since they are lower in complexity, so we'll use those.


def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    x_tk = Tokenizer()
    x_tk.fit_on_texts(x)
    return x_tk.texts_to_sequences(x), x_tk

# Tokenize Example output
text_sentences = [
    'The quick brown fox jumps over the lazy dog .',
    'By Jove , my quick study of lexicography won a prize .',
    'This is a short sentence .']
text_tokenized, text_tokenizer = tokenize(text_sentences)
print(text_tokenizer.word_index)
print()
for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(sent))
    print('  Output: {}'.format(token_sent))


# ### Padding
# When batching the sequence of word ids together, each sequence needs to be the same length. Since sentences are dynamic in length, we can add padding to the end of the sequences to make them the same length.
# 
# All the English sequences and the French sequences should have the same length; in order to do that we will be padding the **end** of each sequence using Keras's [`pad_sequences`](https://keras.io/preprocessing/sequence/#pad_sequences) function.

# In[7]:

# Pad Tokenized output
test_pad = pad_sequences(text_tokenized)

for sample_i, (token_sent, pad_sent) in enumerate(zip(text_tokenized, test_pad)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(np.array(token_sent)))
    print('  Output: {}'.format(pad_sent))


def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad_sequences(preprocess_x)
    preprocess_y = pad_sequences(preprocess_y)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk

preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer =    preprocess(english_sentences, french_sentences)
    
max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)

print('Data Preprocessed')
print("Max English sentence length:", max_english_sequence_length)
print("Max French sentence length:", max_french_sequence_length)
print("English vocabulary size:", english_vocab_size)
print("French vocabulary size:", french_vocab_size)


# ## Models
# 
# Experimenting with various neural network architectures. I'll be training these four simple architectures:
# - Model 1 is a simple RNN
# - Model 2 is a RNN with Embedding
# - Model 3 is a Bidirectional RNN
# - Model 4 is a Encoder-Decoder RNN
# - Model 5 is a Bidirectional RNN with Embedding
# 

print('`logits_to_text` function loaded.')


# ### Model 1: simple RNN 
# A basic RNN model is a good baseline for sequence data.


# ### Model 2: RNN with Embedding
# Converting the words into ids is a good step to preprocess your input inorder to feed them to neural network model, but there's a better representation of a word.  This is called word embeddings.  An embedding is a vector representation of the word that is close to similar words in n-dimensional space, where the n represents the size of the embedding vectors.
# 
# In this model, I have built a simple RNN with Embeddings.

# In[19]:


def embed_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a RNN model using word embedding on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    learning_rate = 1e-3
    
    input_seq = Input(input_shape[1:])
    embed_layer = Embedding(english_vocab_size, 64, input_length=output_sequence_length)(input_seq)
    rnn = GRU(64, return_sequences=True)(embed_layer)

    logits = TimeDistributed(Dense(french_vocab_size, activation='softmax'))(rnn)
    
    model = Model(inputs=input_seq, outputs=logits)
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate))
    return model

#tests.test_embed_model(embed_model)

# Reshape the input
tmp_x = pad_sequences(preproc_english_sentences, max_french_sequence_length)
# tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

# Train the neural network
embed_rnn_model = embed_model(
    tmp_x.shape,
    max_french_sequence_length,
    english_vocab_size +2,
    french_vocab_size + 2)

embed_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=2048, epochs=5,
                    validation_split=0.1, verbose=1)


def decode_logits(logits, target_tokenizer):
    pred_classes = np.argmax(logits, axis=2)
    return target_tokenizer.sequences_to_texts(pred_classes)


def predict_translation(model, enc_texts, target_tokenizer):
    predicted_logits = model.predict(enc_texts)
    return decode_logits(predicted_logits, target_tokenizer)

sample_txts_enc = embed_rnn_model.predict(tmp_x[:10])
predicted_texts = predict_translation(embed_rnn_model, tmp_x[:10], french_tokenizer)

for i in range(10):
    print("   src: ", english_sentences[i])
    print("target: ", french_sentences[i])
    print("  pred: ", predicted_texts[i], "\n")
