#!/usr/bin/env python
# coding: utf-8

import collections

import helper
import numpy as np

# As a toy dataset we will use french-english parallel corpus, created by Udacity.
# see here: https://www.floydhub.com/udacity/datasets/language-translation-en-fr.
# This is a small but high quality dataset similar to those one can find here:
# http://www.manythings.org/anki/ (based on the Tatoeba project)

# Load English data
source_texts = helper.load_data('data/small_vocab_en')
# Load French data
target_texts = helper.load_data('data/small_vocab_fr')

print('Dataset Loaded')


for sample_i in range(2):
    print('small_vocab_en Line {}:  {}'.format(sample_i + 1, source_texts[sample_i]))
    print('small_vocab_fr Line {}:  {}'.format(sample_i + 1, target_texts[sample_i]))

# Vocabulary

source_word_counter = collections.Counter([word for sentence in source_texts for word in sentence.split()])
target_word_counter = collections.Counter([word for sentence in target_texts for word in sentence.split()])

print('{} English words.'.format(len([word for sentence in source_texts for word in sentence.split()])))
print('{} unique English words.'.format(len(source_word_counter)))
print('10 Most common words in the English dataset:')
print('"' + '" "'.join(list(zip(*source_word_counter.most_common(10)))[0]) + '"')
print()
print('{} French words.'.format(len([word for sentence in target_texts for word in sentence.split()])))
print('{} unique French words.'.format(len(target_word_counter)))
print('10 Most common words in the French dataset:')
print('"' + '" "'.join(list(zip(*target_word_counter.most_common(10)))[0]) + '"')


#model = NMTModel.create_from_corpora(source_texts, target_texts)
model = NMTEncoderDecoderModel.create_from_corpora(source_texts, target_texts)

model.train(source_texts, target_texts, 4)

pred_texts = model.predict(source_texts[:10])

for i in range(10):
    print("   src: ", source_texts[i])
    print("target: ", target_texts[i])
    print("  pred: ", pred_texts[i], "\n")
