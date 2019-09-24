from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Input, Dense, TimeDistributed
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam

import numpy as np
import pickle

from helper import bleu_n_gram

class NMTModel:
    # Due to limitations of our model, the the source and target sequences should
    # have the same length
    SOURCE_SEQ_LEN = TARGET_SEQ_LEN = 21

    def __init__(self, source_tokenizer, target_tokenizer):
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer

        self.source_vocab_size = len(source_tokenizer.word_index)
        self.target_vocab_size = len(target_tokenizer.word_index)

        self.model = self.model_description()

    @classmethod
    def create_from_corpora(cls, source_texts, target_texts):
        source_tokenizer = cls.tokenizer(source_texts)
        target_tokenizer = cls.tokenizer(target_texts)
        return cls(source_tokenizer, target_tokenizer)

    @classmethod
    def vectorize(cls, texts, tokenizer, seq_len):
        seqs = tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, seq_len)

    def vectorize_sources(self, texts):
        return self.vectorize(texts, self.source_tokenizer, self.SOURCE_SEQ_LEN)

    def vectorize_targets(self, texts):
        seqs = self.vectorize(texts, self.target_tokenizer, self.TARGET_SEQ_LEN)
        # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
        return seqs.reshape(*seqs.shape, 1)

    @classmethod
    def tokenizer(cls, texts):
        t = Tokenizer()
        t.fit_on_texts(texts)
        return t

    def model_description(self):
        embedding_dim = 64
        learning_rate = 1e-3

        layers = [
            Embedding(self.source_vocab_size + 2, embedding_dim,
                      input_length=self.TARGET_SEQ_LEN),
            GRU(64, return_sequences=True),
            TimeDistributed(
                Dense(self.target_vocab_size + 2, activation='softmax')
            )
        ]

        model = Sequential(layers)
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=Adam(learning_rate))
        return model

    def train(self, source_texts, target_texts, epochs):
        X = self.vectorize_sources(source_texts)
        Y = self.vectorize_targets(target_texts)

        self.model.fit(X, Y, batch_size=2048, epochs=epochs,
                             validation_split=0.1, verbose=2)

        train_source_txts = source_texts[:100]
        train_trgt_txts = target_texts[:100]

        metrics = {
        'train_bleu_1': bleu_n_gram(self, train_source_txts, train_trgt_txts, 1),
        'train_bleu_2': bleu_n_gram(self, train_source_txts, train_trgt_txts, 2),
        }
        return metrics


    def predict(self, input_text):
        X = self.vectorize_sources(input_text)
        logits = self.model.predict(X)
        pred_classes = np.argmax(logits, axis=2)
        return self.target_tokenizer.sequences_to_texts(pred_classes)

    @classmethod
    def load_model(cls, dir_path):
        src_tok = pickle.load(open(dir_path + '/source_tokenizer.pickle', 'rb'))
        trgt_tok = pickle.load(open(dir_path + '/target_tokenizer.pickle', 'rb'))
        model = load_model(dir_path + '/model.h5')
        rv = cls(src_tok, trgt_tok)
        rv.model = model
        return rv
