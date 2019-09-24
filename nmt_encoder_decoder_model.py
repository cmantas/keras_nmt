from nmt_model import NMTModel

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.layers import Embedding
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

class NMTEncoderDecoderModel(NMTModel):
    def model_description(self):
        embedding_dim = 64
        learning_rate = 1e-3

        layers = [
            Embedding(self.source_vocab_size + 2, embedding_dim,
                      input_length=self.SOURCE_SEQ_LEN),
            LSTM(64),
            RepeatVector(self.TARGET_SEQ_LEN),
            LSTM(64, return_sequences=True),
            TimeDistributed(
                Dense(self.target_vocab_size + 2, activation='softmax')
            )
        ]

        model = Sequential(layers)
        model.compile(loss=sparse_categorical_crossentropy,
                      optimizer=Adam(learning_rate))
        return model