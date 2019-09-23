import os
from nltk.translate.bleu_score import corpus_bleu

def load_data(path):
    """
    Load dataset
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data.split('\n')

def bleu_n_gram(model, source_texts, target_texts, n):
    def splitter(corpus):
        return list(map(lambda t: t.split(), corpus))

    source_text_words = splitter(source_texts)
    target_text_words = splitter(target_texts)
    predicted_texts = model.predict(source_texts)
    predicted_texts_words = splitter(predicted_texts)

    bleu_weights = [0] * 3
    bleu_weights[n-1] = 1.
    return corpus_bleu(list(zip(source_text_words, target_text_words)),
                       predicted_texts_words, bleu_weights)