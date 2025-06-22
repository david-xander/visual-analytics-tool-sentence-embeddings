from exptfm.experiment import Experiment, ExperimentStaticModel

from exptfm.similarity import compute_cosine_similarity, compute_euclidean_distance, compute_icmb
from exptfm.datasets import load_dataset_as_pandas

import torch
import os

def main():

    # ======= TRANSFORMERS https://huggingface.co/transformers/v3.0.2/model_doc/auto.html
    # exp = Experiment()

    # exp.set_model(model_name="mspy/twitter-paraphrase-embeddings", number_of_layers=12)
    # exp.run()

    # exp.set_model(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", number_of_layers=12)
    # exp.run()

    # exp.set_model(model_name="w601sxs/b1ade-embed", number_of_layers=12)
    # exp.run()

    # exp.set_model(model_name="bert-base-uncased", number_of_layers=12)
    # exp.run()

    # exp.set_model(model_name="roberta-base", number_of_layers=12)
    # exp.run()

    # exp.set_model(model_name="sentence-transformers/all-MiniLM-L6-v2", number_of_layers=6)
    # exp.run()

    # exp.set_model(model_name="sentence-transformers/all-distilroberta-v1", number_of_layers=6)
    # exp.run()

    # exp.set_model(model_name="sentence-transformers/all-mpnet-base-v2", number_of_layers=12)
    # exp.run()

    # exp.set_model(model_name="sentence-transformers/paraphrase-MiniLM-L12-v2", number_of_layers=12)
    # exp.run()


    # ======= STATIC MODELS:  https://radimrehurek.com/gensim/models/word2vec.html
    # exp = ExperimentStaticModel(model_name="glove-twitter-25")
    # exp.run()
    # exp = ExperimentStaticModel(model_name="glove-twitter-200")
    # exp.run()
    # exp = ExperimentStaticModel(model_name="glove-wiki-gigaword-300")
    # exp.run()
    # exp = ExperimentStaticModel(model_name="fasttext-wiki-news-subwords-300")
    # exp.run()    
    exp = ExperimentStaticModel(model_name="word2vec-google-news-300")
    exp.run()    


    pass


if __name__ == "__main__":
    main()