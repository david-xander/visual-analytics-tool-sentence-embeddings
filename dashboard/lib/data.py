import sys
sys.path.append('../../')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Dataset():
    def __init__(self):
        file_path1 = os.path.join("exptfm", "results", "bert-base-uncased_correlation.csv")
        file_path2 = os.path.join("exptfm", "results", "roberta-base_correlation.csv")
        file_path3 = os.path.join("exptfm", "results", "sentence-transformers-all-distilroberta-v1_correlation.csv")
        file_path4 = os.path.join("exptfm", "results", "sentence-transformers-all-MiniLM-L6-v2_correlation.csv")
        file_path5 = os.path.join("exptfm", "results", "sentence-transformers-all-mpnet-base-v2_correlation.csv")
        file_path6 = os.path.join("exptfm", "results", "sentence-transformers-paraphrase-MiniLM-L12-v2_correlation.csv")

        file_path7 = os.path.join("exptfm", "results", "glove-twitter-25_correlation.csv")
        file_path8 = os.path.join("exptfm", "results", "glove-twitter-200_correlation.csv")
        file_path9 = os.path.join("exptfm", "results", "glove-wiki-gigaword-300_correlation.csv")
        file_path10 = os.path.join("exptfm", "results", "fasttext-wiki-news-subwords-300_correlation.csv")
        # file_path14 = os.path.join("exptfm", "results", "word2vec-google-news-300_correlation.csv")

        file_path11 = os.path.join("exptfm", "results", "mspy-twitter-paraphrase-embeddings_correlation.csv")
        file_path12 = os.path.join("exptfm", "results", "w601sxs-b1ade-embed_correlation.csv")
        file_path13 = os.path.join("exptfm", "results", "sentence-transformers-paraphrase-multilingual-mpnet-base-v2_correlation.csv")

        f1 = pd.read_csv(file_path1)
        f2 = pd.read_csv(file_path2)
        f3 = pd.read_csv(file_path3)
        f4 = pd.read_csv(file_path4)
        f5 = pd.read_csv(file_path5)
        f6 = pd.read_csv(file_path6)
        f7 = pd.read_csv(file_path7)
        f8 = pd.read_csv(file_path8)
        f9 = pd.read_csv(file_path9)
        f10 = pd.read_csv(file_path10)
        f11 = pd.read_csv(file_path11)
        f12 = pd.read_csv(file_path12)
        f13 = pd.read_csv(file_path13)
        # f14 = pd.read_csv(file_path14)


        # ds = pd.concat([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13,f14])
        ds = pd.concat([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13])

        ds.rename(columns={"composition": "sim", "similarity": "comp"}, inplace=True)
        ds.rename(columns={"sim": "similarity", "comp": "composition"}, inplace=True)

        ds.loc[(ds["type"]=="Static") & (ds["composition"]=="Cls"), "pearsoncor"] = 0
        ds.loc[(ds["type"]=="Static") & (ds["composition"]=="Cls"), "spearman"] = 0
 
        self.correlation = ds
        self.model = None
        self.metrics = None
        self.embeddings = None
    
    def get_correlation_dataframe(self):
        return self.correlation

    def set_composition(self, value):
        self.composition = value

    def set_similarity(self, value):
        self.similarity = value

    def set_layer(self, value):
        self.layer = value

    def get_metrics_dataframe(self, model):
        if model != self.model:
            p_aux1 = os.path.join("exptfm", "results", model+"_metrics.csv")
            df = pd.read_csv(p_aux1)
            self.metrics = df
            
        return self.metrics
    
    def get_embeddings_dataframe(self, model):
        if model != self.model:
            p_aux1 = os.path.join("exptfm", "results", model+"_embeddings.pt")

            df_t = torch.load(p_aux1)
            df = pd.DataFrame(df_t)

            self.embeddings = df
        
        return self.embeddings