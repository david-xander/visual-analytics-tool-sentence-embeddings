# Inspired in https://github.com/adriangh-ai/AllSpark/blob/main/src/aspark_server/server_main.py

import torch
import transformers as ts
import numpy as np

import gensim.downloader as gensimapi
import nltk
from nltk.tokenize import word_tokenize

class Model():
    def __init__(self, model_name, device):
        self.model_name=model_name
        self.device = device
        self.special_mask = None
        self.parameters = {}
        self.instantiate()

    def instantiate(self):
        self.model = ts.AutoModel.from_pretrained(self.model_name)
        
        if self.device != "cpu":
            model = model.to(self.device)
                
        self.tokenizer=ts.AutoTokenizer.from_pretrained(self.model_name)     

        tmodel_conf = self.model.config.to_diff_dict()
        tmodel_conf["num_hidden_layers"] = self.model.config.num_hidden_layers
        tmodel_conf["num_param"] = self.model.num_parameters()
        self.parameters = tmodel_conf


    def tokenize(self, sentences):
        """
        Tokenize a sentence of sentences. Adds the mask of the special_tokens in the tokenized sentence.
        This mask is popped, so we can save the value but still gets removed from the token dictionary,
        as huggingface models don't support the passing of this parameter to the model
        """
        _tokens = self.tokenizer(sentences, padding=True
                                    ,truncation=True
                                    ,return_tensors="pt"
                                    ,return_special_tokens_mask=True).to(self.device)
        self.special_mask = _tokens.pop("special_tokens_mask")
        return _tokens
    
    def inference(self, tokens):
        """
        Sends the batch of tokens to the model for the feed forward pass.
        """
        # return self.model(input_ids=tokens['input_ids']
        #                     ,decoder_input_ids=tokens['input_ids']
        #                     ,output_hidden_states=True)
        return self.model(input_ids=tokens['input_ids'] ,output_hidden_states=True)

    def to(self, device):
        """
        Sends model to device's memory.
        """
        self.model = self.model.to(device)
        return self
    
    def paddding(self):
        """
        Getter for pad
        """
        return self.pad
    

class StaticModel():
    def __init__(self, model_name):
        nltk.download('punkt_tab')

        self.model_name=model_name
        self.model = self.instantiate()

    def instantiate(self):
        return gensimapi.load(self.model_name)

    def tokenize(self, sentences):
        res = []
        for sentence in sentences:
            sentence = sentence.lower()
            res.append(word_tokenize(sentence))
        return res
    
    def inference(self, sentences):
        embedding = []
        for tokens in sentences:
            embedding.append(
                torch.from_numpy(np.array([
                    self.model[token]
                    for token in tokens
                    if token in self.model
                ]))
            )
        return embedding
    