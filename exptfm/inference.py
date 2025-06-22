from exptfm.models import Model, StaticModel
from allspark.composition.composition_func import Comp_factory

class EmbeddingModel():

    def __init__(self, model_name):
        self.model = None
        self.model_name = model_name
        self.device = "cpu"
        self.model = Model(self.model_name, self.device)
        
    def set_device(self, device):
        self.device = device

    def select_layer(self, output, layer_number):
        return output[2][layer_number:layer_number+1]
    
    def get_embeddings(self, sentences):
        tokens = self.model.tokenize(sentences)
        output = self.model.inference(tokens)
        return output

    def get_embedding_from_layer_with_composition_function(self, embeddings, composition_function, layer_number):
        composition = Comp_factory.get_compfun(composition_function)
        output = embeddings
        output = self.select_layer( output, layer_number )
        # output = composition.clean_special(output, self.model.special_mask)
        output = composition.compose(output)
        return output

    def encode_with_composition_function(self, sentences, composition_function, layer_number):
        embeddings = self.get_embeddings( sentences )
        return self.get_embedding_from_layer_with_composition_function(embeddings, composition_function, layer_number)



class StaticEmbeddingModel():

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = StaticModel(self.model_name)
    
    def encode_with_composition_function(self, sentences, composition_function):
        composition = Comp_factory.get_compfun(composition_function)
        sentences_tokens = self.model.tokenize(sentences)
        output = self.model.inference(sentences_tokens)
        # .unsqueeze(0) is used here also because composition expects 2D. Compatibility with NonStaticModels.
        output = composition.compose([row.unsqueeze(0) for row in output])
        # The output is a list of sentences embeddings
        return output