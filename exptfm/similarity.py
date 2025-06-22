import torch
import torch.nn.functional as F
from itertools import repeat
import numpy as np

def compute_cosine_similarity(embedding1, embedding2):
    """
    Compute the cosine similarity between two sentence embeddings.
    
    Args:
        embedding1 (torch.Tensor): The first sentence embedding.
        embedding2 (torch.Tensor): The second sentence embedding.
    
    Returns:
        similarity (float): Cosine similarity score.
    """
    similarity = F.cosine_similarity(embedding1, embedding2)
    return similarity.item()

def compute_euclidean_distance(embedding1, embedding2):
    """
    Compute the Euclidean distance between two sentence embeddings.
    
    Args:
        embedding1 (torch.Tensor): The first sentence embedding.
        embedding2 (torch.Tensor): The second sentence embedding.
    
    Returns:
        distance (float): Euclidean distance between the embeddings.
    """
    distance = torch.norm(embedding1 - embedding2, p=2)  # L2 norm for Euclidean distance
    return distance.item()

def compute_icmb(embedding1, embedding2, b = 1):
        """
        Calculates the Vector-based Information Contrast Model with
        B = 1 by default, equivalent to inner product
        """
        def _icmb(v1, v2, b):
           
            v1_norm = np.linalg.norm(v1)                    # ||v1||
            v2_norm = np.linalg.norm(v2)                    # ||v2||
            sqr_v1norm = np.square(v1_norm)                 # ||v1||^2
            sqr_v2norm = np.square(v2_norm)                 # ||v2||^2

            rleft = sqr_v1norm + sqr_v2norm                 # (A+B)

            rright = b * (sqr_v1norm + sqr_v2norm - np.dot(v1,v2))# b(A+B-<v1,v2>)

     
            return rleft - rright # A-B
        
        return list(map(_icmb, embedding1.detach().numpy(), embedding2.detach().numpy(), repeat(b)))[0]