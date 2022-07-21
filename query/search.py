import numpy as np
import pickle
import os.path

from torch import cosine_similarity
from sentence_transformers.util import semantic_search

DATA_PATH = '/Users/gauthierguinet/Github/RecoSystDPR/data/'
EMBEDDING_PATH = DATA_PATH+ 'embedding_data/emd.pkl'
PASSAGE_PATH =  DATA_PATH+ 'passage_data/passage.pkl'

def search(sentence_embedding,
           top_k =10):
    """
    Search for similar passages using cosine similarity
    Args:
        sentence_embedding: arr
        top_k: int
    Returns:
        output: Dict[str, List[float]]
    """
    # Check if we previously stored data
    try:
        with open(EMBEDDING_PATH, "rb") as f:
            emb_data = pickle.load(f)
        with open(PASSAGE_PATH, "rb") as f:
            passage_data = pickle.load(f)
    except:
        raise Exception('No embedding or passage data found')

    # Compute the cosine similarity between the sentence embedding and the embeddings
    
    candidates = semantic_search(query_embeddings=sentence_embedding,
                                 corpus_embeddings=emb_data,
                                 top_k=top_k)[0]
    
    # Access the passages corresponding to the candidates

    for cand in candidates:
        cand['passage'] = passage_data[cand['corpus_id']]

    output = {'corpus_id': [cand['corpus_id'] for cand in candidates],
              'passage': [cand['passage'] for cand in candidates],
              'scores': [cand['score'] for cand in candidates]}

    print(sentence_embedding)

    return output


