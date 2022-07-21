from sentence_transformers import SentenceTransformer 
import numpy as np
import torch

MODEL_TYPE = 'all-MiniLM-L6-v2'

def embedding_model(passages,
                    normalize = True):
    """
    Compute dense embeddings of a sentence 
    Args: 
        passage: List[str]
        normalize: Bool
    Returns:
        embeddings: List[arr] 
    """  
    # Import Model
    model = SentenceTransformer(MODEL_TYPE)

    # Check if CUDA is available ans switch to GPU
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
        print(f'Model runing on {model.device}')

    # Convert passages to vectors
    embeddings = model.encode(passages,
                              show_progress_bar=True)

    # Normalize the embeddings to facilitate Cosine Similarity Search
    if normalize:
        embeddings /= np.linalg.norm(embeddings, axis=1)[:, np.newaxis] + 1e-8

    return embeddings
