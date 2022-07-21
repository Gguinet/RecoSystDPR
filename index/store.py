import numpy as np
import pickle
import os.path

DATA_PATH = '/Users/gauthierguinet/Github/RecoSystDPR/data/'
EMBEDDING_PATH = DATA_PATH+ 'embedding_data/emd.pkl'
PASSAGE_PATH =  DATA_PATH+ 'passage_data/passage.pkl'

def store(embeddings,
          passages):
    """
    Store embeddings and passages in a pickle file
    Args:
        embeddings: List[arr]
        passages: List[str]
    """
    #### Save embeddings

    # Check if we previously stored data
    try:
        with open(EMBEDDING_PATH, "rb") as f:
            emb_data = pickle.load(f)
            # Update data
            emb_data = np.concatenate([emb_data,embeddings])
    except:
        emb_data = embeddings

    # Store the new data
    with open(EMBEDDING_PATH, 'wb') as f:
        pickle.dump(emb_data, f)    

    #### Save passages

    try: 
        with open(PASSAGE_PATH, "rb") as f:
            passage_data = pickle.load(f)

            # Update data
            passage_data = np.concatenate([passage_data,
                                           np.array(passages)])
    except: 
        passage_data = np.array(passages)
        
    # Store the new data
    with open(PASSAGE_PATH, 'wb') as f:
        pickle.dump(passage_data, f) 
