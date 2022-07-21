import numpy as np
import pandas as pd
import pickle
import os.path
import os

from torch import cosine_similarity
from sentence_transformers.util import semantic_search

os.environ['KMP_DUPLICATE_LIB_OK']='True'

DATA_PATH = '/Users/gauthierguinet/Github/RecoSystDPR/data/'
EMBEDDING_PATH = DATA_PATH+ 'embedding_data/emd.pkl'
PASSAGE_PATH =  DATA_PATH+ 'passage_data/passage.pkl'
VIS_PATH = '/Users/gauthierguinet/Github/RecoSystDPR/data/data_visu'

def save_to_tsv():
    """
    Save embeddings and passages in a tsv file to use in tensorboard projector
    """
    # Check if we previously stored data
    try:
        with open(EMBEDDING_PATH, "rb") as f:
            emb_data = pickle.load(f)
        with open(PASSAGE_PATH, "rb") as f:
            passage_data = pickle.load(f)
    except:
        raise Exception('No embedding or passage data found')

    # Save the data on tsv format

    # Convert NumPy array of embedding into data frame
    embedding_df = pd.DataFrame(emb_data)

    # Save dataframe as as TSV file without any index and header
    embedding_df.to_csv(VIS_PATH+'/output.tsv', sep='\t', index=None, header=None)

    # Save MetaData without any index
    passage_df = pd.DataFrame(passage_data)

    passage_df.to_csv(VIS_PATH+'/metadata.tsv', sep='\t', index=None, header=None)



