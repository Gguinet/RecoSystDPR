from index.parser import parser
from embeddings.transformer import embedding_model
from index.store import store

def index(document):
    """
    Args: 
        document: String 
    Returns:
        None  
    Raises:
        ArgumentTypeError: If v is not a String nor a bool
    """ 

    # Split the document in overlapping passages
    passages = parser(document)

    # Compute normalized embeddings of the passages
    embeddings = embedding_model(passages,
                                 normalize=True)

    # Store the embeddings in an index suited for fast search
    store(embeddings,
          passages)

    return

