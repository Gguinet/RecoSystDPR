from embeddings.transformer import embedding_model
from query.search import search

def query(sentence):
    """
    Args: 
        sentence: String 
    Returns:
        output:  
    Raises:
        ArgumentTypeError: If v is not a String nor a bool
    """ 

    # Find the embeddings of the sentence

    sentence_embedding = embedding_model([sentence])

    # Output candidates index and scores

    candidates = search(sentence_embedding)

    # Step 3: output = {cand = ["","",""],scores = [0.4,0.3,0.2]}

    return candidates

