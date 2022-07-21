import re
import nltk
from nltk import sent_tokenize

nltk.download('punkt')

PASSAGE_LENGTH = 4

def parser(text): 
    """
    Args: 
        text: String 
    Returns:
        result: List[str]  
    """    
    # Regular expression matching to remove '[XYZ]' and '   '
    # remove_acc is applied twice for '[[XZY]]'
    text = re.sub(r'\[[A-Za-z0-9:\.x_-]*\]|[\n]','',text)
    text = re.sub(r'\[[A-Za-z0-9:\.x_-]*\]|[\n]','',text)
    text = re.sub(r"\s+",' ',text)

    # Split the text into sentences
    sentences = sent_tokenize(text)
    
    # Drop short sentences
    sentences = [s for s in sentences if len(s)>10]

    # Create passage using PASSAGE_LENGTH sentences
    # and a rolling window of PASSAGE_LENGTH//2
    n = len(sentences)
    res = []
    for i in range(0,max(1,n-PASSAGE_LENGTH),PASSAGE_LENGTH//2): 
        res.append(' '.join(sentences[i:i+PASSAGE_LENGTH+1]))

    return res
