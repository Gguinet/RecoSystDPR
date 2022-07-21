# RecoSystDPR
Semantic Search Engine leveraging Dense Passage Retrieval. 
Two main components are indexing and querying over documents.

### API Documentation:

Launch FastAPI locally using ```uvicorn main:app``` with ```--reload``` if in testing mode.

 ```/index``` ENDPOINT accepts ```str``` and ```.txt``` files. It can be accessed by using the following instructions:
```
curl -X 'POST' 'http://127.0.0.1:8000/query' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{"texts":["text1","text2"], \
         "files": ["file1","file2"]}'
```

```/query``` ENDPOINT accepts ```str``` and ouput json doc. It can be queried by using the following instructions:
```
curl -X 'POST' 'http://127.0.0.1:8000/query' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{"text":"How to choose your cofounder ?"}'
```
The ouput is a dict with the top 10 predicted candidates in currently stored data, with the confidence score and the corpus id.

### Code Structure:
* ```index/index```supports the indexing of documents and accepts as input files or texts.
    * Calls ```index/parser``` receive a doc and ouput a list of passages.
    * Calls ```embeddings/transformer``` to generate emdeddings_vec for each passage.
    * Calls ```index/store``` to save (id_doc,id_passage,passage_txt,emdeddings_vec) in a convenient datastructure for Search.
* ```query/query``` supports the semantic search over documents by 
    * Calls ```embeddings/transformer``` to generate embedding for this passage.
    * Calls ```query/search``` to compute cosine similarity score between this passage and the list of emdebbings stored in the database. 
    * Return top-k passages, with cosine score as a confidence.

### Frontend
* ```visualization/visu.py``` allows to store data in tsv file to leverage tensorboard embedding visualizer.
* ```visualization/fontend``` allows to generate a gradio chatbot to query the data and an online link to interact with it.

### Some ideas of improvement:
- [ ] (ML) Allows the chatbot to leverage in some way the previous queries and answers to generate output to current query.
- [ ] (ML) Focus on other metrics to generate optimal candidates to the query, with particular interest on diversity of outputs.
- [ ] (Soft. Eng. + ML) Use [hnswlib](https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic-search/semantic_search_quora_hnswlib.py) to improve the indexing and the search over the documents.
- [ ] (Soft. Eng.) Improve and Automate the visualization on [Tensorboard Embedding Projector](https://projector.tensorflow.org).

