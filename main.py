from fastapi import FastAPI,UploadFile,File
from pydantic import BaseModel, constr, validator
from typing import List
from index.index import index
from query.query import query


app = FastAPI(
    title="RecoSystDPR",
    description="Search Engine with Dense Passage Retrieval.",
    version="1.0",
)

# TODO look at async

### /Index API ENDPOINT

# class UserRequestIndex(BaseModel):
#      texts: List[str] = None
#      files: UploadFile = File(description="Multiple files as UploadFile")

class OutputIndex(BaseModel):
     message: str

@app.post("/index", response_model=OutputIndex)
async def upload(texts:List[str],
                 files: List[UploadFile] = File(description="Multiple files as UploadFile")):

    if not texts and not files:
        raise Exception('No text or file provided')
    
    if files is not None:
        for file in files:
            try:
                content = await file.read()
                content = content.decode()
                index(content)
                
            except Exception:
                return {"message": "There was an error uploading the file"}
            finally:
                await file.close()
    
    if texts is not None:
        for text in texts:
            index(text)
    
    return {"message": "Successfully indexing."}

### /Query API ENDPOINT

class UserRequestInQuery(BaseModel):
    text: constr(min_length=1)

class CandidatesOut(BaseModel):
    passage: List[str]
    corpus_id: List[int]
    scores: List[float] 

@app.post("/query", response_model=CandidatesOut)
async def query_endpoint(user_request_in: UserRequestInQuery):
    return query(user_request_in.text)