from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from retrieval_04 import router as retrieval_router
from indexing_03 import router as indexing_router 

app = FastAPI(title="text to sql converter and executer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Or specify ["GET", "POST", "PUT", "DELETE"]
    allow_headers=["*"],
)

app.include_router(retrieval_router, tags=["Retrieval"])
app.include_router(indexing_router, tags=["Indexing"])
