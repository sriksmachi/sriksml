from fastapi import FastAPI, File, Form, HTTPException, Query, Body, Request, UploadFile
from fastapi import Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import os
from dotenv import load_dotenv
from utils.llmManager import RAGAgent
from utils.rulesEngine import RulesEngine

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()


def Setup():
    azure_openai_key = os.environ["AZURE_OPENAI_API_KEY"]
    azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    azure_openai_version = os.environ["AZURE_OPENAI_API_VERSION"]
    azure_openai_chat_deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
    azure_openai_embeddings_deployment = os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"]
    rag_agent = RAGAgent(azure_openai_version, azure_openai_chat_deployment,
                         azure_openai_embeddings_deployment, azure_openai_endpoint, azure_openai_key)
    return rag_agent


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
rag_agent = Setup()
rulesEngine = RulesEngine()


@ app.get("/")
async def root(request: Request):
    '''
    Welcome message
    '''
    logger.info("Hello World")
    return {"message": "hello world"}


@ app.post("/upload")
async def upload_file(file: UploadFile = File(...), file2: UploadFile = File(...)):
    '''
    Upload file
    '''
    logger.info(f"File: {file.filename}")
    logger.info(f"File2: {file2.filename}")

    response = []

    # Read the file to azure blob container and invoke the chunker
    # chunker = Chunker()
    # chunker.process(containerName, file1.filename)

    # Get Rules from Rules Engine
    rules = rulesEngine.get_rules()

    # Run the RAG model for each rule and concatenate the results
    for rule in rules:
        result = rag_agent.executeRule(rule)
        response.append(result)

    return {"response": response}
