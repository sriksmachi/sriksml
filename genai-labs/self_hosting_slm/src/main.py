###########################
# Author : Srikanth Machiraju
# The code in this file is experimental and not production ready.
#############################


import os
import openai
import fastapi
import debugpy
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

ollama_url = "http://host.docker.internal:7869/"

# ollama_url = "http://localhost:11434/" # use this for testing locally, when ollaama is running on your local machine

class Input(BaseModel):
    query: str

app = fastapi.FastAPI()

# Allow all origins for CORS (you can customize this based on your requirements)
origins = ["*"]

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

debugpy.listen(("0.0.0.0", 5678))

llm = Ollama(
    base_url=ollama_url,
    model="phi3",
    verbose=True)

@app.get("/")
async def read_root():
    return {"Hello": "World"}

def setup():
    """
    Set up the retrieval QA chain.
    Returns:
        RetrievalQA: The retrieval QA chain object.
    """
    # Create embeddings
    print(f'Current working directory : {os.getcwd()}')
    # the data folder should be in the same directory as this file
    # the folder name here should match with the folder name used in the docker-compose file
    data_folder = "rag_data" 
    files = os.listdir(os.getcwd() + f"/{data_folder}")
    
    QA_CHAIN_PROMPT = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Context: {context}
    Question: {question}
    Answer:"""
    prompt = PromptTemplate(template=QA_CHAIN_PROMPT, input_variables=["context", "question"])
    
    all_pages = []
    for file in files:
        print(f"Processing file: {file}")
        # Load the document
        loader = PyPDFLoader(os.getcwd() + f'/{data_folder}/' + file)
        print("Loading pages...")
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(pages)
        print(f"Loaded {len(pages)} pages")
        all_pages.extend(pages)    
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings(gpt4all_kwargs={}))
    print("Vectorstore created")
    vectorstore = vectorstore.as_retriever()
    #Setup RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore,
        chain_type_kwargs={"prompt": prompt},
        verbose=True
    )
    return chain
    
agent = setup()

# Define a POST endpoint to generate a response to a question
@app.post("/generate")
async def chat(question: Input):
    print('question:', question.query)
    response = agent(question.query)
    return response