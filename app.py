from autollm import AutoQueryEngine, read_files_as_documents
import os
from dotenv import load_dotenv

# Loading the .env file
# Please put your API like following 
# OPENAI_API_KEY = "your API key"
load_dotenv()

# For local model
#  llm_model = "ollama/llama2"
#  llm_api_base = "http://localhost:1234/v1"

#For ChatGPT API, please put .env file in the same directory as app.py
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

#Read all documents in the ./data folder
documents = read_files_as_documents(input_dir="./data")
query_engine = AutoQueryEngine.from_defaults(
     documents=documents,
     # default means using GPT, local means the local LLM 
     embed_model = "default",  
     #for local model
     #llm_model=llm_model,
     #llm_api_base=llm_api_base,
 )
# The sample query 
response = query_engine.query ("Please give the summary of the document, 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks'")

print(response)

