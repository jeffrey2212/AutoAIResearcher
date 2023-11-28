from autollm import AutoQueryEngine, read_files_as_documents
import os
from dotenv import load_dotenv

load_dotenv()

#For local model
llm_model = "ollama/llama2"
llm_api_base = "http://localhost:1234/v1"

#For ChatGPT API, please put .env file in the same directory as app.py
# OPENAI_API_KEY = "your API key"
#os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

documents = read_files_as_documents(input_dir="./data")
query_engine = AutoQueryEngine.from_defaults(
     documents=documents,
     embed_model = "local:perplexity/mistral-7b-instruct",  # ["default", "local:intfloat/multilingual-e5-large"]
     llm_model=llm_model,
     llm_api_base=llm_api_base,
 )

response = query_engine.query ("Please give the summary of the document, 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks'")

print(response)