from autollm import AutoQueryEngine, read_files_as_documents
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

documents = read_files_as_documents(input_dir="./data")
query_engine = AutoQueryEngine.from_defaults(
     documents=documents,
     embed_model = "defaul",  # ["default", "local:intfloat/multilingual-e5-large"]
 )

response = query_engine.query ("Please give the summary of the document, \ 
     'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks'")

print(response)
