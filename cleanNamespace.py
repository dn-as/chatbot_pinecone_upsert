import pinecone
import os
from dotenv import load_dotenv

load_dotenv()

# find API key in console at app.pinecone.io
os.environ['PINECONE_API_KEY'] = os.getenv('pinecone_api_key')
# environment is found next to API key in the console
os.environ['PINECONE_ENVIRONMENT'] = os.getenv('pinecone_env')

pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment=os.environ['PINECONE_ENVIRONMENT']
)

# for indexName in pinecone.list_indexes():
#     pinecone.delete_index(indexName)
# pinecone.list_indexes()
# pinecone.index.delete(delete_all=True, namespace="notion-kb") 

index = pinecone.Index("dnas-wiki")

delete_response = index.delete(delete_all=True, namespace="notion-kb")

print(delete_response)