from pathlib import Path
from llama_hub.file.unstructured import UnstructuredReader
from pathlib import Path
from llama_index import download_loader
from llama_index import SimpleDirectoryReader, VectorStoreIndex
from dotenv import load_dotenv
import os
from llama_index.node_parser import SimpleNodeParser
import pinecone
from llama_index.vector_stores import PineconeVectorStore
from llama_index import GPTVectorStoreIndex, StorageContext, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
import openai

###################################################
# 
# This file upserts documents in data to pinecone.
# 
###################################################

load_dotenv()
# os.environ['OPENAI_API_KEY'] = os.getenv('api_key')  # platform.openai.com
openai.api_key = os.getenv('api_key')

loader = UnstructuredReader()

documents = SimpleDirectoryReader(input_dir="./data/", file_extractor={".htm": UnstructuredReader()}).load_data()
# documents = loader.load_data(Path('./data/*'))
# parser = SimpleNodeParser.from_defaults()

# nodes = parser.get_nodes_from_documents(documents)

# find API key in console at app.pinecone.io
os.environ['PINECONE_API_KEY'] = os.getenv('pinecone_api_key')
# environment is found next to API key in the console
os.environ['PINECONE_ENVIRONMENT'] = os.getenv('pinecone_env')

# initialize connection to pinecone
pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment=os.environ['PINECONE_ENVIRONMENT']
)

# create the index if it does not exist already
index_name = 'dnas-sops'
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        dimension=1536,
        metric='cosine'
    )

# connect to the index
pinecone_index = pinecone.Index(index_name)

# we can select a namespace (acts as a partition in an index)
namespace = '' # default namespace

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# setup our storage (vector db)
storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

# setup the index/query process, ie the embedding model (and completion if used)
embed_model = OpenAIEmbedding(model='text-embedding-ada-002', embed_batch_size=100)
service_context = ServiceContext.from_defaults(embed_model=embed_model)

index = GPTVectorStoreIndex.from_documents(
    documents=documents, storage_context=storage_context,
    service_context=service_context
)

# query_engine = index.as_query_engine()
# res = query_engine.query("What can you tell me about technicians on boarding?")
# print(res)

# query_engine.query("What does a new technician need to do?")
# pinecone.delete_index(index_name)