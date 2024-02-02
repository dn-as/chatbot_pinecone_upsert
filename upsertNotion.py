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
from notion_client import Client

#####################################################
#                                                   #
# This file upserts noition KB in data to pinecone. #
#                                                   #
#####################################################

load_dotenv()
openai.api_key = os.getenv('api_key')
notion_key = os.getenv("notion_key")

# find API key in console at app.pinecone.io
os.environ['PINECONE_API_KEY'] = os.getenv('pinecone_api_key')
# environment is found next to API key in the console
os.environ['PINECONE_ENVIRONMENT'] = os.getenv('pinecone_env')

# initialize connection to pinecone
pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment=os.environ['PINECONE_ENVIRONMENT']
)

# setup the index/query process, i.e. the embedding model (and completion if used)
embed_model = OpenAIEmbedding(model='text-embedding-ada-002', embed_batch_size=100)
service_context = ServiceContext.from_defaults(embed_model=embed_model)

# Building namespace
index_name = "dnas-wiki"
namespace = "notion-kb"
print(f"Building namespace: {namespace} under index {index_name} ...\n")
# Get documents from notion
client = Client(auth=notion_key)
# TODO: recursively get all pages
response = client.search(
    query="",
    filter={"value": "page", "property": "object"}
)
notion_pages = response['results']

NotionPageReader = download_loader('NotionPageReader')
notion_pages_ids = [str(page['id']) for page in notion_pages]
notion_pages_urls = [str(page['url']) for page in notion_pages]
# if len(notion_pages_ids) == 0:
#     return "Sorry, there is no relevant information in Notion."
# else:
reader = NotionPageReader(integration_token=notion_key)
documents = reader.load_data(notion_pages_ids)
# index = VectorStoreIndex.from_documents(docments)
# query_engine = index.as_query_engine()
# return query_engine.query(question).response + "\nReferences:\n" + "\n".join(notion_pages_urls)

# documents = SimpleDirectoryReader(input_dir=input_dir).load_data()

# create the index if it does not exist already
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        pod_type="s1"
    )

# connect to the index
pineconeIndex = pinecone.Index(index_name)

vectorStore = PineconeVectorStore(
    pinecone_index=pineconeIndex,
    namespace=namespace
)

# setup our storage (vector db)
storageContext = StorageContext.from_defaults(
    vector_store=vectorStore
)

index = GPTVectorStoreIndex.from_documents(
    documents=documents, 
    storage_context=storageContext,
    service_context=service_context
)
print(f"Done building {namespace} !\n{len(documents)} notion pages upserted.")
