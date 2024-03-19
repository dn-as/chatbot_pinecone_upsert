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

# Recursively gets all notion pages
start_cursor = None
results = []
has_more = True
while has_more:
    if start_cursor is None:
        response = client.search(
            query="",
            filter={"value": "page", "property": "object"}
        )
    else:
        response = client.search(
            query="",
            start_cursor=start_cursor,
            filter={"value": "page", "property": "object"}
        )
    results.extend(response.get('results', []))
    has_more = response.get('has_more', False)
    
    if has_more:
        start_cursor = response.get('next_cursor')


# response = client.search(
#     query="",
#     filter={"value": "page", "property": "object"}
# )
# notion_pages = response['results']
notion_pages = results

# Remove Notion pages that do not have a title
notion_valid_pages = []
for i, page in enumerate(notion_pages):
    if 'title' in page['properties'].keys():
        notion_valid_pages.extend([page])
        # if ('SOPs' not in page['properties']['title']) and ('How To\'s' not in page['properties']['title']) and ('Others' not in page['properties']['title']):
        #     notion_valid_pages.extend([page])
print("Total count of valid Notion pages:", len(notion_valid_pages))

NotionPageReader = download_loader('NotionPageReader')
notion_pages_ids = [str(page['id']) for page in notion_valid_pages]
notion_pages_urls = [str(page['url']) for page in notion_valid_pages]
# notion_pages_titles = [page['properties']['title']['title'][0]['plain_text'] for page in notion_pages]

reader = NotionPageReader(integration_token=notion_key)
documents = reader.load_data(notion_pages_ids)

# # Add metadata
# for i, doc in enumerate(documents):
#     doc.metadata['url'] = notion_pages_urls[i]
#     # doc.metadata['title'] = notion_pages_titles[i]
#     if 'title' in notion_pages[i]['properties'].keys():
#         # notion_pages[10]['properties']['title']['title'][0]['plain_text']
#         doc.metadata['title'] = notion_pages[i]['properties']['title']['title'][0]['plain_text']
#         print("Indexing", doc.metadata['title'])
#     else:
#         doc.metadata['title'] = notion_pages_urls[i]

# Add metadata
for i, doc in enumerate(documents):
    doc.metadata['url'] = notion_pages_urls[i]
    # doc.metadata['title'] = notion_pages_titles[i]
    # notion_pages[10]['properties']['title']['title'][0]['plain_text']
    doc.metadata['title'] = notion_valid_pages[i]['properties']['title']['title'][0]['plain_text']
    doc.metadata['file_name'] = notion_valid_pages[i]['properties']['title']['title'][0]['plain_text']
    print("Indexing:", doc.metadata['title'])
    

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
