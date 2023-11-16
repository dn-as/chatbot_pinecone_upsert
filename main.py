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

####################################################
#                                                  #
# This file upserts documents in data to pinecone. #
#                                                  #
####################################################

load_dotenv()
# os.environ['OPENAI_API_KEY'] = os.getenv('api_key')  # platform.openai.com
openai.api_key = os.getenv('api_key')

loader = UnstructuredReader()

# find API key in console at app.pinecone.io
os.environ['PINECONE_API_KEY'] = os.getenv('pinecone_api_key')
# environment is found next to API key in the console
os.environ['PINECONE_ENVIRONMENT'] = os.getenv('pinecone_env')

# initialize connection to pinecone
pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment=os.environ['PINECONE_ENVIRONMENT']
)

# # Load docs
# sampro_webhelp_documents = SimpleDirectoryReader(input_dir="./SamproWebhelp/", file_extractor={".htm": UnstructuredReader()}).load_data()

# # create the index if it does not exist already
# sampro_webhelp_index_name = 'sampro-webhelp'
# if sampro_webhelp_index_name not in pinecone.list_indexes():
#     pinecone.create_index(
#         sampro_webhelp_index_name,
#         dimension=1536,
#         metric='cosine'
#     )

# # connect to the index
# sampro_webhelp_pinecone_index = pinecone.Index(sampro_webhelp_index_name)

# # we can select a namespace (acts as a partition in an index)
# namespace = '' # default namespace

# sampro_webhelp_vector_store = PineconeVectorStore(pinecone_index=sampro_webhelp_pinecone_index)

# # setup our storage (vector db)
# sampro_webhelp_storage_context = StorageContext.from_defaults(
#     vector_store=sampro_webhelp_vector_store
# )

# # setup the index/query process, ie the embedding model (and completion if used)
# embed_model = OpenAIEmbedding(model='text-embedding-ada-002', embed_batch_size=100)
# service_context = ServiceContext.from_defaults(embed_model=embed_model)

# sampro_webhelp_index = GPTVectorStoreIndex.from_documents(
#     documents=sampro_webhelp_documents, 
#     storage_context=sampro_webhelp_storage_context,
#     service_context=service_context
# )

# setup the index/query process, ie the embedding model (and completion if used)
embed_model = OpenAIEmbedding(model='text-embedding-ada-002', embed_batch_size=100)
service_context = ServiceContext.from_defaults(embed_model=embed_model)

# Readers
PDFReader = download_loader("PDFReader")
BSReader = download_loader("BeautifulSoupWebReader")

# Load docs
def upsert_docs(input_dir: str, namespace: str, index_name: str):
    print(f"Building namespace: {namespace} from {input_dir} under index {index_name}...\n")
    # documents = SimpleDirectoryReader(input_dir=input_dir, file_extractor=file_extractor).load_data()
    documents = SimpleDirectoryReader(input_dir=input_dir).load_data()

    namespace = namespace

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
    print(f"Done building {namespace} !\n")

dirsAndNames = [
    ["./data/SamproWebhelp/", "sampro-webhelp"],
    ["./data/Policies/", "policies"],
    ["./data/GeneralInfo", "general-info"],
    ["./data/SOPs/", "sops"]
]
# indexNames = ["sampro-webhelp", "policies", "general-info", "sops"]
# fileExtractors = {
#     ".htm": BSReader(), 
#     ".html": BSReader(), 
#     ".txt": UnstructuredReader(),
#     ".pdf": PDFReader()
# }

for dirAndName in dirsAndNames:
    upsert_docs(input_dir=dirAndName[0], namespace=dirAndName[1], index_name="dnas-wiki")
    # upsert_docs(input_dir=dirAndName[0], file_extractor=fileExtractors, index_name=dirAndName[1])


# # Remove index
# for indexName in pinecone.list_indexes():
#     pinecone.delete_index(indexName)