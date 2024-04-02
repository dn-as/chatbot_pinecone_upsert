from llama_index.core import download_loader
from llama_index.core import Document
from dotenv import load_dotenv
import os
import pinecone
import openai
from notion_client import Client
from pinecone_text.sparse import BM25Encoder
import re
import logging

#####################################################
#                                                   #
# This file upserts noition KB in data to pinecone. #
#                                                   #
#####################################################

load_dotenv()
openai.api_key = os.getenv('api_key')
notion_key = os.getenv("test_notion_key")

# find API key in console at app.pinecone.io
os.environ['PINECONE_API_KEY'] = os.getenv('pinecone_api_key')
# environment is found next to API key in the console
os.environ['PINECONE_ENVIRONMENT'] = os.getenv('pinecone_env')

# Configure the logging system
logging.basicConfig(filename='upsertNotion.log',  # Log file path
                    filemode='w',  # 'a' for append, 'w' for overwrite
                    format='%(asctime)s - %(levelname)s - %(message)s',  # Format of log messages
                    level=logging.INFO)  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

# initialize connection to pinecone
pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment=os.environ['PINECONE_ENVIRONMENT']
)

# Delete old vectors
index = pinecone.Index("notion-kb-spare-dense")
delete_response = index.delete(delete_all=True, namespace="notion-kb")
if len(delete_response) == 0:
    logging.info("Delete successful, Index: notion-kb-spare-dense, Namespace: notion-kb")
    print("Delete successful")

# setup Spare vector encoder
bm25 = BM25Encoder()

# Building namespace
index_name = "notion-kb-spare-dense"
print(f"Building index {index_name} ...\n")
logging.info(f"Building index {index_name} ...\n")
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

notion_pages = results

# Remove Notion pages that do not have a title
notion_valid_pages = []
notion_valid_page_titles = []

for page in notion_pages:
    page_title_object = client.pages.properties.retrieve(page_id=page['id'], property_id="title")
    # Filter out pages with no title
    if len(page_title_object['results']) != 0:
        page_title = page_title_object['results'][0]['title']['plain_text']
        if page_title != '' and page_title != 'SOPs' and page_title != "How To’s" and page_title != 'Other':
            notion_valid_pages.extend([page])
            notion_valid_page_titles.extend([page_title])

print("Total count of valid Notion pages:", len(notion_valid_pages))
logging.info("Total count of valid Notion pages: %s", len(notion_valid_pages))

NotionPageReader = download_loader('NotionPageReader')
notion_pages_ids = [str(page['id']) for page in notion_valid_pages]
notion_pages_urls = [str(page['url']) for page in notion_valid_pages]

reader = NotionPageReader(integration_token=notion_key)
documents = reader.load_data(notion_pages_ids)

# Add metadata
for i, doc in enumerate(documents):
    doc.metadata['url'] = notion_pages_urls[i]
    doc.metadata['title'] = notion_valid_page_titles[i]

def divide_string_with_overlap(text, chunk_size=200, overlap=50):
    # Split the text into words
    words = text.split()
    
    # Check if the text can be divided into chunks with the given size and overlap
    if len(words) < chunk_size:
        return [text]  # Return the whole text if it's shorter than a chunk size
    
    chunks = []
    i = 0
    while i + chunk_size <= len(words):
        # Extract the chunk of words and join them into a string
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        # Move the starting index of the next chunk, considering the overlap
        i += (chunk_size - overlap)
        
    # Handling the last chunk if there are remaining words that were not covered
    # due to the division not fitting perfectly
    if i < len(words):
        last_chunk = " ".join(words[-chunk_size:])
        if last_chunk not in chunks:  # Avoid duplication if the last chunk is already added
            chunks.append(last_chunk)
    
    return chunks

def normalize_whitespace(text):
    # This pattern matches any sequence of whitespace characters (spaces, tabs, newlines, etc.)
    pattern = r'\s+'
    # Replace the matched patterns with a single space
    normalized_text = re.sub(pattern, ' ', text)
    return normalized_text

# Filter out pages that have empty body
# Divide documents into chunks
valid_documents = []
for i, doc in enumerate(documents):
    if doc.text.strip() != "":
        # valid_documents.append(doc)
        meta = doc.metadata
        text = doc.text
        text = normalize_whitespace(text) # remove whitespaces, tabs, new line char, etc.
        chunks = divide_string_with_overlap(text=text)
        print("Indexing title:", meta['title'])
        logging.info("Indexing title: %s", meta['title'])
        for chunk in chunks:
            valid_documents.append(Document(text=chunk.strip(), metadata=meta))
      
# report chunks of text
print("Count of chunks of text:", len(valid_documents))
logging.info("Count of chunks of text: %s", len(valid_documents))

# create the index if it does not exist already
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=1536,
        metric='dotproduct',
        pod_type="s1"
    )

# connect to the index
pineconeIndex = pinecone.Index(index_name)

# fit sparse model to title
bm25.fit(corpus=[doc.metadata['title'] for doc in valid_documents])

# create sparse embedding
spare_embeds = bm25.encode_documents([(doc.metadata['title'] + doc.text) for doc in valid_documents])
  
# create dense embedding
dense_embeds = [[embed.embedding for embed in openai.embeddings.create(model="text-embedding-ada-002", input=(doc.metadata['title'] + doc.text)).data][0] for doc in valid_documents]

# create node content
text_content = [doc.text for doc in valid_documents]

# create metadata
metadata = [doc.metadata for doc in valid_documents]

# create upserts to vector DB
upserts = []
for i, (dense, sparse, text, meta) in enumerate(zip(dense_embeds, spare_embeds, text_content, metadata)):
    meta['body'] = text
    upserts.append({
        'id': str(i),
        'values': dense,
        'sparse_values': sparse,
        'metadata': meta,
    })

print("Upserts length:", len(upserts))
logging.info("Upserts length: %s", len(upserts))

# upsert into vector db and report status
def divide_list(input_list, chunk_size=25):
    """
    Divides a list into chunks of specified size.

    Parameters:
    - input_list: The list to be divided.
    - chunk_size: The maximum size of each chunk.

    Returns:
    - A list of lists, where each inner list has at most `chunk_size` elements.
    """
    # Using list comprehension to create chunks
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

# Example usage
upserts_batch = divide_list(upserts, 25)
for batch in upserts_batch:
    pineconeIndex.upsert(vectors=batch, namespace="notion-kb")
pineconeIndex.describe_index_stats()

print(f"Done building {index_name}!\n{len(documents)} notion pages upserted.")
logging.info(f"Done building {index_name}!\n{len(documents)} notion pages upserted.")
