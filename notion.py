import os
from dotenv import load_dotenv
from llama_index import download_loader
from notion_client import Client

load_dotenv()
notion_key = os.getenv("notion_key")

# recursively get pages from Notion
def get_pages():
    client = Client(auth=notion_key)
    # Get docs from IT Knowledge Base
    response = client.databases.query(
        database_id="7c86e99314014d429fc1849948ab041e"
    )
    all_pages = response['results']
    result = []
    for page in all_pages:
        result.append(page['id'])
    return result

# define function to query Notion
def get_notion_documents():
    NotionPageReader = download_loader('NotionPageReader')
    integration_token = notion_key
    page_ids = get_pages()
    reader = NotionPageReader(integration_token=integration_token)
    documents = reader.load_data(page_ids=page_ids)
    documents = [
        doc.to_langchain_format()
        for doc in documents
    ]
    return documents

docs = get_notion_documents()
print(len(docs))