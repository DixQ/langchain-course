import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter


load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    print(os.environ['PINECONE_API_KEY'])
    loader = TextLoader('/home/aerill/projects/langchain-course/mediumblog1.txt', encoding='UTF-8')
    document = loader.load()

    print("splitting...")
    test_spliiter = CharacterTextSplitter(chunk_size=1000, chunk_overlab = 0)
    texts = test_spliiter.split_documents(document)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    print("ingesting...")
    PineconeVectorStore.from_documents(
        texts, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("finish")