from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain.chains import RetrievalQA
# from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import YoutubeLoader
from uuid import uuid4
import os

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
ollama_model = "llama3.2:latest"

COLLECTION_NAME = f"yt-transcripts-{uuid4()}"

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

embeddings = OllamaEmbeddings(
    model=ollama_model,
)


qdrant_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
)

vectordb = None


def generate_video_transcript(video_url: str):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    print("transcript............", transcript)
    return transcript


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    print("splitters....", splitter.split_documents(documents))
    return splitter.split_documents(documents)


def store_in_qdrant(video_url: str):
    global vectordb
    transcript = generate_video_transcript(video_url)
    documents = split_documents(transcript)

    vectordb = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        force_recreate=True,
    )
    return vectordb


def get_context_from_store(query, k=4):
    global vectordb
    if not vectordb:
        raise ValueError("Vector store not initialized. Call `store_in_qdrant` first.")

    docs = vectordb.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    return docs_page_content
