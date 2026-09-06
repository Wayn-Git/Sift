import glob

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, PARSED_DIR


def load_and_chunk(
    parsed_dir: str = PARSED_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list:
    """Load .md files from parsed_dir and split into document chunks."""
    pages_path = glob.glob(f"{parsed_dir}/*.md")

    docs = []
    for page in pages_path:
        text_loader = TextLoader(page, encoding="utf-8")
        docs.extend(text_loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)
