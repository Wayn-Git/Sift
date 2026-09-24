from langchain_mineru import MinerULoader
import pymupdf

import fitz

import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path # Will add a function to check the path and shi later
# importing the paper and checking for the path of the paper for further convertion and loading
main_path = Path("../data")
data_path = Path("../data/raw/2309.15217v2.pdf")
parsed_data_path = Path("../data/parsed")

# Later implimentation
# copali_image_path = Path("../../data/parsed/paper_images")
main_path.mkdir(parents=True, exist_ok=True)
# data_path.mkdir(parents=True, exist_ok=True) Shouldn't be happening
parsed_data_path.mkdir(parents=True, exist_ok=True)
# output_path.mkdir(parents=True, exist_ok=True)
# Loading the paper with minerU
loader = MinerULoader(source=str(data_path), split_pages=True) # Need to keep the split pages True because imma use copali later that'll analyze the pdf page by page for visual elements

docs = loader.load()

print(docs[3])

print(len(docs))

for i, doc in enumerate(docs[:3]):
    print(f"Document {i}")
    print(f"Metadata: {doc.metadata}")
    print(f"Content:\n{doc.page_content[:1000]}")
for i, doc in enumerate(docs, start=1):
    page_path = parsed_data_path / f"page_{i:03d}.md" # Three zeroes before the page number

    page_path.write_text(
        doc.page_content,
    )

print(f"Saved {len(docs)} pages to {parsed_data_path}")