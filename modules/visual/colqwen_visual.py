from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

import pymupdf
from pathlib import Path # Will add a function to check the path and shi later
import fitz

paper_path = "../../data/paper/attention_is_all_you_need.pdf"
output_path = Path("../../data/parsed")
copali_image_path = Path("../../data/parsed/paper_images")

zoom = 2.0  
matrix = fitz.Matrix(zoom, zoom)

paper = pymupdf.open(str(paper_path))

page_1 = paper.load_page(1)
page_1.number

model_name = "vidore/colqwen2-v1.0"

model = ColQwen2_5.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0" ,
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()

processor = ColQwen2_5_Processor.from_pretrained(model_name)