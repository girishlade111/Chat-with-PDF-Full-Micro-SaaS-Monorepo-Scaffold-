import fitz # PyMuPDF
from typing import List, Dict


class Chunk:
def __init__(self, text, page_start, page_end, para_index):
self.text = text
self.page_start = page_start
self.page_end = page_end
self.para_index = para_index


# Two-tier chunking: structural -> semantic


def extract_chunks(path: str, target_tokens: int = 850, overlap_tokens: int = 180) -> List[Chunk]:
doc = fitz.open(path)
chunks: List[Chunk] = []
for page_idx in range(len(doc)):
page = doc[page_idx]
text = page.get_text("text")
paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
buf = []
token_est = 0
para_index = 0
for p in paragraphs:
t = len(p.split())
if token_est + t > target_tokens and buf:
joined = "\n\n".join(buf)
chunks.append(Chunk(joined, page_idx+1, page_idx+1, para_index))
# overlap
overlap_words = " ".join(joined.split()[-overlap_tokens:])
buf = [overlap_words, p]
token_est = len(overlap_words.split()) + t
para_index += 1
else:
buf.append(p)
token_est += t
para_index += 1
return chunks
