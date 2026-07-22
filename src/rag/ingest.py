# src/rag/ingest.py — extract, chunk, embed, and store the PDF knowledge base in Chroma
import os
import re
import glob

import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_DIR = os.path.join(_REPO_ROOT, "data", "knowledge_base")
CHROMA_DIR = os.path.join(_REPO_ROOT, "data", "chroma_db")
COLLECTION_NAME = "pm25_knowledge_base"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

GRAP_STAGE_HEADER_RE = re.compile(
    r"Stage\D{0,5}(IV|III|II|I)\D{0,6}(Poor|Very Poor|Severe\s*\+|Severe)",
    re.IGNORECASE,
)
STAGE_LABELS = ["Stage I", "Stage II", "Stage III", "Stage IV"]


def _bbox_overlaps(a, b) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


MIN_TABLE_DATA_ROWS = 1


def _table_data_rows(table) -> int:
    """Count rows past the header that carry any content. PyMuPDF sometimes
    detects a table it can't actually parse (unruled or whitespace-aligned
    layouts) and returns just a header skeleton — that must not be treated as
    a successful extraction, or the real content is silently discarded."""
    rows = table.extract()
    return sum(1 for row in rows[1:] if any((c or "").strip() for c in row))


def _table_to_markdown(table) -> str:
    rows = table.extract()
    lines = []
    for i, row in enumerate(rows):
        cells = [(c or "").strip().replace("\n", " ") for c in row]
        lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
    return "\n".join(lines)


def _chunk_grap_doc(path: str, filename: str) -> list[dict]:
    """Chunk the GRAP schedule by stage (I-IV), merging each stage's tables
    across however many pages it spans into a single whole chunk."""
    doc = fitz.open(path)

    boundary_pages = []  # 0-indexed page numbers where a new stage section starts
    for i in range(1, len(doc)):  # skip page 0 (intro, mentions all stages)
        if GRAP_STAGE_HEADER_RE.search(doc[i].get_text()):
            boundary_pages.append(i)

    if len(boundary_pages) != 4:
        # Fallback: couldn't cleanly find 4 stage boundaries — chunk per page instead
        # of silently mis-attributing content to the wrong stage.
        chunks = []
        for i in range(len(doc)):
            text = doc[i].get_text().strip()
            if text:
                chunks.append(
                    {
                        "text": text,
                        "source_document": filename,
                        "section_title": f"page {i + 1}",
                        "page_start": i + 1,
                        "page_end": i + 1,
                        "chunk_type": "text",
                    }
                )
        doc.close()
        return chunks

    chunks = []
    ranges = list(zip(boundary_pages, boundary_pages[1:] + [len(doc)]))
    for label, (start, end) in zip(STAGE_LABELS, ranges):
        parts = [doc[start].get_text().strip()]  # header + AQI range line
        for i in range(start, end):
            good = [
                t
                for t in doc[i].find_tables().tables
                if _table_data_rows(t) >= MIN_TABLE_DATA_ROWS
            ]
            for t in good:
                parts.append(_table_to_markdown(t))
            # Same guard as the generic chunker: a page whose tables all failed
            # to parse would otherwise contribute nothing at all to its stage.
            if not good and i != start:
                parts.append(doc[i].get_text().strip())
        chunks.append(
            {
                "text": "\n\n".join(p for p in parts if p),
                "source_document": filename,
                "section_title": label,
                "page_start": start + 1,
                "page_end": end,
                "chunk_type": "table",
            }
        )

    doc.close()
    return chunks


# A whole page of prose is too coarse a unit: a dense run of numbers (e.g. the
# WHO guideline levels) gets averaged out by the surrounding narrative and the
# chunk stops matching queries about those numbers. Splitting keeps each
# embedding focused on a smaller span of text.
MAX_CHUNK_CHARS = 600
MIN_TAIL_CHARS = 120  # a shorter trailing piece is merged back, not left alone


def _split_lines(lines: list[str]) -> list[str]:
    """Group lines into chunks of at most MAX_CHUNK_CHARS, breaking only on
    line boundaries so numbers stay attached to their labels."""
    segments = []
    current = []
    size = 0
    for line in lines:
        if current and size + len(line) + 1 > MAX_CHUNK_CHARS:
            segments.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        segments.append("\n".join(current))

    if len(segments) > 1 and len(segments[-1]) < MIN_TAIL_CHARS:
        segments[-2] = segments[-2] + "\n" + segments.pop()

    return [s for s in (seg.strip() for seg in segments) if s]


def _chunk_generic_doc(path: str, filename: str) -> list[dict]:
    """Chunk a generic PDF: each detected table is its own chunk; remaining
    prose on each page (tables excluded) is one chunk per page, titled by
    the largest-font line on that page."""
    doc = fitz.open(path)
    chunks = []

    for i in range(len(doc)):
        page = doc[i]
        tabs = page.find_tables()

        # Only tables we actually extracted rows from get their own chunk and
        # have their region excluded from the prose chunk. A failed detection
        # (header-only skeleton) is treated as no table at all: its bbox stays
        # *in* the prose, so the page's real content — often the numbers that
        # matter most — still reaches the collection instead of being excluded
        # from the prose and captured nowhere.
        good_tables = [t for t in tabs.tables if _table_data_rows(t) >= MIN_TABLE_DATA_ROWS]
        table_bboxes = [t.bbox for t in good_tables]

        for t in good_tables:
            md = _table_to_markdown(t)
            if md.strip():
                chunks.append(
                    {
                        "text": md,
                        "source_document": filename,
                        "section_title": f"table (page {i + 1})",
                        "page_start": i + 1,
                        "page_end": i + 1,
                        "chunk_type": "table",
                    }
                )

        page_dict = page.get_text("dict")
        lines = []
        max_size = 0.0
        heading = None
        for block in page_dict.get("blocks", []):
            bbox = block.get("bbox")
            if bbox and any(_bbox_overlaps(bbox, tb) for tb in table_bboxes):
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                line_text = "".join(s["text"] for s in spans).strip()
                if not line_text:
                    continue
                size = max(s["size"] for s in spans)
                lines.append(line_text)
                if size > max_size and len(line_text) < 120:
                    max_size = size
                    heading = line_text

        for prose in _split_lines(lines):
            chunks.append(
                {
                    "text": prose,
                    "source_document": filename,
                    "section_title": heading or f"page {i + 1}",
                    "page_start": i + 1,
                    "page_end": i + 1,
                    "chunk_type": "text",
                }
            )

    doc.close()
    return chunks


def extract_all_chunks() -> list[dict]:
    chunks = []
    for path in sorted(glob.glob(os.path.join(KB_DIR, "*.pdf"))):
        filename = os.path.basename(path)
        if "grap" in filename.lower():
            chunks.extend(_chunk_grap_doc(path, filename))
        else:
            chunks.extend(_chunk_generic_doc(path, filename))
    return chunks


def build_collection(reset: bool = True):
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL_NAME
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embed_fn
    )

    chunks = extract_all_chunks()
    if not chunks:
        raise RuntimeError(f"No chunks extracted from {KB_DIR}")

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    # Prepend the section title to what gets embedded (not to the stored/displayed
    # text) so it's a strong lexical/semantic anchor rather than sidecar metadata
    # the embedding model has no reason to weight heavily.
    documents = [f"{c['section_title']}\n\n{c['text']}" for c in chunks]
    metadatas = [
        {
            "source_document": c["source_document"],
            "section_title": c["section_title"],
            "page_start": c["page_start"],
            "page_end": c["page_end"],
            "chunk_type": c["chunk_type"],
        }
        for c in chunks
    ]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return collection, len(chunks)


if __name__ == "__main__":
    collection, n = build_collection()
    print(f"Ingested {n} chunks into Chroma collection '{COLLECTION_NAME}' at {CHROMA_DIR}")
