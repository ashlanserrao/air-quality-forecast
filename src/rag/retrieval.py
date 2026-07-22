# src/rag/retrieval.py — query the PM2.5 knowledge base with stage-aware retrieval
import re

import chromadb
from chromadb.utils import embedding_functions

from rag.ingest import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL_NAME

STAGE_QUERY_RE = re.compile(r"stage\s*(?:[-–]\s*)?(iv|iii|ii|i|[1-4])\b", re.IGNORECASE)
DIGIT_TO_ROMAN = {"1": "I", "2": "II", "3": "III", "4": "IV"}

# Keyword -> filename substring, for queries that explicitly name a source.
SOURCE_KEYWORDS = {
    "who": "9789240034433-eng.pdf",
    "world health organization": "9789240034433-eng.pdf",
    "grap": "GRAP Schedule7e3bd726-7308-484f-995f-7cb3fa705b8d.pdf",
    "graded response": "GRAP Schedule7e3bd726-7308-484f-995f-7cb3fa705b8d.pdf",
    "epa": "pm-naaqs-air-quality-index-fact-sheet.pdf",
    "naaqs": "pm-naaqs-air-quality-index-fact-sheet.pdf",
    "cpcb": "cpcb_about_national_aqi.pdf",
}
SOURCE_KEYWORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in SOURCE_KEYWORDS) + r")\b", re.IGNORECASE
)

WHO_DOC = "9789240034433-eng.pdf"

# WHO 2021 Global Air Quality Guidelines, PM2.5. The document's numeric table is
# a near-textless, number-dense chunk that all-MiniLM ranks 8-13 for WHO queries
# (it loses to fluent prose that discusses guidelines without containing any), so
# retrieval cannot reliably surface it. These values are stable, authoritative,
# and tiny — hardcoding them, exactly as aqi.py hardcodes the CPCB breakpoints,
# is more robust than iterating further on the embedding. Injected as the top
# priority chunk on any WHO-specific query so the numbers are always in context.
WHO_PM25_GUIDELINE_CHUNK = {
    "id": "who_pm25_guideline_hardcoded",
    "text": (
        "WHO 2021 Global Air Quality Guidelines - PM2.5 (fine particulate matter, "
        "< 2.5 micrometres), concentrations in ug/m3:\n"
        "- Air Quality Guideline (AQG) level: 5 ug/m3 annual mean; "
        "15 ug/m3 24-hour mean.\n"
        "- Interim targets, annual mean: IT-1 = 35, IT-2 = 25, IT-3 = 15, IT-4 = 10.\n"
        "- Interim targets, 24-hour mean: IT-1 = 75, IT-2 = 50, IT-3 = 37.5, IT-4 = 25.\n"
        "The AQG level is the long-term health-based goal; interim targets are "
        "milestones for highly polluted areas. This is a WHO health guideline and "
        "is not the same scale as, and is far stricter than, the CPCB National AQI "
        "categories (Good/Satisfactory/Moderate/...)."
    ),
    "source_document": WHO_DOC,
    "section_title": "WHO 2021 PM2.5 guideline levels",
    "page_start": 8,
    "page_end": 8,
    "chunk_type": "reference",
}

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL_NAME
        )
        _collection = client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)
    return _collection


def _detect_stage(query: str) -> str | None:
    m = STAGE_QUERY_RE.search(query)
    if not m:
        return None
    raw = m.group(1).upper()
    roman = DIGIT_TO_ROMAN.get(raw, raw)
    return f"Stage {roman}"


def _detect_source(query: str) -> str | None:
    m = SOURCE_KEYWORD_RE.search(query)
    if not m:
        return None
    return SOURCE_KEYWORDS[m.group(1).lower()]


def _result_to_chunks(results, source_key="id") -> list[dict]:
    chunks = []
    for doc, meta, chunk_id in zip(
        results["documents"][0], results["metadatas"][0], results["ids"][0]
    ):
        chunks.append(
            {
                "id": chunk_id,
                "text": doc,
                "source_document": meta["source_document"],
                "section_title": meta["section_title"],
                "page_start": meta["page_start"],
                "page_end": meta["page_end"],
                "chunk_type": meta["chunk_type"],
            }
        )
    return chunks


def get_health_guidance(query: str, top_k: int = 3) -> list[dict]:
    """Retrieve health/regulatory guidance chunks relevant to `query`, each
    with source_document and section_title for citation.

    Two metadata-filtered lookups run alongside the normal vector search and
    are prioritized ahead of it when they match, since a shared-vocabulary
    document set (all GRAP stages, all standards docs) can otherwise swamp
    the weaker semantic signal of a stage number or source name:
      - a specific GRAP stage (digit or roman numeral) -> filter on section_title
      - a named source (WHO, GRAP, EPA/NAAQS, CPCB)    -> filter on source_document
    """
    collection = _get_collection()

    vector_results = collection.query(query_texts=[query], n_results=top_k)
    vector_chunks = _result_to_chunks(vector_results)

    priority_chunks = []

    source_file = _detect_source(query)
    if source_file == WHO_DOC:
        # The WHO numeric table cannot be reliably retrieved (see the chunk's
        # docstring), so inject the authoritative values directly, ahead of
        # everything, guaranteeing they occupy a top_k slot. Real WHO prose the
        # retriever finds still fills the remaining slots below.
        priority_chunks.append(dict(WHO_PM25_GUIDELINE_CHUNK))

    stage_label = _detect_stage(query)
    if stage_label is not None:
        filtered = collection.query(
            query_texts=[query], n_results=1, where={"section_title": stage_label}
        )
        priority_chunks.extend(_result_to_chunks(filtered))

    if source_file is not None:
        filtered = collection.query(
            query_texts=[query],
            n_results=min(top_k, 2),
            where={"source_document": source_file},
        )
        priority_chunks.extend(_result_to_chunks(filtered))

    merged = []
    seen_ids = set()
    for chunk in priority_chunks + vector_chunks:
        if chunk["id"] in seen_ids:
            continue
        seen_ids.add(chunk["id"])
        merged.append(chunk)
        if len(merged) == top_k:
            break

    for chunk in merged:
        chunk.pop("id", None)
    return merged
