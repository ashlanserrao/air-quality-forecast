# scripts/test_rag_retrieval.py — eyeball retrieval quality before wiring RAG into the agent
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chromadb
from chromadb.utils import embedding_functions

from rag.ingest import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL_NAME

QUERIES = [
    "what should I do at AQI 350",
    "what is GRAP stage 3",
    "what are WHO PM2.5 guideline levels",
    "what changed in the 2024 US AQI breakpoints",
]

TOP_K = 3


def main():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL_NAME
    )
    collection = client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)

    for query in QUERIES:
        print("=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)
        results = collection.query(query_texts=[query], n_results=TOP_K)

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
            snippet = doc[:300].replace("\n", " ")
            print(
                f"\n[{rank}] distance={dist:.4f} "
                f"source={meta['source_document']} "
                f"section={meta['section_title']!r} "
                f"pages={meta['page_start']}-{meta['page_end']} "
                f"type={meta['chunk_type']}"
            )
            print(f"    {snippet}...")
        print()


if __name__ == "__main__":
    main()
