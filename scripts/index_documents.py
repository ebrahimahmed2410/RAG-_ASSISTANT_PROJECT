"""
Document Ingestion, Chunking, Embedding Generation, and ChromaDB Persistence Script.
Can be executed as standalone CLI or imported into notebooks and evaluation pipelines.
"""

import os
import json
import glob
from typing import List, Dict, Any, Tuple
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions


def extract_pdf_documents(raw_dir: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extracts text page-by-page from all PDF files in the raw_dir directory.
    Returns:
        - List of page records with text, document name, and page number.
        - Summary statistics dict.
    """
    pdf_files = sorted(glob.glob(os.path.join(raw_dir, "*.pdf")))
    documents = []
    stats = {
        "num_documents": len(pdf_files),
        "total_pages": 0,
        "total_characters": 0,
        "total_words": 0,
        "failed_pages": 0,
        "documents_requiring_ocr": 0,
        "files": []
    }

    print(f"[*] Found {len(pdf_files)} PDF documents in '{raw_dir}'")

    for file_path in pdf_files:
        doc_name = os.path.basename(file_path)
        file_stats = {"filename": doc_name, "pages": 0, "chars": 0, "ocr_flag": False}
        
        try:
            reader = PdfReader(file_path)
            num_pages = len(reader.pages)
            file_stats["pages"] = num_pages
            stats["total_pages"] += num_pages

            doc_text_accum = 0

            for page_idx, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                    clean_text = " ".join(text.split())
                    chars = len(clean_text)
                    words = len(clean_text.split())

                    if chars < 30:  # Suspiciously low text might indicate scanned image / needs OCR
                        file_stats["ocr_flag"] = True

                    doc_text_accum += chars
                    stats["total_characters"] += chars
                    stats["total_words"] += words

                    documents.append({
                        "document": doc_name,
                        "page": page_idx,
                        "text": clean_text,
                        "char_count": chars,
                        "word_count": words
                    })

                except Exception as page_err:
                    print(f"[-] Error parsing {doc_name} page {page_idx}: {page_err}")
                    stats["failed_pages"] += 1

            if file_stats["ocr_flag"]:
                stats["documents_requiring_ocr"] += 1

            file_stats["chars"] = doc_text_accum
            stats["files"].append(file_stats)

        except Exception as doc_err:
            print(f"[-] Failed to open document {doc_name}: {doc_err}")

    return documents, stats


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    Splits documents into overlapping chunks with preserved metadata.
    Chunk size: 800 characters (fits coherent paragraphs and concepts).
    Chunk overlap: 150 characters (prevents loss of semantic context at boundaries).
    """
    chunks = []
    global_chunk_id = 0

    for doc in documents:
        text = doc["text"]
        doc_name = doc["document"]
        page_num = doc["page"]

        if not text:
            continue

        start = 0
        text_len = len(text)
        chunk_idx = 0

        while start < text_len:
            end = start + chunk_size
            chunk_content = text[start:end]

            # If not at the end of the text, try to break cleanly on space or punctuation
            if end < text_len:
                last_space = chunk_content.rfind(" ")
                last_period = chunk_content.rfind(". ")
                break_point = -1
                if last_period > chunk_size // 2:
                    break_point = last_period + 1
                elif last_space > chunk_size // 2:
                    break_point = last_space

                if break_point != -1:
                    chunk_content = text[start : start + break_point]
                    end = start + break_point

            chunk_content = chunk_content.strip()

            if len(chunk_content) >= 40:  # Filter out trivial fragments
                global_chunk_id += 1
                chunk_idx += 1
                chunk_id = f"{doc_name}_p{page_num}_c{chunk_idx}"

                chunks.append({
                    "id": chunk_id,
                    "text": chunk_content,
                    "document": doc_name,
                    "page": page_num,
                    "chunk_index": chunk_idx,
                    "char_count": len(chunk_content)
                })

            start += (chunk_size - chunk_overlap)
            if start >= text_len:
                break

    return chunks


def build_and_persist_vector_store(
    chunks: List[Dict[str, Any]],
    persist_dir: str,
    collection_name: str = "cs_documents",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> chromadb.Collection:
    """
    Computes embeddings using SentenceTransformers and persists to ChromaDB.
    """
    os.makedirs(persist_dir, exist_ok=True)
    print(f"[*] Initializing ChromaDB PersistentClient at '{persist_dir}'...")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )

    client = chromadb.PersistentClient(path=persist_dir)

    # Recreate or retrieve collection
    try:
        client.delete_collection(name=collection_name)
        print(f"[*] Cleared existing collection '{collection_name}' for clean rebuild.")
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    # Prepare batches
    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {"document": c["document"], "page": int(c["page"]), "chunk_id": c["id"]}
        for c in chunks
    ]

    batch_size = 64
    total = len(chunks)
    print(f"[*] Embedding and storing {total} chunks in batches of {batch_size}...")

    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        collection.add(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end]
        )
        print(f"    - Indexed chunks {i+1} to {end} / {total}")

    print(f"[OK] Vector store successfully created with {collection.count()} chunks.")
    return collection


def run_pipeline():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")
    vector_dir = os.path.join(base_dir, "backend", "data", "vector_store")

    os.makedirs(processed_dir, exist_ok=True)

    # 1. Extraction
    docs, stats = extract_pdf_documents(raw_dir)
    print("\n--- Document Extraction Statistics ---")
    print(f"Number of documents: {stats['num_documents']}")
    print(f"Number of pages: {stats['total_pages']}")
    print(f"Total characters: {stats['total_characters']}")
    print(f"Total words: {stats['total_words']}")
    print(f"Failed pages: {stats['failed_pages']}")
    print(f"Documents requiring OCR: {stats['documents_requiring_ocr']}")

    # 2. Chunking
    chunks = chunk_documents(docs, chunk_size=800, chunk_overlap=150)
    print(f"\n--- Chunking Results ---")
    print(f"Total chunks extracted: {len(chunks)}")
    if chunks:
        avg_len = sum(c["char_count"] for c in chunks) / len(chunks)
        print(f"Average chunk character length: {avg_len:.1f}")

    # Save processed chunk metadata
    processed_file = os.path.join(processed_dir, "chunks_metadata.json")
    with open(processed_file, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "chunks": chunks}, f, indent=2)
    print(f"[*] Saved chunk metadata to '{processed_file}'")

    # 3. Vector Database Build & Persistence
    collection = build_and_persist_vector_store(
        chunks=chunks,
        persist_dir=vector_dir,
        collection_name="cs_documents",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Verification Query
    test_query = "What is the time complexity of binary search?"
    print(f"\n[*] Executing verification query: '{test_query}'")
    results = collection.query(
        query_texts=[test_query],
        n_results=2,
        include=["documents", "metadatas", "distances"]
    )
    print("[*] Top verification result:")
    if results["documents"] and results["documents"][0]:
        print(f"    Source: {results['metadatas'][0][0]['document']} (Page {results['metadatas'][0][0]['page']})")
        print(f"    Distance: {results['distances'][0][0]:.4f}")
        print(f"    Snippet: {results['documents'][0][0][:180]}...")

    print("\n[OK] Pipeline completed successfully!")


if __name__ == "__main__":
    run_pipeline()
