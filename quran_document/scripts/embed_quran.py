import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. Menentukan lokasi project
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "quran_documents.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

EMBEDDINGS_FILE = OUTPUT_DIR / "quran_embeddings.npy"
IDS_FILE = OUTPUT_DIR / "quran_embedding_ids.json"


# ============================================================
# 2. Load dokumen Quran
# ============================================================

print(f"Loading documents from: {INPUT_FILE}")

with INPUT_FILE.open("r", encoding="utf-8") as file:
    documents = json.load(file)

print(f"Total documents: {len(documents)}")


# ============================================================
# 3. Validasi jumlah dokumen
# ============================================================

expected_documents = 6236

if len(documents) != expected_documents:
    raise ValueError(
        f"Jumlah dokumen tidak sesuai. "
        f"Ditemukan {len(documents)}, seharusnya {expected_documents}."
    )


# ============================================================
# 4. Mengambil embedding_text
# ============================================================

embedding_texts = []

for document in documents:
    embedding_text = document["embedding_text"]

    # E5 menggunakan prefix "passage:" untuk dokumen
    embedding_texts.append(f"passage: {embedding_text}")


# ============================================================
# 5. Load embedding model
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer("intfloat/multilingual-e5-base")

print("Model loaded.")


# ============================================================
# 6. Generate embeddings
# ============================================================

print("Generating embeddings...")

embeddings = model.encode(
    embedding_texts,
    batch_size=16,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)

print("Embedding generation completed.")


# ============================================================
# 7. Validasi hasil embedding
# ============================================================

print(f"Embedding shape: {embeddings.shape}")

expected_dimension = 768

if embeddings.shape != (expected_documents, expected_dimension):
    raise ValueError(
        f"Shape embedding tidak sesuai. "
        f"Ditemukan {embeddings.shape}, "
        f"seharusnya ({expected_documents}, {expected_dimension})."
    )


# ============================================================
# 8. Simpan embedding
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.save(EMBEDDINGS_FILE, embeddings)

print(f"Embeddings saved to: {EMBEDDINGS_FILE}")


# ============================================================
# 9. Simpan hubungan index embedding -> document_id
# ============================================================

document_ids = [
    document["document_id"]
    for document in documents
]

with IDS_FILE.open("w", encoding="utf-8") as file:
    json.dump(document_ids, file, ensure_ascii=False, indent=2)

print(f"Document IDs saved to: {IDS_FILE}")


# ============================================================
# 10. Informasi akhir
# ============================================================

print("\n=== DONE ===")
print(f"Documents : {len(documents)}")
print(f"Dimension : {embeddings.shape[1]}")
print(f"Shape     : {embeddings.shape}")
print(f"Embeddings: {EMBEDDINGS_FILE}")
print(f"IDs       : {IDS_FILE}")