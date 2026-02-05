import sqlite3
import pickle
import requests
import json

REMOTE_URL = "http://83.212.81.114:8000"
LOCAL_DB_PATH = "./src/VECTORSTORE/chroma/minilm/CSV/HUMMUS_SOURCE/chroma.sqlite3"
LOCAL_INDEX_PATH = "./src/VECTORSTORE/chroma/minilm/CSV/HUMMUS_SOURCE/2afb73dc-cbdb-4f3c-a2cd-242df4685f1b"

# Connect to local SQLite
conn = sqlite3.connect(LOCAL_DB_PATH)
conn.row_factory = sqlite3.Row

# Get collection info
collection = conn.execute("SELECT id, name FROM collections").fetchone()
collection_name = collection["name"]
print(f"Found collection: {collection_name}")

# Get all embeddings with their IDs
embeddings_rows = conn.execute("""
    SELECT id, embedding_id FROM embeddings ORDER BY id
""").fetchall()

# Build a map of internal id -> embedding_id
id_map = {row["id"]: row["embedding_id"] for row in embeddings_rows}
print(f"Found {len(id_map)} embeddings")

# Get metadata for each embedding
metadata_rows = conn.execute("""
    SELECT id, key, string_value, int_value, float_value, bool_value
    FROM embedding_metadata
""").fetchall()

# Group metadata by embedding id
metadata_by_id = {}
documents_by_id = {}
for row in metadata_rows:
    emb_id = row["id"]
    key = row["key"]

    # Get the value (whichever is not null)
    value = row["string_value"] or row["int_value"] or row["float_value"]
    if row["bool_value"] is not None:
        value = bool(row["bool_value"])

    if key == "chroma:document":
        documents_by_id[emb_id] = value
    else:
        if emb_id not in metadata_by_id:
            metadata_by_id[emb_id] = {}
        metadata_by_id[emb_id][key] = value

conn.close()

# Load the HNSW index metadata to get embeddings
with open(f"{LOCAL_INDEX_PATH}/index_metadata.pickle", "rb") as f:
    index_metadata = pickle.load(f)

# id_to_label maps embedding_id (UUID string) -> HNSW internal index (int)
# label_to_id maps HNSW internal index -> embedding_id (UUID string)
id_to_label = index_metadata.get("id_to_label", {})  # UUID -> int
label_to_id = index_metadata.get("label_to_id", {})  # int -> UUID

# Read the actual vectors from data_level0.bin
import struct
import numpy as np

# Read header to get dimension
with open(f"{LOCAL_INDEX_PATH}/header.bin", "rb") as f:
    header_data = f.read()
    # HNSW header format varies, but typically first values are dimension-related

# Dimension for MiniLM is typically 384
DIMENSION = 384

# Read all vectors
with open(f"{LOCAL_INDEX_PATH}/data_level0.bin", "rb") as f:
    data = f.read()

# Each vector storage includes: links data + vector data
# The format is: 4 bytes for size_links_level0, then links, then vector
# Let's calculate based on file size and count
num_vectors = len(id_to_label)
bytes_per_vector = len(data) // num_vectors if num_vectors > 0 else 0
print(f"Bytes per vector entry: {bytes_per_vector}")

# HNSW stores: uint32 size + (M0 * 2 * uint32 for links) + (dim * sizeof(float) for vector)
# M0 is typically 2*M where M=16 default, so M0=32
# So: 4 + 32*4 + 384*4 = 4 + 128 + 1536 = 1668 bytes
VECTOR_SIZE = DIMENSION * 4  # 384 floats * 4 bytes = 1536
LINK_HEADER_SIZE = bytes_per_vector - VECTOR_SIZE

vectors = {}
with open(f"{LOCAL_INDEX_PATH}/data_level0.bin", "rb") as f:
    for idx in range(num_vectors):
        # Skip the links portion
        f.read(LINK_HEADER_SIZE)
        # Read the vector
        vector_bytes = f.read(VECTOR_SIZE)
        vector = list(struct.unpack(f'{DIMENSION}f', vector_bytes))

        # Map HNSW index to embedding_id using label_to_id (int -> UUID)
        if idx in label_to_id:
            emb_id_str = label_to_id[idx]
            vectors[emb_id_str] = vector

print(f"Loaded {len(vectors)} vectors")

# Prepare data for upload
ids = []
embeddings_list = []
documents = []
metadatas = []

for internal_id, emb_id_str in id_map.items():
    if emb_id_str not in vectors:
        print(f"Warning: no vector for {emb_id_str}")
        continue

    ids.append(emb_id_str)
    embeddings_list.append(vectors[emb_id_str])
    documents.append(documents_by_id.get(internal_id, ""))
    metadatas.append(metadata_by_id.get(internal_id, {}))

print(f"Prepared {len(ids)} records for upload")

# Get or create collection on remote
resp = requests.get(f"{REMOTE_URL}/api/v1/collections/{collection_name}")
if resp.status_code == 200:
    remote_collection_id = resp.json()["id"]
    print(f"Found existing collection: {remote_collection_id}")
else:
    resp = requests.post(
        f"{REMOTE_URL}/api/v1/collections",
        json={"name": collection_name, "get_or_create": True}
    )
    resp.raise_for_status()
    remote_collection_id = resp.json()["id"]
    print(f"Created collection: {remote_collection_id}")

# Upload in batches
BATCH_SIZE = 1000
for i in range(0, len(ids), BATCH_SIZE):
    batch_ids = ids[i:i+BATCH_SIZE]
    batch_embeddings = embeddings_list[i:i+BATCH_SIZE]
    batch_documents = documents[i:i+BATCH_SIZE]
    batch_metadatas = metadatas[i:i+BATCH_SIZE]

    resp = requests.post(
        f"{REMOTE_URL}/api/v1/collections/{remote_collection_id}/add",
        json={
            "ids": batch_ids,
            "embeddings": batch_embeddings,
            "documents": batch_documents,
            "metadatas": batch_metadatas
        }
    )
    resp.raise_for_status()
    print(f"Uploaded batch {i//BATCH_SIZE + 1}: {len(batch_ids)} records")

print(f"Done! Pushed {len(ids)} records to remote collection '{collection_name}'")
