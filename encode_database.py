import json
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import numpy as np

# === CONFIG ===
DATABASE_JSON = Path("database/database.json")
MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
COLLECTION_NAME = "articles_4B_2560"
VECTOR_SIZE = 2560
MAX_LEN = 2560  # ✅ Token limit updated here
QDRANT_PORT = 16333
DEVICE = "cuda:1"
BATCH_SIZE = 1

# === LOAD MODEL ===
print("🧠 Loading model...")
model = SentenceTransformer(
    MODEL_NAME,
    device=DEVICE,
    tokenizer_kwargs={"padding_side": "left"}
)

# === CONNECT TO QDRANT ===
client = QdrantClient(host="localhost", port=QDRANT_PORT)

# === DELETE + RECREATE COLLECTION ===
print(f"🗑️ Resetting collection '{COLLECTION_NAME}'...")
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
)

# === LOAD DATABASE ===
print("📖 Loading database.json...")
with open(DATABASE_JSON, "r", encoding="utf-8") as f:
    db = list(json.load(f).items())  # [(article_id, article_data), ...]

# === ENCODE + UPSERT IN BATCHES ===
print("🚀 Encoding and inserting into Qdrant...")
total = 0
for i in tqdm(range(0, len(db), BATCH_SIZE), desc="Batches"):
    batch = db[i:i + BATCH_SIZE]
    texts, article_ids = [], []

    for article_id, article in batch:
        title = article.get("title", "")
        date = article.get("date", "")
        content = article.get("content", "")
        full_text = f"Title: {title}\nDate: {date}\nContent: {content}".strip()

        texts.append(full_text)
        article_ids.append(article_id)

    try:
        embs = model.encode(
            texts,
            convert_to_tensor=False,
            device=DEVICE,

            truncation=True,
            batch_size=BATCH_SIZE
        )
   
        points = [
            PointStruct(
                id=i + j,
                vector=np.array(embs[j], dtype=np.float32).tolist(),
                payload={"article_id": article_ids[j]}
            )
            for j in range(len(embs))
        ]

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total += len(points)

    except Exception as e:
        print(f"❌ Batch {i // BATCH_SIZE} failed: {e}")

print(f"✅ All done! Total inserted: {total}")
