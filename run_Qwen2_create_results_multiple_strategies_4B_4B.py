import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time

import json
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoModel

# === CONFIGURATION ===
TOP_K = 10
QUERY_CSV = f"Qwen4B_rerank4B_{TOP_K}_article_ids.csv"
ARTICLE_JSON = "../database/database.json"
IMAGE_FOLDER = "../database/database_images_compressed90"
DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
OUTPUT_CACHE = "image_scores_cache_4B_rerank4B_top10.json"


check_interval_seconds = 30  # Check every 60 seconds

print(f"⏳ Waiting for file: {QUERY_CSV} to appear...")
while not os.path.exists(QUERY_CSV):
    time.sleep(check_interval_seconds)

# === Load model ===
print(f"📦 Loading Qwen2-VL model: {MODEL_ID}...")
model = AutoModel.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map=DEVICE,
    trust_remote_code=True
)
model.to(DEVICE).eval()

# === Load query and article database ===
df_queries = pd.read_csv(QUERY_CSV)
with open(ARTICLE_JSON, "r") as f:
    article_db = json.load(f)

# === Prepare cache ===
image_scores_cache = {}

# === Helper to batch image embeddings ===
def get_image_embeddings_batched(images_list_pil, model_obj, batch_size=10):
    all_image_embeds = []
    for i in range(0, len(images_list_pil), batch_size):
        batch = images_list_pil[i:i + batch_size]
        with torch.no_grad():
            batch_embs = model_obj.get_image_embeddings(images=batch, is_query=False)
        all_image_embeds.append(batch_embs)
    return torch.cat(all_image_embeds, dim=0)

# === Run caption-image scoring ===
print("🚀 Computing and caching all caption-image scores using Single-modal Qwen2...")
for _, row in tqdm(df_queries.iterrows(), total=len(df_queries)):
    qid = row["query_index"]
    qtext = row["query_text"]

    with torch.no_grad():
        text_emb = model.get_text_embeddings(texts=[qtext]).to(DEVICE)
        text_emb = torch.nn.functional.normalize(text_emb, dim=-1)

    scored_images = []
    cnt_valid_image  = 0
    cnt_valid_rank = 0 
    for rank in range(1, 11):
        
        article_id = str(row.get(f"article_id_{rank}", "#"))
        images = article_db[article_id].get("images", [])
        image_pil_list, valid_ids = [], []

        for iid in images:
            image_path = os.path.join(IMAGE_FOLDER, f"{iid}.jpg")
            if not os.path.exists(image_path):
                print(f"❌ Image not found: {image_path} for article {article_id}, rank {rank}") 
                continue
            image_pil_list.append(Image.open(image_path).convert("RGB"))
            valid_ids.append(iid)

        if not image_pil_list:
            print(f"⚠️ No valid images found for article {article_id}, rank {rank}, skipping.")
            continue               
        image_embeds = get_image_embeddings_batched(image_pil_list, model)
        image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1).to(DEVICE)

        sim_scores = torch.matmul(image_embeds, text_emb.T).squeeze(1).cpu().numpy()

        for iid, score in zip(valid_ids, sim_scores):
            scored_images.append({"image_id": iid, "score": float(score), "article_rank": rank})
            
        cnt_valid_image+= len(image_pil_list) 
        cnt_valid_rank+=1
        if cnt_valid_image > 10 and cnt_valid_rank>=3:
            break
    print("done for query:", qid, "with", cnt_valid_image, "valid images from", cnt_valid_rank)
    image_scores_cache[str(qid)] = scored_images
    
# === Save to cache ===
with open(OUTPUT_CACHE, "w") as f:
    json.dump(image_scores_cache, f,indent=2)

print(f"✅ All similarity scores saved to {OUTPUT_CACHE}")
