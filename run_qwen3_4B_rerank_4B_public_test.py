import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# ==== Config ====
QUERY_CSV = "query.csv"
JSON_PATH = "../database/database.json"
COLLECTION = "articles_4B_2560"
QDRANT_PORT = 16333
TOP_K = 10
DEVICE = "cuda:1"
DEVICE2= "cuda:2"
OUTPUT_CSV = f"Qwen4B_rerank4B_{TOP_K}_article_ids.csv"
INSTRUCTION = (
    "Given a caption describing a real-world event, determine if the document provides relevant details "
    "to identify the corresponding image. Only answer 'yes' or 'no'."
)

# ==== Load data ====
print("📥 Loading data...")
df = pd.read_csv(QUERY_CSV).dropna(subset=["query_index", "query_text"])

with open(JSON_PATH, "r") as f:
    article_db = json.load(f)

# ==== Load models ====
print("📦 Loading models...")
encoder = SentenceTransformer("Qwen/Qwen3-Embedding-4B", device=DEVICE)
client = QdrantClient(host="localhost", port=QDRANT_PORT)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-4B", padding_side='left')
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-4B").to(DEVICE2).eval()

token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")
prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
max_length = 8192

def format_instruction(instruction, query, doc):
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

def process_inputs(pairs):
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(DEVICE)
    return inputs

@torch.no_grad()
def compute_logits(inputs):
    logits = model(**inputs).logits[:, -1, :]
    scores = torch.stack([logits[:, token_false_id], logits[:, token_true_id]], dim=1)
    probs = torch.nn.functional.log_softmax(scores, dim=1)
    return probs[:, 1].exp().tolist()

# ==== Process queries ====
print(f"\n🚀 Processing {len(df)} queries...")
submission_rows = []

def batched(iterable, batch_size):
    """Yield successive batches from iterable"""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

@torch.no_grad()
def compute_batched_logits(pairs, batch_size=5):
    all_scores = []
    for batch in batched(pairs, batch_size):
        inputs = process_inputs(batch).to(DEVICE2)
        scores = compute_logits(inputs)
        all_scores.extend(scores)
    return all_scores

for _, row in tqdm(df.iterrows(), total=len(df)):
    query_index = row["query_index"]
    query_text = row["query_text"]

    q_emb = encoder.encode(query_text, prompt_name="query", convert_to_tensor=False)
    hits = client.search(collection_name=COLLECTION, query_vector=q_emb, limit=TOP_K)

    # Prepare articles
    docs, ids = [], []
    for hit in hits:
        aid = str(hit.payload["article_id"])
        article = article_db.get(aid, {})
        title = article.get("title", "")
        date_raw = article.get("date", "")
        content = article.get("content", "")
        try:
            date_obj = datetime.fromisoformat(date_raw.replace("Z", "+00:00"))
            date_str = date_obj.strftime("%B %d, %Y")
        except Exception:
            date_str = date_raw or "Unknown"
        article_text = f"Title: {title}\nDate: {date_str}\nContent: {content}"
        docs.append(article_text)
        ids.append(aid)

    # Rerank
    pairs = [format_instruction(INSTRUCTION, query_text, doc) for doc in docs]
    # inputs = process_inputs(pairs)
    # scores = compute_logits(inputs)
    scores = compute_batched_logits(pairs, batch_size=5) 

    reranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
    reranked_ids = [i for i, _ in reranked][:TOP_K]

    # Save result row
    row_data = {
        "query_index": query_index,
        "query_text": query_text,
    }
    for i, aid in enumerate(reranked_ids, 1):
        row_data[f"article_id_{i}"] = aid
    # breakpoint()
    submission_rows.append(row_data)
    # if (len(submission_rows)>3):
    #     break

# ==== Save to CSV ====
submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved to {OUTPUT_CSV}")
