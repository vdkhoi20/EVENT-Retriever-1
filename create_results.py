import json
import pandas as pd
import numpy as np
import os

model_rerank="8B_8B"
# === CONFIGURATION ===
INPUT_CACHE_FILE = "image_scores_cache_8B_rerank8B_top10.json" # File cache điểm số của bạn
TOP_IMAGES = 10 # Số lượng ảnh top muốn lấy cho mỗi truy vấn

# Định nghĩa các giá trị MAX_IMAGES_PER_ARTICLE để kiểm tra
# Lưu ý: Giá trị này sẽ được dùng để lọc ảnh từ cache trước khi áp dụng Strategy B.
# Nếu quá trình caching của bạn đã giới hạn số ảnh (ví dụ: `cnt_valid_image > 10`),
# thì việc tăng giá trị này ở đây sẽ không "thêm" ảnh mà không có trong cache.
MAX_IMAGES_PER_ARTICLE_VALUES = [2,5,7]

# Các trọng số (không dùng cho Strategy B thuần túy, nhưng giữ lại nếu bạn có kế hoạch dùng cho tương lai)
WEIGHTS = {1: 0.58, 2: 0.06, 3: 0.03, 4: 0.01, 5: 0.01, 6: 0.001, 7: 0.001, 8: 0.001, 9: 0.001, 10: 0.001}

# === Tải điểm số đã cache ===
print(f"📦 Đang tải điểm số ảnh đã cache từ {INPUT_CACHE_FILE}...")
with open(INPUT_CACHE_FILE, "r") as f:
    image_scores_cache = json.load(f)
print(f"✅ Đã tải điểm số cho {len(image_scores_cache)} truy vấn.")

# === Hàm áp dụng Strategy B và lưu kết quả ===
def apply_strategy_B_and_save_results(
    cached_scores_dict,
    top_images_per_query,
    max_images_per_article
):
    results_B = []
    cnt_less_than_top_images = 0
    # Chuẩn bị hậu tố cho tên file đầu ra
    output_suffix = f"max{max_images_per_article}"

    print(f"\n⚙️ Đang áp dụng Strategy B cho cấu hình: MaxImg/Art={max_images_per_article}")

    for qid_str, scored_images_list in cached_scores_dict.items():
        qid = qid_str # Giữ nguyên định dạng ID truy vấn

        # Lọc ảnh theo mỗi bài báo dựa trên max_images_per_article
        # và tái cấu trúc để xử lý dễ dàng
        images_by_article_rank = {} # {rank: [(iid, score, rank), ...]}
        for item in scored_images_list:
            rank = item["article_rank"]
            if rank not in images_by_article_rank:
                images_by_article_rank[rank] = []
            images_by_article_rank[rank].append((item["image_id"], item["score"], rank))

        # Danh sách này sẽ lưu trữ tất cả các ảnh tiềm năng (sau khi cắt bớt theo max_images_per_article ở mỗi bài báo)
        # Sẽ được sắp xếp theo rank bài báo, sau đó theo điểm số trong bài báo, dùng để lấp đầy nếu thiếu.
        all_potential_fill_images = []

        # Danh sách chứa các nhóm ảnh cho Strategy B
        flat_B = []

        sorted_ranks = sorted(images_by_article_rank.keys())
        for rank in sorted_ranks:
            images_in_this_article = images_by_article_rank[rank]
            # Sắp xếp theo điểm số trong bài báo và lấy top N (max_images_per_article)
            top_n_images_in_article = sorted(images_in_this_article,
                                               key=lambda x: x[1],
                                               reverse=True)

            if not top_n_images_in_article:
                continue

            # Điền vào cho Strategy B (chỉ ID ảnh)
            flat_B.extend([x[0] for x in top_n_images_in_article[:max_images_per_article]])
            
            # Điền vào cho tất cả các ảnh tiềm năng để lấp đầy (cho Strategy B khi thiếu)
            all_potential_fill_images.extend([x[0] for x in top_n_images_in_article[max_images_per_article:]])

        # --- Áp dụng Strategy B cho truy vấn hiện tại ---
        
        # Nếu flat_B không đủ ảnh, lấp đầy từ `all_potential_fill_images`
        # đã được sắp xếp theo rank bài báo, sau đó theo điểm số trong bài báo.
        if len(flat_B) < top_images_per_query:
            flat_B.extend(all_potential_fill_images[:top_images_per_query-len(flat_B)])
            # print(len(all_potential_fill_images), "images after filling for query:", qid)
        
        # Đảm bảo không quá TOP_IMAGES và lấp đầy bằng '#' nếu cần
        flat_B = flat_B[:top_images_per_query]
        flat_B += ["#"] * (top_images_per_query - len(flat_B))
        results_B.append([qid] + flat_B)

    # === Lưu kết quả ra file CSV ===
    header = ["query_id"] + [f"image_id_{i+1}" for i in range(top_images_per_query)]

    output_filename = f"{model_rerank}_{output_suffix}.csv"
    if results_B:
        pd.DataFrame(results_B, columns=header).to_csv(output_filename, index=False)
        print(f"🎉 Kết quả Strategy B đã được lưu vào {output_filename}")
        print(cnt_less_than_top_images, "truy vấn có ít hơn", top_images_per_query, "ảnh, đã lấp đầy bằng '#'.")
    else:
        print(f"⚠️ Không có kết quả nào cho Strategy B với cấu hình {output_suffix}, bỏ qua việc lưu.")


# === Vòng lặp chính để chạy các cấu hình khác nhau ===
all_configs_to_process = []

# Chỉ cấu hình cho Strategy B (không có article_rank_limit)
for max_img_art in MAX_IMAGES_PER_ARTICLE_VALUES:
    all_configs_to_process.append({
        "max_images_per_article": max_img_art,
    })

print(f"\nBắt đầu tạo kết quả cho {len(all_configs_to_process)} cấu hình duy nhất...")

for config in all_configs_to_process:
    apply_strategy_B_and_save_results(
        image_scores_cache,
        TOP_IMAGES,
        config["max_images_per_article"]
    )

print("\n✅ Tất cả các file kết quả đã được tạo thành công!")