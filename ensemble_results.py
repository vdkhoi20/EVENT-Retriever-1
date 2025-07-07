import pandas as pd

def reciprocal_rank_fusion(rank_lists, k=60):
    """
    Performs Reciprocal Rank Fusion (RRF) on a list of ranked lists.

    Args:
        rank_lists (list of lists): A list where each inner list contains
                                    image_ids ordered by their rank from one model.
                                    Example: [['imgA', 'imgB'], ['imgC', 'imgA']]
        k (int): A constant to soften the impact of high ranks. Common values are 60.

    Returns:
        dict: A dictionary where keys are image_ids and values are their fused RRF scores.
    """
    fused_scores = {}
    for ranks in rank_lists:
        # We only consider actual image_ids for scoring, ignoring placeholders
        for rank, image_id in enumerate(ranks):
            if image_id == "#":  # Skip placeholder image IDs
                continue
            
            # Calculate score for the current image based on its 1-indexed rank
            score = 1.0 / (k + rank + 1)
            
            # Add score to the image's total fused score
            if image_id not in fused_scores:
                fused_scores[image_id] = 0.0
            fused_scores[image_id] += score
    return fused_scores

def create_ensemble_submission(csv_files, k=60):
    """
    Reads multiple CSV submission files (query_id, image_id_1...10),
    applies RRF, and generates a new ensemble submission CSV.

    Args:
        csv_files (list): A list of file paths to the individual model submission CSVs.
        k (int): The 'k' parameter for Reciprocal Rank Fusion.

    Returns:
        pandas.DataFrame: A DataFrame representing the final ensemble submission.
    """
    all_query_rank_lists = {}

    print("🚀 Reading individual submission files and collecting ranks...")
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            query_id = row['query_id']
            
            # Initialize if this is a new query_id
            if query_id not in all_query_rank_lists:
                all_query_rank_lists[query_id] = {'rank_lists': []}
            
            # Extract the 10 image_ids for the current query
            # Ensure image_ids are treated as strings
            current_query_ranks = [str(row[f'image_id_{rank}']) for rank in range(1, 11)]
            all_query_rank_lists[query_id]['rank_lists'].append(current_query_ranks)
    
    print("✨ Performing Reciprocal Rank Fusion...")
    final_submission_data = []
    # Process queries in a sorted order (optional, but good for consistency)
    for qid, data in all_query_rank_lists.items():
        fused_scores = reciprocal_rank_fusion(data['rank_lists'], k=k)
        
        # Sort image_ids by their fused scores in descending order
        # If no images have fused scores (e.g., all were '#'), sorted_images will be empty
        sorted_images = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        
        # Get the top 10 image_ids for the final submission
        top_10_images = sorted_images[:10]
        
        # Prepare the row for the new DataFrame
        row_data = {'query_id': qid}
        
        # Fill in the top 10 image_ids. If less than 10 are found, fill remaining with '#'
        for i in range(10):
            if i < len(top_10_images):
                # Use the actual image_id if available
                row_data[f'image_id_{i+1}'] = top_10_images[i][0]
            else:
                # Fill with '#' if not enough valid images
                row_data[f'image_id_{i+1}'] = "#" 
        final_submission_data.append(row_data)

    return pd.DataFrame(final_submission_data)

# --- Configuration and Execution ---
if __name__ == "__main__":
    # List of your individual submission CSV files
    # Make sure these file names exactly match your output files
    csv_files_to_ensemble = [
        "8B_8B_max3.csv",
        "4B_4B_max3.csv",
        # "submission_ensemble_final.csv",
    ]

    # You can experiment with different 'k' values (e.g., 10, 30, 100)
    # The default of 60 is a common starting point.
    rrf_k_value = 60 

    # Generate the ensemble submission
    final_ensemble_df = create_ensemble_submission(csv_files_to_ensemble, k=rrf_k_value)

    # Save the ensemble results to a new CSV file
    output_filename = "submission_ensemble_final_final.csv"
    final_ensemble_df.to_csv(output_filename, index=False)

    print(f"\n✅ RRF Ensemble submission successfully created and saved to {output_filename}")
    print(f"Total queries processed: {len(final_ensemble_df)}")
    print(f"Example of the first few rows of the ensemble submission:\n{final_ensemble_df.head()}")