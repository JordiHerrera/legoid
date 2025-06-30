import pandas as pd
import re
from itertools import combinations
from rapidfuzz.fuzz import token_set_ratio


def strings_match(text1, text2, threshold: float = 90.0):
    similarity = token_set_ratio(text1, text2)
    return similarity >= threshold


def find_matching_sets(piece_name, color, df, threshold=80.0, is_uneven=False):

    def similar_to_input_name(part_name):
        return strings_match(piece_name, part_name, threshold=threshold)

    required_columns = ['set_num', 'part_name', 'simplified']
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Columnes no valides, falta '{column}'")

    color_filtered_df = df[df['simplified'] == color]

    if color_filtered_df.empty:
        return pd.DataFrame(columns=df.columns)

    if is_uneven:
        final_filtered_df = color_filtered_df[color_filtered_df['part_name'].apply(similar_to_input_name)]

        return final_filtered_df


    def extract_grid_pattern(text):
        if not isinstance(text, str):
            text = str(text)
        match = re.search(r'\d+\s*x\s*\d+', text)
        return match.group() if match else None

    input_grid_pattern = extract_grid_pattern(piece_name)

    def grid_pattern_matches(row):
        row_grid_pattern = extract_grid_pattern(row)
        return row_grid_pattern == input_grid_pattern

    grid_filtered_df = color_filtered_df[color_filtered_df['part_name'].apply(grid_pattern_matches)]

    if grid_filtered_df.empty:
        return pd.DataFrame(columns=df.columns)

    final_filtered_df = grid_filtered_df[grid_filtered_df['part_name'].apply(similar_to_input_name)]

    return final_filtered_df


def generate_combinations(data, fraction=0.7):
    sublist_size = int(len(data) * fraction)
    all_combinations = combinations(data, sublist_size)
    return all_combinations