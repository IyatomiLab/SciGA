import os
import pandas as pd


def load_and_prepare_split(
    split_name: str,
    dataset_json_dir: str,
    dataset_figure_dir: str = '',
    is_classification: bool = False,
) -> pd.DataFrame:
    """
    Load and prepare the dataset split for training or evaluation.

    Args:
        split_name (str): The name of the dataset split (e.g., 'train', 'valid', 'test').
        dataset_json_dir (str): Directory containing the JSON files for the dataset.
        dataset_figure_dir (str, optional): Directory containing the figure files for the dataset.
        is_classification (bool, optional): Flag indicating whether the dataset is for GA binary classification or not.

    Returns:
        pd.DataFrame: A DataFrame containing the prepared dataset split.
    """

    # Load dataset
    path = os.path.join(dataset_json_dir, f'{split_name}.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist.")

    df = pd.read_json(path, dtype={'paper_id': str})

    # Prepend figure directory path
    df['GA_path'] = dataset_figure_dir + df['GA_path']
    df['figure_paths'] = df['figure_paths'].apply(lambda paths: [dataset_figure_dir + path for path in paths])

    if not is_classification:
        return df

    # GAs
    ga_df = df[['paper_id', 'GA_path', 'GT_figure_ids', 'research_fields']].copy()
    ga_df['label'] = 1
    ga_df = ga_df.rename(columns={'GA_path': 'image_path'})

    # Non-GAs
    fig_df = df[['paper_id', 'figure_paths', 'GT_figure_ids', 'research_fields']].copy()
    fig_df = fig_df.explode('figure_paths').reset_index(drop=True)
    fig_df['label'] = 0
    fig_df = fig_df.rename(columns={'figure_paths': 'image_path'})

    # Concatenate GAs and non-GAs
    df = pd.concat([ga_df, fig_df], axis=0).reset_index(drop=True)

    return df
