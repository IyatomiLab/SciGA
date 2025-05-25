import os
import pandas as pd


def load_and_prepare_split(
    split_name: str,
    dataset_json_dir: str,
    dataset_figure_dir: str = '',
) -> pd.DataFrame:
    """
    Load and prepare the dataset split for training or evaluation.

    Args:
        split_name (str): The name of the dataset split (e.g., 'train', 'valid', 'test').
        dataset_json_dir (str): Directory containing the JSON files for the dataset.
        dataset_figure_dir (str, optional): Directory containing the figure files for the dataset.

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

    return df
