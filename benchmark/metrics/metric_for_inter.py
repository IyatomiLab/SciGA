import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import clip
from dreamsim import dreamsim
from tqdm import tqdm
from typing import List, Dict
from PIL import Image


def _field_precision_at_k(
    preds: List[List[str]],
    GT: List[str],
    k: int,
) -> float:
    """
    Compute Field-Precision@k as the proportion of top-k predictions that share 
    at least one common field with any of the ground-truth fields.

    Args:
        preds (List[List[str]]): Ranked list of predicted fields (each item is a list of fields).
        GT (List[str]): List of ground-truth fields.
        k (int): Cutoff rank for computing Field-Precision@k.

    Returns:
        float: Field-Precision@k score, a value between 0 and 1.
    """

    if k == 0 or not preds:
        return 0.0

    hits = sum(1 for fields in preds[:k] if set(GT) & set(fields))
    precision = hits / k
    return precision


@torch.inference_mode()
def save_SBERT_embeddings(
    paper_ids: List[str],
    abstracts: List[str],
    cache_path: str,
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    batch_size: int = 128,
) -> Dict[str, torch.Tensor]:
    """
    Compute SBERT (https://doi.org/10.48550/arXiv.1908.10084) embeddings for a list of abstracts and save them as a cache.

    Args:
        paper_ids (List[str]): List of paper IDs corresponding to each abstract.
        abstracts (List[str]): List of abstract texts.
        cache_path (str): Path to save the embedding cache (as a .pt file).
        model_name (str): HuggingFace model name for SBERT. Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.
        device (torch.device): Device to run the model on.
        batch_size (int): Batch size for encoding.

    Returns:
        Dict[str, torch.Tensor]: Dictionary mapping paper_id to its SBERT embedding.
    """

    if len(paper_ids) != len(abstracts):
        raise ValueError('The lengths of paper_ids and abstracts must be the same.')

    # Load the SBERT model (https://www.sbert.net/)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    embeddings_dict = {}

    for i in tqdm(range(0, len(paper_ids), batch_size), ncols=80):
        batch_paper_ids = paper_ids[i:i + batch_size]
        batch_abstracts = abstracts[i:i + batch_size]

        # Tokenize abstracts
        tokenized_abstracts = tokenizer(
            batch_abstracts,
            padding=True,
            truncation=True,
            return_tensors='pt',
        ).to(device)

        # Encode abstracts
        with torch.cuda.amp.autocast(device.type):
            output = model(**tokenized_abstracts)
            last_hidden_state = output.last_hidden_state
            attention_mask = tokenized_abstracts['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
            pooled = torch.sum(last_hidden_state * attention_mask, dim=1) / torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            normalized_abstract_embeddings = F.normalize(pooled, p=2, dim=1)
            normalized_abstract_embeddings.cpu()

        # Record embeddings
        for paper_id, embedding in zip(batch_paper_ids, normalized_abstract_embeddings):
            embeddings_dict[paper_id] = embedding

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(embeddings_dict, cache_path)
    return embeddings_dict


@torch.inference_mode()
def save_CLIP_embeddings(
    paper_ids: List[str],
    GA_paths: List[str],
    cache_path: str,
    model_name: str = 'ViT-L/14',
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    batch_size: int = 128,
) -> Dict[str, torch.Tensor]:
    """
    Compute CLIP (https://doi.org/10.48550/arXiv.2103.00020) image embeddings for a list of GAs and save them as a cache.

    Args:
        paper_ids (List[str]): List of paper IDs corresponding to each GA.
        GA_paths (List[str]): List of file paths to GAs.
        cache_path (str): Path to save the embedding cache (as a .pt file).
        model_name (str): CLIP model name. Defaults to 'ViT-L/14'.
        device (torch.device): Device to run the model on.
        batch_size (int): Batch size for encoding.

    Returns:
        Dict[str, torch.Tensor]: Dictionary mapping paper_id to its CLIP image embedding.
    """

    if len(paper_ids) != len(GA_paths):
        raise ValueError('The lengths of paper_ids and GA_paths must be the same.')

    # Load the CLIP model (https://github.com/openai/CLIP)
    model, preprocess = clip.load(model_name, device='cpu')
    model.to(device)

    embeddings_dict = {}

    for i in tqdm(range(0, len(GA_paths), batch_size), ncols=80):
        batch_paper_ids = paper_ids[i:i + batch_size]
        batch_GA_paths = GA_paths[i:i + batch_size]

        # Preprocess GAs
        GAs = [Image.open(path).convert('RGB') for path in batch_GA_paths]
        preprocessed_GA = torch.stack([preprocess(GA) for GA in GAs]).to(device)

        # Encode GAs
        with torch.cuda.amp.autocast(device.type):
            GA_embeddings = model.encode_image(preprocessed_GA)
            normalized_GA_embeddings = GA_embeddings / GA_embeddings.norm(dim=-1, keepdim=True)
            normalized_GA_embeddings.cpu()

        # Record embeddings
        for paper_id, embedding in zip(batch_paper_ids, normalized_GA_embeddings):
            embeddings_dict[paper_id] = embedding

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(embeddings_dict, cache_path)
    return embeddings_dict


def save_DreamSim_embeddings(
    paper_ids: List[str],
    GA_paths: List[str],
    cache_path: str,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    batch_size: int = 128,
) -> Dict[str, torch.Tensor]:
    """
    Compute DreamSim (https://doi.org/10.48550/arXiv.2306.09344) embeddings for a list of GAs and save them as a cache.

    Args:
        paper_ids (List[str]): List of paper IDs corresponding to each GA.
        GA_paths (List[str]): List of file paths to GAs.
        cache_path (str): Path to save the embedding cache (as a .pt file).
        device (torch.device): Device to run the model on.
        batch_size (int): Batch size for encoding.

    Returns:
        Dict[str, torch.Tensor]: Dictionary mapping paper_id to its DreamSim embedding.
    """

    if len(paper_ids) != len(GA_paths):
        raise ValueError('The lengths of paper_ids and GA_paths must be the same.')

    # Load the DreamSim model (https://pypi.org/project/dreamsim/)
    model, preprocess = dreamsim(pretrained=True, device=device)

    embeddings_dict = {}

    for i in tqdm(range(0, len(GA_paths), batch_size), ncols=80):
        batch_paper_ids = paper_ids[i:i + batch_size]
        batch_GA_paths = GA_paths[i:i + batch_size]

        # Preprocess GAs
        GAs = [Image.open(path).convert('RGB') for path in batch_GA_paths]
        preprocessed_GA = torch.cat([preprocess(GA) for GA in GAs]).to(device)

        # Encode GAs
        with torch.cuda.amp.autocast(device.type):
            GA_embeddings = model.embed(preprocessed_GA)
            normalized_GA_embeddings = GA_embeddings / GA_embeddings.norm(dim=-1, keepdim=True)
            normalized_GA_embeddings.cpu()

        # Record embeddings
        for paper_id, embedding in zip(batch_paper_ids, normalized_GA_embeddings):
            embeddings_dict[paper_id] = embedding

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(embeddings_dict, cache_path)
    return embeddings_dict


def evaluate_interGA_recommendation_metrics(
    result_df: pd.DataFrame,
    SBERT_embeddings: Dict[str, torch.Tensor],
    CLIP_embeddings: Dict[str, torch.Tensor],
    DreamSim_embeddings: Dict[str, torch.Tensor],
    k_for_field_precision: List[int] = [5, 10],
    k_for_abs2abs_SBERT: List[int] = [5, 10],
    k_for_GA2GA_CLIPScore: List[int] = [5, 10],
    k_for_GA2GA_DreamSim: List[int] = [5, 10],
):
    """
    Evaluate retrieval performance for the Inter-GA Recommendation task.
    This function computes Field-Precision@k (Field-P@k), abstract-to-abstract SBERT similarity@k (Abs2Abs SBERT@k), and GA-to-GA CLIPScore@k (GA2GA CLIP-S@k)
    containing model predictions and ground truth labels.

    Args:
        result_df (pd.DataFrame): A DataFrame containing columns:
            - 'paper_id' (str): Paper identifier.
            - 'research_field' (str): Research field identifier.
            - 'retrieved_paper_id' (str): Retrieved paper identifier.
            - 'retrieved_research_fields' (str): Retrieved research field identifier.
            - 'prob' (float): Probability score for the retrieved paper.
        k_for_field_precision (List[int], optional): A list of cutoff values for computing Field-Precision@k. Defaults to [1, 5, 10].
        k_for_abs2abs_SBERT (List[int], optional): A list of cutoff values for computing Abs2Abs similarity. Defaults to [1, 5, 10].
        k_for_GA2GA_CLIPScore (List[int], optional): A list of cutoff values for computing GA2GA similarity. Defaults to [1, 5, 10].

    Returns:
        dict[str, float]: Dictionary mapping each k (as string) to the average Field-Precision@k.
        dict[str, float]: Dictionary mapping each k (as string) to the average Abs2Abs SBERT@k.
        dict[str, float]: Dictionary mapping each k (as string) to the average GA2GA CLIP-S@k.
    """

    field_precision_results = {str(k): [] for k in k_for_field_precision}
    abs2abs_SBERT_results = {
        'mean': {str(k): [] for k in k_for_abs2abs_SBERT},
        'std': {str(k): [] for k in k_for_abs2abs_SBERT}
    }
    GA2GA_CLIPScore_results = {
        'mean': {str(k): [] for k in k_for_GA2GA_CLIPScore},
        'std': {str(k): [] for k in k_for_GA2GA_CLIPScore}
    }
    GA2GA_DreamSim_results = {
        'mean': {str(k): [] for k in k_for_GA2GA_DreamSim},
        'std': {str(k): [] for k in k_for_GA2GA_DreamSim}
    }

    for paper_id, paper_df in result_df.groupby('paper_id'):
        # Prepare reference and retrieved data
        sorted = paper_df.sort_values(by='prob', ascending=False)
        retrieved_paper_ids = sorted['retrieved_paper_id'].tolist()
        reference_research_fields = sorted['research_fields'].iloc[0]
        retrieved_research_fields = sorted['retrieved_research_fields'].tolist()

        # Compute metrics
        for k in k_for_field_precision:
            field_precision_results[str(k)].append(_field_precision_at_k(retrieved_research_fields, reference_research_fields, k))

        max_k = max(k_for_abs2abs_SBERT)
        retrieved_paper_ids = retrieved_paper_ids[:max_k]
        query_abstract_embeddings = SBERT_embeddings[paper_id]
        retrieved_abstract_embeddings = torch.stack([SBERT_embeddings[retrieved_paper_id] for retrieved_paper_id in retrieved_paper_ids])
        abs2abs_SBERT_similarities = torch.matmul(query_abstract_embeddings, retrieved_abstract_embeddings.T)
        for k in k_for_abs2abs_SBERT:
            top_k = abs2abs_SBERT_similarities[:k]
            abs2abs_SBERT_results['mean'][str(k)].append(top_k.mean().item())
            abs2abs_SBERT_results['std'][str(k)].append(top_k.std().item())

        max_k = max(k_for_GA2GA_CLIPScore)
        retrieved_paper_ids = retrieved_paper_ids[:max_k]
        query_GA_embeddings = CLIP_embeddings[paper_id]
        retrieved_GA_embeddings = torch.stack([CLIP_embeddings[retrieved_paper_id] for retrieved_paper_id in retrieved_paper_ids])
        GA2GA_CLIPScore_similarities = torch.matmul(query_GA_embeddings, retrieved_GA_embeddings.T)
        for k in k_for_GA2GA_CLIPScore:
            top_k = GA2GA_CLIPScore_similarities[:k]
            GA2GA_CLIPScore_results['mean'][str(k)].append(top_k.mean().item())
            GA2GA_CLIPScore_results['std'][str(k)].append(top_k.std().item())

        max_k = max(k_for_GA2GA_DreamSim)
        retrieved_paper_ids = retrieved_paper_ids[:max_k]
        query_GA_embeddings = DreamSim_embeddings[paper_id]
        retrieved_GA_embeddings = torch.stack([DreamSim_embeddings[retrieved_paper_id] for retrieved_paper_id in retrieved_paper_ids])
        GA2GA_DreamSim_similarities = torch.matmul(query_GA_embeddings, retrieved_GA_embeddings.T)
        for k in k_for_GA2GA_DreamSim:
            top_k = GA2GA_DreamSim_similarities[:k]
            GA2GA_DreamSim_results['mean'][str(k)].append(top_k.mean().item())
            GA2GA_DreamSim_results['std'][str(k)].append(top_k.std().item())

    # Calulate mean and std for each metric across all papers
    mean_field_precision = {
        k: torch.tensor(score).mean().item()
        for k, score in field_precision_results.items()
    }
    mean_abs2abs_SBERT = {
        k: torch.tensor(score).mean().item()
        for k, score in abs2abs_SBERT_results['mean'].items()
    }
    std_abs2abs_SBERT = {
        k: torch.tensor(score).mean().item()
        for k, score in abs2abs_SBERT_results['std'].items()
    }
    mean_GA2GA_CLIPScore = {
        k: torch.tensor(score).mean().item()
        for k, score in GA2GA_CLIPScore_results['mean'].items()
    }
    std_GA2GA_CLIPScore = {
        k: torch.tensor(score).mean().item()
        for k, score in GA2GA_CLIPScore_results['std'].items()
    }
    mean_GA2GA_DreamSim = {
        k: torch.tensor(score).mean().item()
        for k, score in GA2GA_DreamSim_results['mean'].items()
    }
    std_GA2GA_DreamSim = {
        k: torch.tensor(score).mean().item()
        for k, score in GA2GA_DreamSim_results['std'].items()
    }

    return mean_field_precision, mean_abs2abs_SBERT, std_abs2abs_SBERT, mean_GA2GA_CLIPScore, std_GA2GA_CLIPScore, mean_GA2GA_DreamSim, std_GA2GA_DreamSim
