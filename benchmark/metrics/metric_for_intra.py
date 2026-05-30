import pandas as pd
import torch
from car import confidence_adjusted_top1_gt_ratio
from typing import List


def _recall_at_k(preds: List[str], GT: List[str], k: int) -> float:
    """
    Compute Recall@k as the number of ground-truth items ranked within top-k predictions.

    Args:
        preds (List[str]): Ranked list of predicted item IDs (highest confidence first).
        GT (List[str]): List of ground-truth item IDs.
        k (int): Cutoff rank for computing Recall@k.

    Returns:
        float: Recall@k score, a value between 0 and 1.
    """
    relevant = sum(pred in GT for pred in preds[:k])
    recall = relevant / min(len(GT), k)
    return recall


def _reciprocal_rank(preds: List[str], GT: List[str]) -> float:
    """
    Compute Reciprocal Rank (RR) as 1 / rank of the first relevant prediction.

    Args:
        preds (List[str]): Ranked list of predicted item IDs.
        GT (List[str]): List of ground-truth item IDs.
    Returns:
        float: Reciprocal Rank score; 0 if no relevant item is found.
    """

    for i, pred in enumerate(preds):
        if pred in GT:
            return 1 / (i + 1)
    return 0


def _nDCG_at_k(preds: List[str], GT: List[str], k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (nDCG) at k.

    Args:
        preds (List[str]): Ranked list of predicted item IDs.
        GT (List[str]): List of ground-truth item IDs.
        k (int): Cutoff rank for computing nDCG.

    Returns:
        float: nDCG@k score, a value between 0 and 1.
    """
    DCG = 0.0
    for i, pred in enumerate(preds[:k]):
        if pred in GT:
            DCG += 1 / torch.log2(torch.tensor(i + 2, dtype=torch.float32))
    
    IDCG = sum(1 / torch.log2(torch.tensor(i + 2, dtype=torch.float32)) for i in range(min(len(GT), k)))
    
    return DCG / IDCG if IDCG > 0 else 0.0


def evaluate_intraGA_recommendation_metrics(
    result_df: pd.DataFrame,
    k_for_recall: List[int] = [1, 2, 3],
    k_for_car: List[int] = [5],
    alpha_for_car: float = 0.5
) -> tuple[dict[str, float], float, float, float]:
    """
    Evaluate retrieval performance for the Intra-GA Recommendation task.
    This function computes Recall@k (R@k), Mean Reciprocal Rank (MRR), and Confidence-Adjusted Top-1 Ground Truth Ratio (CAR@k)
    from a result DataFrame containing model predictions and ground truth labels.

    Args:
        result_df (pd.DataFrame): A DataFrame containing columns:
            - 'paper_id' (str): Paper identifier.
            - 'figure_id' (str): Figure identifier.
            - 'prob' (float): Predicted score or confidence for the figure.
            - 'GT' (str or List[str]): Ground-truth GA figure ID(s) for the paper.
        k_for_recall (List[int], optional): A list of cutoff values for computing R@k. Defaults to [1, 2, 3].
        k_for_car (List[int], optional): A list of cutoff values for computing CAR@k. Defaults to [5].
        alpha_for_car (float, optional): Threshold parameter used to scale CAR confidence. Defaults to 0.5.

    Returns:
        dict[str, float]: Dictionary mapping each k (as string) to the average R@k across all papers.
        float: Mean Reciprocal Rank (MRR) averaged across all papers.
        dict[str, float]: Dictionary mapping each k (as string) to the average CAR@k.
        dict[str, float]: Dictionary mapping each k (as string) to the fraction of papers where CAR@k > 0.5.
    """

    recall = {str(k): [] for k in k_for_recall}
    rr = []
    nDCG = []
    car_k = {str(k): [] for k in k_for_car}

    save_data = []

    for paper_id, paper_df in result_df.groupby('paper_id'):
        # Aggregate subfigure predictions and retain the one with the highest probability for each figure
        grouped = paper_df.groupby('figure_id').agg({
            'prob': 'max',
            'GT_figure_ids': 'first',
        }).reset_index()

        # Prepare ranked predictions, relevance scores, and ground truth
        sorted = grouped.sort_values(by='prob', ascending=False)
        preds = sorted['figure_id'].tolist()
        probs = sorted['prob'].tolist()
        GT = sorted['GT_figure_ids'].iloc[0]

        # Compute metrics
        for k in k_for_recall:
            recall[str(k)].append(_recall_at_k(preds, GT, k))

        rr.append(_reciprocal_rank(preds, GT))
        nDCG.append(_nDCG_at_k(preds, GT, k=5))

        for k in k_for_car:
            car_k[str(k)].append(confidence_adjusted_top1_gt_ratio(probs, k, preds.index('GA'), alpha_for_car))

            if car_k[str(k)][-1] > 0.9:
                print(f"High CAR@{k} for paper {paper_id}: {car_k[str(k)][-1]}")
        
        # PaperID, Top-K の preds, CAR@5 をjsonで保存する
        save_data.append({
            'paper_id': paper_id,
            'preds': preds,
            'probs': probs,
            'GT': GT,
            'CAR@5': car_k['5'][-1],
        })



    # Calculate mean metrics across all papers
    mean_recall = {
        k: torch.tensor(score).mean().item()
        for k, score in recall.items()
    }
    mean_rr = torch.tensor(rr).mean().item()
    mean_nDCG = torch.tensor(nDCG).mean().item()
    mean_car = {
        k: torch.tensor(score).mean().item()
        for k, score in car_k.items()
    }
    car_above_05 = {
        k: torch.tensor([1.0 if score > 0.5 else 0.0 for score in scores]).mean().item()
        for k, scores in car_k.items()
    }

    # Save detailed results to a CSV file
    save_df = pd.DataFrame(save_data)
    save_df.to_csv('intraGA_recommendation_detailed_results.csv', index=False)


    return mean_recall, mean_rr, mean_nDCG, mean_car, car_above_05
