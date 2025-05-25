import torch
import torch.nn.functional as F


def confidence_adjusted_top1_gt_ratio(
    scores: list[float],
    k: int,
    gt_index: int,
    alpha: float = 0.5
) -> float:
    """
    Calculate the Confidence-Adjusted Top-1 Ground Truth Ratio at k (CAR@k).

    Args:
        scores (List[float]): List of the relevance scores for each candidate.
        k (int): The number of top candidates to consider.
        GT_index (int): The index of the ground truth candidate in scores.
        alpha (float): Weight for the entropy threshold.

    Returns:
        float: The CAR@k score.
    """

    # Sort the relevance scores.
    score_tensor = torch.tensor(scores)
    sorted_values, sorted_indices = torch.sort(score_tensor, descending=True)

    # Find the new rank of the GT candidate.
    gt_positions = (sorted_indices == gt_index).nonzero(as_tuple=True)[0]
    if gt_positions.numel() == 0:
        raise ValueError(f'Ground truth candidate with index {gt_index} was not found in the input scores.')
    gt_rank = gt_positions.item()
    
    # If the ground truth candidate is not among the top-k, return 0.0.
    if gt_rank >= k:
        return 0.0

    # Consider only the top-k candidates.
    topk_scores = sorted_values[:k]
    
    # Z-score normalization: subtract mean and divide by standard deviation.
    mean = topk_scores.mean()
    std = topk_scores.std(unbiased=False)
    if std > 0:
        normalized = (topk_scores - mean) / std
    else:
        normalized = topk_scores - mean
    
    # Convert normalized scores into a probability distribution using softmax.
    probs = F.softmax(normalized, dim=0)

    # Compute model's confidence:
    #  - Full confidence (1.0) if entropy is less than or equal to threshold.
    #  - Otherwise, confidence weakens linearly (with a floor of 0.5) as entropy increases.
    entropy = -torch.sum(probs * torch.log2(probs))
    max_entropy = torch.log2(torch.tensor(k))
    threshold = max_entropy * alpha

    if entropy <= threshold:
        confidence = 1.0
    else:
        confidence = 1.0 - (entropy - threshold) / ((max_entropy - threshold) * 2)
        confidence = max(confidence, 0.5)

    # Compute the ratio between the ground truth candidate's probability and the top candidate's probability.
    top_prob = probs[0]
    gt_prob = probs[gt_rank]

    ratio = gt_prob / top_prob if top_prob > 0 else 0.0

    # Compute the CAR@k score.
    car = confidence * ratio

    return car.item()
