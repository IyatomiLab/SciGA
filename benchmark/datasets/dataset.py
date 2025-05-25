import re
import random
import copy
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from collections.abc import Callable
from transformers import BatchEncoding
from PIL import Image, ImageFile


# ════════════════════════════════════════════════════════════
# 📘 Intra-GA Recommendation | (iii - iv) Abs2Fig
# ════════════════════════════════════════════════════════════

class Abs2FigRetrieverDatasetForIntraGARecommendation(Dataset):
    def __init__(
        self,
        paper_id: list[str],
        research_fields: list[str],
        abstract: list[str],
        GA_path: list[str],
        GA_caption: list[str],
        figure_paths: list[list[str]],
        figure_captions: list[list[str]],
        tokenizer: Callable[[str | list[str], bool], torch.Tensor | BatchEncoding],
        transform: Callable[[Image.Image], torch.Tensor],
        GT_figure_ids: list[str],
        figure_sample_size: int = None
    ) -> None:
        self.paper_id = paper_id
        self.research_fields = research_fields
        self.abstract = abstract
        self.GA_path = GA_path
        self.GA_caption = GA_caption
        self.figure_paths = figure_paths
        self.figure_captions = figure_captions
        self.tokenizer = tokenizer
        self.transform = transform
        self.GT_figure_ids = GT_figure_ids

        self.figure_sample_size = figure_sample_size
        if self.figure_sample_size is None:
            self.figure_sample_size = max([len(figures_path) for figures_path in self.figure_paths])

        self.collate_fn = abs2fig_retriever_collate_fn_for_intraGA_recommendation

        Image.MAX_IMAGE_PIXELS = None
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def __len__(self):
        return len(self.paper_id)

    def __getitem__(self, idx) -> tuple[str, list[str], list[str], list[str], torch.Tensor | BatchEncoding, torch.Tensor, torch.Tensor | BatchEncoding, int]:
        paper_id = self.paper_id[idx]
        research_fields = self.research_fields[idx]
        abstract = self.abstract[idx]
        GA_path = self.GA_path[idx]
        GA_caption = self.GA_caption[idx]
        figure_paths = copy.deepcopy(self.figure_paths[idx])
        figure_captions = copy.deepcopy(self.figure_captions[idx])
        GT_figure_ids = self.GT_figure_ids[idx]

        # Preprocess abstract
        tokenized_abstract = self.tokenizer(abstract, batched=False)

        # Decide sampling indices
        if len(figure_paths) >= self.figure_sample_size:
            sampled_idxs = random.sample(range(len(figure_paths)), self.figure_sample_size)
        else:
            sampled_idxs = list(range(len(figure_paths)))

        padding_size = self.figure_sample_size - len(sampled_idxs)

        # Extract figure IDs
        figure_ids = ['GA'] + [
            re.match(r'.*_(F\d+)(?:\.\d+|\(\d+\))?.*', path).group(1) for path in figure_paths
        ] + ['PAD'] * padding_size

        # Preprocess figures
        paths = [GA_path] + [figure_paths[i] for i in sampled_idxs]
        figures = torch.stack([
            self.transform(Image.open(path).convert('RGB')) for path in paths
        ])
        if padding_size > 0:
            figures = torch.cat([
                figures,
                torch.zeros((padding_size, *figures.shape[1:]), device=figures.device, dtype=figures.dtype)
            ], dim=0)

        # Preprocess figure captions
        captions = [GA_caption] + [figure_captions[i] for i in sampled_idxs] + [''] * padding_size
        tokenized_captions = self.tokenizer(captions)

        # NOTE: `labels` is index of the 'GA' in the `figure_ids`
        label = 0

        return paper_id, research_fields, figure_ids, GT_figure_ids, tokenized_abstract, figures, tokenized_captions, label


def abs2fig_retriever_collate_fn_for_intraGA_recommendation(batch: list[tuple]) -> tuple:
    """
    Collate function for Abs2Fig Retriever used in Intra-GA Recommendation.
    The default PyTorch collate function `default_collate()` fails when each sample contains fields like list[str].
    This function selectively applies `default_collate()` only to tensor-compatible fields, while preserving non-tensor fields in their original structure.

    Args:
        batch (List[Tuple]): A list of dataset samples, where each sample is a tuple of multiple fields.

    Returns:
        Tuple: Batched data in the same field order as the dataset output.
    """
    (
        batch_paper_id,
        batch_research_fields,
        batch_figure_ids,
        batch_GT_figure_ids,
        batch_abstract,
        batch_figures,
        batch_captions,
        batch_label,
    ) = zip(*batch)

    batch_paper_id, batch_abstract, batch_figures, batch_captions, batch_label = default_collate([
        (paper_id, abstract, figures, captions, label)
        for paper_id, abstract, figures, captions, label
        in zip(batch_paper_id, batch_abstract, batch_figures, batch_captions, batch_label)
    ])

    return (
        batch_paper_id,
        batch_research_fields,
        batch_figure_ids,
        batch_GT_figure_ids,
        batch_abstract,
        batch_figures,
        batch_captions,
        batch_label,
    )
