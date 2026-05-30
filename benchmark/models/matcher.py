from .base import BaseAbs2CapMatcher, Abs2CapMatcherOutput
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from pycocoevalcap.cider.cider import Cider
import bm25s
import Stemmer
import evaluate
from bert_score import score as bert_score
from transformers import AutoTokenizer

# ════════════════════════════════════════════════════════════
# 📘 Intra-GA / 📙 Inter-GA Recommendation | (i) Abs2Cap
# ════════════════════════════════════════════════════════════

class Abs2CapMatcherWithROUGE(BaseAbs2CapMatcher):
    """
    ROUGE () for Abs2Cap Matching
    """

    def __init__(
        self,
        model_name: str,
    ):
        super().__init__()
        self.model_name = model_name
        self.scorer = rouge_scorer.RougeScorer([self.model_name], use_stemmer=True)

    def match(self, abstract: str, captions: list[str]) -> Abs2CapMatcherOutput:
        sim_abs2cap = [
            self.scorer.score(abstract, caption)[self.model_name].fmeasure
            for caption in captions
        ]
        return Abs2CapMatcherOutput(sim_abs2cap=sim_abs2cap)


class Abs2CapMatcherWithMETEOR(BaseAbs2CapMatcher):
    """
    METEOR () for Abs2Cap Matching
    """

    def __init__(self):
        super().__init__()

    def match(self, abstract: str, captions: list[str]) -> Abs2CapMatcherOutput:
        sim_abs2cap = [
            meteor_score([word_tokenize(abstract)], word_tokenize(caption))
            for caption in captions
        ]
        return Abs2CapMatcherOutput(sim_abs2cap=sim_abs2cap)


class Abs2CapMatcherWithCIDEr(BaseAbs2CapMatcher):
    """
    CIDEr () for Abs2Cap Matching
    """

    def __init__(self):
        super().__init__()

    def match(self, abstract: str, captions: list[str]) -> Abs2CapMatcherOutput:
        # NOTE: CIDEr requires a non-empty corpus to compute scores
        if all(caption == '' for caption in captions):
            sim_abs2cap = [0.0] * len(captions)
            return Abs2CapMatcherOutput(sim_abs2cap=sim_abs2cap)

        cider = Cider()
        candidates = {i: [caption.lower()] for i, caption in enumerate(captions)}
        references = {i: [abstract.lower()] for i in range(len(captions))}
        _, cider_scores = cider.compute_score(candidates, references)
        sim_abs2cap = cider_scores.tolist()

        return Abs2CapMatcherOutput(sim_abs2cap=sim_abs2cap)


class Abs2CapMatcherWithBM25(BaseAbs2CapMatcher):
    """
    BM25 () for Abs2Cap Matching
    """

    def __init__(
        self,
        stem_language: str,
        stopwords_language: str,
    ):
        super().__init__()
        self.stem_language = stem_language
        self.stemmer = Stemmer.Stemmer(self.stem_language)
        self.stopwords_language = stopwords_language

    def match(self, abstract: str, captions: list[str]) -> Abs2CapMatcherOutput:
        # Create corpus
        corpus_tokens = bm25s.tokenize(captions, stopwords=self.stopwords_language, stemmer=self.stemmer)

        # NOTE: BM25 requires a non-empty corpus to compute scores
        if len(corpus_tokens.vocab) == 0:
            sim_abs2cap = [0.0] * len(captions)
            return Abs2CapMatcherOutput(sim_abs2cap=sim_abs2cap)

        # Compute BM25 scores
        bm25 = bm25s.BM25()
        bm25.index(corpus_tokens)
        tokenized_abstract = bm25s.tokenize(abstract, stemmer=self.stemmer)
        sorted_idxs, sorted_sim_abs2cap = bm25.retrieve(tokenized_abstract, k=len(captions))

        sorted_idxs = sorted_idxs[0].tolist()
        sorted_sim_abs2cap = sorted_sim_abs2cap[0].tolist()

        # NOTE: Restore the sorted indices to the original order
        sim_abs2cap = [0.0] * len(captions)
        for idx, sorted_idx in enumerate(sorted_idxs):
            sim_abs2cap[sorted_idx] = sorted_sim_abs2cap[idx]

        return Abs2CapMatcherOutput(sim_abs2cap=sim_abs2cap)


class Abs2CapMatcherWithBERTScore(BaseAbs2CapMatcher):
    """
    BERTScore () for Abs2Cap Matching
    """

    def __init__(self, language: str, device: str):
        super().__init__()
        self.language = language
        self.device = device
        self.model_type = "allenai/scibert_scivocab_uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        self.max_length = 512

    def _truncate(self, decoded: str) -> str:
        encoded = self.tokenizer(
            decoded,
            truncation=True,
            max_length=self.max_length-2,
            add_special_tokens=False,
            return_tensors=None,
        )
        decoded = self.tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)
        return decoded

    def match(self, abstract: str, captions: list[str]) -> Abs2CapMatcherOutput:
        abstract = self._truncate(abstract)
        captions = [self._truncate(c) for c in captions]

        P, R, F1 = bert_score(
            captions,
            [abstract] * len(captions),
            lang="en-sci",
            device=self.device,
            batch_size=2048,
        )
        sim_abs2cap = F1.tolist()

        return Abs2CapMatcherOutput(sim_abs2cap=sim_abs2cap)
