from .base import BaseAbs2CapMatcher, BaseAbs2CapMatcherLoader
from .matcher import (
    Abs2CapMatcherWithROUGE,
    Abs2CapMatcherWithMETEOR,
    Abs2CapMatcherWithCIDEr,
    Abs2CapMatcherWithBM25,
    Abs2CapMatcherWithBERTScore,
)


# ════════════════════════════════════════════════════════════
# 📘 Intra-GA / 📙 Inter-GA Recommendation | (i) Abs2Cap
# ════════════════════════════════════════════════════════════

def load_abs2cap_matcher(
    model_type: str,
    **kwargs
) -> BaseAbs2CapMatcher:
    """
    Load abstract-to-figure retrieval model designated for intra-GA recommendation.

    Args:
        model_type (str): A identifier to specify the backbone metric (e.g., 'ROUGE', 'BM25', 'BERTScore'). Used to select the corresponding model wrapper class.
        **kwargs: Additional keyword arguments forwarded to the model loader class. Useful for passing model-specific configurations.

    Returns:
        models.BaseAbs2CapMatcher: Loaded model.
    """

    MODEL_REGISTRY = {
        model_loader.model_type: model_loader
        for model_loader in BaseAbs2CapMatcherLoader.__subclasses__()
    }

    if model_type not in MODEL_REGISTRY.keys():
        raise ValueError(f'Invalid model type \'{model_type}\'. Available models: {list(MODEL_REGISTRY.keys())}')

    model_loader = MODEL_REGISTRY[model_type](**kwargs)
    model = model_loader.get_model()

    return model


class Abs2CapMatcherLoaderWithROUGE(BaseAbs2CapMatcherLoader):
    model_type = 'ROUGE'
    default_model_name = 'rougeL'

    def __init__(self, model_name: str = None, **kwargs):
        model_name = kwargs.get('model_name', None)
        self.model_name = model_name or self.default_model_name

        super().__init__()

    def _load(self) -> BaseAbs2CapMatcher:
        return Abs2CapMatcherWithROUGE(model_name=self.model_name)


class Abs2CapMatcherLoaderWithMETEOR(BaseAbs2CapMatcherLoader):
    model_type = 'METEOR'

    def __init__(self, **kwargs):
        super().__init__()

    def _load(self) -> BaseAbs2CapMatcher:
        return Abs2CapMatcherWithMETEOR()


class Abs2CapMatcherLoaderWithCIDEr(BaseAbs2CapMatcherLoader):
    model_type = 'CIDEr'

    def __init__(self, **kwargs):
        super().__init__()

    def _load(self) -> BaseAbs2CapMatcher:
        return Abs2CapMatcherWithCIDEr()


class Abs2CapMatcherLoaderWithBM25(BaseAbs2CapMatcherLoader):
    model_type = 'BM25'
    default_stem_language = 'english'
    default_stopwords_language = 'en'

    def __init__(self, stem_language: str = None, stopwords_language: str = None, **kwargs):
        stem_language = kwargs.get('stem_language', None)
        self.stem_language = stem_language or self.default_stem_language
        stopwords_language = kwargs.get('stopwords_language', None)
        self.stopwords_language = stopwords_language or self.default_stopwords_language

        super().__init__()

    def _load(self) -> BaseAbs2CapMatcher:
        return Abs2CapMatcherWithBM25(
            stem_language=self.stem_language,
            stopwords_language=self.stopwords_language,
        )


class Abs2CapMatcherLoaderWithBERTScore(BaseAbs2CapMatcherLoader):
    model_type = 'BERTScore'
    default_model_name = 'bert-base-uncased'
    default_num_layers = 12
    default_batch_size = 32

    def __init__(self, model_name: str = None, num_layers: int = None, batch_size: int = None, **kwargs):
        model_name = kwargs.get('model_name', None)
        self.model_name = model_name or self.default_model_name
        num_layers = kwargs.get('num_layers', None)
        self.num_layers = num_layers or self.default_num_layers
        batch_size = kwargs.get('batch_size', None)
        self.batch_size = batch_size or self.default_batch_size

        super().__init__()

    def _load(self) -> BaseAbs2CapMatcher:
        return Abs2CapMatcherWithBERTScore(
            model_name=self.model_name,
            num_layers=self.num_layers,
            batch_size=self.batch_size,
        )
