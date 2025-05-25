from .base import (
    BaseAbs2FigRetrieverLoaderForIntraGARecommendation,
    BaseAbs2FigRetrieverForIntraGARecommendation,
    BaseAbs2FigRetrieverLoaderForInterGARecommendation,
    BaseAbs2FigRetrieverForInterGARecommendation,
)
from .retriever import (
    CLIPAsAbs2FigRetrieverForIntraGARecommendation,
    LongCLIPAsAbs2FigRetrieverForIntraGARecommendation,
    OpenCLIPAsAbs2FigRetrieverForIntraGARecommendation,
    BLIP2AsAbs2FigRetrieverForIntraGARecommendation,
    X2VLMAsAbs2FigRetrieverForIntraGARecommendation,
    CLIPAsAbs2FigRetrieverForInterGARecommendation,
    LongCLIPAsAbs2FigRetrieverForInterGARecommendation,
    OpenCLIPAsAbs2FigRetrieverForInterGARecommendation,
    BLIP2AsAbs2FigRetrieverForInterGARecommendation,
    X2VLMAsAbs2FigRetrieverForInterGARecommendation,
)
import torch


# ════════════════════════════════════════════════════════════
# 📘 Intra-GA Recommendation | (iii - iv) Abs2Fig
# ════════════════════════════════════════════════════════════

def load_abs2fig_retriever_for_intra_GA_recommendation(
    model_type: str,
    model_name: str = None,
    ckpt_path: str = None,
    **kwargs
) -> BaseAbs2FigRetrieverForIntraGARecommendation:
    """
    Load abstract-to-figure retrieval model designated for intra-GA recommendation.

    Args:
        model_type (str): A identifier to specify the backbone architecture (e.g., 'CLIP', 'Long-CLIP', 'BLIP-2'). Used to select the corresponding model wrapper class.
        model_name (str, optional): Identifier used to load the model. Depending on the model type, this may refer to:
            - A HuggingFace repository name (e.g., 'Salesforce/blip2-itm-vit-g' for BLIP-2)
            - A local path to a pretrained checkpoint directory (e.g., './weights/pretrained/longclip-L.pt' for Long-CLIP)
            - A shorthand string used by the underlying model loader (e.g., 'ViT-L/14' for CLIP via openai/clip)
        ckpt_path (str, optional): Path to a fine-tuned checkpoint (.pt) file. This is loaded *after* the model is initialized with `model_name`.
        **kwargs: Additional keyword arguments forwarded to the model loader class. Useful for passing model-specific configurations.

    Returns:
        models.BaseAbs2FigRetrieverForIntraGARecommendation: Loaded model.
    """

    MODEL_REGISTRY = {
        model_loader.model_type: model_loader
        for model_loader in BaseAbs2FigRetrieverLoaderForIntraGARecommendation.__subclasses__()
    }

    if model_type not in MODEL_REGISTRY.keys():
        raise ValueError(f'Invalid model type \'{model_type}\'. Available models: {list(MODEL_REGISTRY.keys())}')

    model_loader = MODEL_REGISTRY[model_type](model_name=model_name, **kwargs)
    model = model_loader.get_model()

    if ckpt_path:
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.get_backbone().load_state_dict(state_dict)
        print()

    return model


class CLIPAsAbs2FigRetrieverLoaderForIntraGARecommendation(BaseAbs2FigRetrieverLoaderForIntraGARecommendation):
    model_type = 'CLIP'
    default_model_name = 'ViT-L/14'

    def __init__(self, model_name: str = None, **kwargs) -> None:
        super().__init__(model_name=model_name)

    def _load(self) -> CLIPAsAbs2FigRetrieverForIntraGARecommendation:
        return CLIPAsAbs2FigRetrieverForIntraGARecommendation(self.model_name)


class LongCLIPAsAbs2FigRetrieverLoaderForIntraGARecommendation(BaseAbs2FigRetrieverLoaderForIntraGARecommendation):
    model_type = 'Long-CLIP'
    default_model_name = './weights/pretrained/longclip-L.pt'

    def __init__(self, model_name: str = None, **kwargs) -> None:
        super().__init__(model_name=model_name)

    def _load(self) -> LongCLIPAsAbs2FigRetrieverForIntraGARecommendation:
        return LongCLIPAsAbs2FigRetrieverForIntraGARecommendation(self.model_name)


class OpenCLIPAsAbs2FigRetrieverLoaderForIntraGARecommendation(BaseAbs2FigRetrieverLoaderForIntraGARecommendation):
    model_type = 'OpenCLIP'
    default_model_name = 'hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K'

    def __init__(self, model_name: str = None, **kwargs) -> None:
        super().__init__(model_name=model_name)

    def _load(self) -> OpenCLIPAsAbs2FigRetrieverForIntraGARecommendation:
        return OpenCLIPAsAbs2FigRetrieverForIntraGARecommendation(self.model_name)


class BLIP2AsAbs2FigRetrieverLoaderForIntraGARecommendation(BaseAbs2FigRetrieverLoaderForIntraGARecommendation):
    model_type = 'BLIP-2'
    default_model_name = 'Salesforce/blip2-itm-vit-g'

    def __init__(self, model_name: str = None, **kwargs) -> None:
        super().__init__(model_name=model_name)

    def _load(self) -> BLIP2AsAbs2FigRetrieverForIntraGARecommendation:
        return BLIP2AsAbs2FigRetrieverForIntraGARecommendation(self.model_name)


class X2VLMAsAbs2FigRetrieverLoaderForIntraGARecommendation(BaseAbs2FigRetrieverLoaderForIntraGARecommendation):
    model_type = 'X2-VLM'
    default_model_name = './weights/pretrained/x2vlm_large_4m.th'
    default_model_config_path = './model_configs/x2vlm_large_4m_for_SciGA.yaml'

    def __init__(self, model_name: str = None, **kwargs) -> None:
        model_config_path = kwargs.pop('model_config_path', None)
        self.model_config_path = model_config_path or self.default_model_config_path

        super().__init__(model_name=model_name)

    def _load(self) -> X2VLMAsAbs2FigRetrieverForIntraGARecommendation:
        return X2VLMAsAbs2FigRetrieverForIntraGARecommendation(
            model_config_path=self.model_config_path,
            model_name=self.model_name
        )


# ════════════════════════════════════════════════════════════
# 📙 Inter-GA Recommendation | (iii - iv) Abs2Fig
# ════════════════════════════════════════════════════════════

def load_abs2fig_retriever_for_inter_GA_recommendation(
    model_type: str,
    model_name: str = None,
    ckpt_path: str = None,
    **kwargs
) -> BaseAbs2FigRetrieverForInterGARecommendation:
    """
    Load abstract-to-figure retrieval model designated for Inter-GA recommendation.

    Args:
        model_type (str): A identifier to specify the backbone architecture (e.g., 'CLIP', 'Long-CLIP', 'BLIP-2'). Used to select the corresponding model wrapper class.
        model_name (str, optional): Identifier used to load the model. Depending on the model type, this may refer to:
            - A HuggingFace repository name (e.g., 'Salesforce/blip2-itm-vit-g' for BLIP-2)
            - A local path to a pretrained checkpoint directory (e.g., './weights/pretrained/longclip-L.pt' for Long-CLIP)
            - A shorthand string used by the underlying model loader (e.g., 'ViT-L/14' for CLIP via openai/clip)
        ckpt_path (str, optional): Path to a fine-tuned checkpoint (.pt) file. This is loaded *after* the model is initialized with `model_name`.
        **kwargs: Additional keyword arguments forwarded to the model loader class. Useful for passing model-specific configurations.

    Returns:
        models.BaseAbs2FigRetrieverForInterGARecommendation: Loaded model.
    """

    MODEL_REGISTRY = {
        model_loader.model_type: model_loader
        for model_loader in BaseAbs2FigRetrieverLoaderForInterGARecommendation.__subclasses__()
    }

    if model_type not in MODEL_REGISTRY.keys():
        raise ValueError(f'Invalid model type \'{model_type}\'. Available models: {list(MODEL_REGISTRY.keys())}')

    model_loader = MODEL_REGISTRY[model_type](model_name=model_name, **kwargs)
    model = model_loader.get_model()

    if ckpt_path:
        state_dict = torch.load(ckpt_path)
        model.get_backbone().load_state_dict(state_dict)

    return model


class CLIPAsAbs2FigRetrieverLoaderForInterGARecommendation(BaseAbs2FigRetrieverLoaderForInterGARecommendation):
    model_type = 'CLIP'
    default_model_name = 'ViT-L/14'

    def __init__(self, model_name: str = None, **kwargs) -> None:
        super().__init__(model_name=model_name)

    def _load(self) -> CLIPAsAbs2FigRetrieverForInterGARecommendation:
        return CLIPAsAbs2FigRetrieverForInterGARecommendation(self.model_name)


class LongCLIPAsAbs2FigRetrieverLoaderForInterGARecommendation(BaseAbs2FigRetrieverLoaderForInterGARecommendation):
    model_type = 'Long-CLIP'
    default_model_name = './weights/pretrained/longclip-L.pt'

    def __init__(self, model_name: str = None, **kwargs) -> None:
        super().__init__(model_name=model_name)

    def _load(self) -> LongCLIPAsAbs2FigRetrieverForInterGARecommendation:
        return LongCLIPAsAbs2FigRetrieverForInterGARecommendation(self.model_name)


class OpenCLIPAsAbs2FigRetrieverLoaderForInterGARecommendation(BaseAbs2FigRetrieverLoaderForInterGARecommendation):
    model_type = 'OpenCLIP'
    default_model_name = 'hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K'

    def __init__(self, model_name: str = None, **kwargs) -> None:
        super().__init__(model_name=model_name)

    def _load(self) -> OpenCLIPAsAbs2FigRetrieverForInterGARecommendation:
        return OpenCLIPAsAbs2FigRetrieverForInterGARecommendation(self.model_name)


class BLIP2AsAbs2FigRetrieverLoaderForInterGARecommendation(BaseAbs2FigRetrieverLoaderForInterGARecommendation):
    model_type = 'BLIP-2'
    default_model_name = 'Salesforce/blip2-itm-vit-g'

    def __init__(self, model_name: str = None, **kwargs) -> None:
        super().__init__(model_name=model_name)

    def _load(self) -> BLIP2AsAbs2FigRetrieverForInterGARecommendation:
        return BLIP2AsAbs2FigRetrieverForInterGARecommendation(self.model_name)


class X2VLMAsAbs2FigRetrieverLoaderForInterGARecommendation(BaseAbs2FigRetrieverLoaderForInterGARecommendation):
    model_type = 'X2-VLM'
    default_model_name = './weights/pretrained/x2vlm_large_4m.th'
    default_model_config_path = './model_configs/x2vlm_large_4m_for_SciGA.yaml'

    def __init__(self, model_name: str = None, **kwargs) -> None:
        model_config_path = kwargs.pop('model_config_path', None)
        self.model_config_path = model_config_path or self.default_model_config_path

        super().__init__(model_name=model_name)

    def _load(self) -> X2VLMAsAbs2FigRetrieverForInterGARecommendation:
        return X2VLMAsAbs2FigRetrieverForInterGARecommendation(
            model_config_path=self.model_config_path,
            model_name=self.model_name
        )
