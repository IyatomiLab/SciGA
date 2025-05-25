from .base import BaseGAClassifier, BaseGAClassifierLoader
from .classifier import (
    ViTAsGAClassifier,
    CLIPImageEncoderAsGAClassifier,
    EfficientNetV2AsGAClassifier,
    ConvNeXtV2AsGAClassifier,
    SwinTransformerV2AsGAClassifier,
)
import torch


# ════════════════════════════════════════════════════════════
# 📘 Intra-GA Recommendation | (ii) GA-BC
# ════════════════════════════════════════════════════════════

def load_GA_classifier(
    model_type: str,
    model_name: str = None,
    loss_weight: torch.Tensor = None,
    ckpt_path: str = None,
    **kwargs
) -> BaseGAClassifier:
    """
    Load abstract-to-figure retrieval model designated for intra-GA recommendation.

    Args:
        model_type (str): A identifier to specify the backbone architecture (e.g., 'CLIP', 'Long-CLIP', 'BLIP-2'). Used to select the corresponding model wrapper class.
        model_name (str, optional): Identifier used to load the model. Depending on the model type, this may refer to:
            - A HuggingFace repository name (e.g., 'google/vit-large-patch16-224-in21k' for ViT)
            - A local path to a pretrained checkpoint directory (e.g., `'./models/x2vlm_large_4m/'`)
            - A shorthand string used by the underlying model loader (e.g., 'ViT-L/14' for CLIP via openai/clip, 'efficientnet_v2_l' for torchvision)
        loss_weight (torch.Tensor, optional): Weighting factor for the loss function. This is used to balance the contribution of different classes during training. If not provided, a default value will be used.
        ckpt_path (str, optional): Path to a fine-tuned checkpoint (.pt) file. This is loaded *after* the model is initialized with `model_name`.
        **kwargs: Additional keyword arguments forwarded to the model loader class. Useful for passing model-specific configurations.

    Returns:
        models.BaseGAClassifier: Loaded model.
    """

    MODEL_REGISTRY = {
        model_loader.model_type: model_loader
        for model_loader in BaseGAClassifierLoader.__subclasses__()
    }

    if model_type not in MODEL_REGISTRY.keys():
        raise ValueError(f'Invalid model type \'{model_type}\'. Available models: {list(MODEL_REGISTRY.keys())}')

    model_loader = MODEL_REGISTRY[model_type](model_name=model_name, loss_weight=loss_weight, **kwargs)
    model = model_loader.get_model()

    if ckpt_path:
        state_dict = torch.load(ckpt_path, map_location='cpu')
        model.get_backbone().load_state_dict(state_dict)
        print()

    return model


class ViTAsGAClassifierLoader(BaseGAClassifierLoader):
    model_type = 'ViT'
    default_model_name = 'google/vit-large-patch16-224-in21k'

    def __init__(self, model_name: str = None, loss_weight: torch.Tensor = None, **kwargs) -> None:
        super().__init__(model_name=model_name, loss_weight=loss_weight)

    def _load(self) -> ViTAsGAClassifier:
        return ViTAsGAClassifier(
            model_name=self.model_name,
            loss_weight=self.loss_weight
        )


class CLIPImageEncoderAsGAClassifierLoader(BaseGAClassifierLoader):
    model_type = 'CLIP'
    default_model_name = 'ViT-L/14'

    def __init__(self, model_name: str = None, loss_weight: torch.Tensor = None, **kwargs) -> None:
        super().__init__(model_name=model_name, loss_weight=loss_weight)

    def _load(self) -> CLIPImageEncoderAsGAClassifier:
        return CLIPImageEncoderAsGAClassifier(
            model_name=self.model_name,
            loss_weight=self.loss_weight
        )


class EfficientNetV2AsGAClassifierLoader(BaseGAClassifierLoader):
    model_type = 'EfficientNetV2'
    default_model_name = 'efficientnet_v2_l'

    def __init__(self, model_name: str = None, loss_weight: torch.Tensor = None, **kwargs) -> None:
        super().__init__(model_name=model_name, loss_weight=loss_weight)

    def _load(self) -> EfficientNetV2AsGAClassifier:
        return EfficientNetV2AsGAClassifier(
            self.model_name,
            loss_weight=self.loss_weight
        )


class ConvNeXtV2AsGAClassifierLoader(BaseGAClassifierLoader):
    model_type = 'ConvNeXtV2'
    default_model_name = 'facebook/convnextv2-large-22k-224'

    def __init__(self, model_name: str = None, loss_weight: torch.Tensor = None, **kwargs) -> None:
        super().__init__(model_name=model_name, loss_weight=loss_weight)

    def _load(self) -> ConvNeXtV2AsGAClassifier:
        return ConvNeXtV2AsGAClassifier(
            self.model_name,
            loss_weight=self.loss_weight
        )


class SwinTransformerV2AsGAClassifierLoader(BaseGAClassifierLoader):
    model_type = 'SwinTransformerV2'
    default_model_name = 'microsoft/swin-large-patch4-window7-224-in22k'

    def __init__(self, model_name: str = None, loss_weight: torch.Tensor = None, **kwargs) -> None:
        super().__init__(model_name=model_name, loss_weight=loss_weight)

    def _load(self) -> SwinTransformerV2AsGAClassifier:
        return SwinTransformerV2AsGAClassifier(
            self.model_name,
            loss_weight=self.loss_weight
        )
