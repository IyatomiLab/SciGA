import torch
from torch import nn
from .base import BaseGAClassifier
from transformers import AutoImageProcessor, ViTForImageClassification, ConvNextV2ForImageClassification, SwinForImageClassification
import clip
from torchvision import models
from PIL import Image, ImageFile


# ════════════════════════════════════════════════════════════
# 📘 Intra-GA Recommendation | (ii) GA-BC
# ════════════════════════════════════════════════════════════

class ViTAsGAClassifier(BaseGAClassifier):
    """
    ViT (https://doi.org/10.48550/arXiv.2010.11929) for Intra-GA Recommendation
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 2,
        loss_weight: torch.Tensor = None,
    ):
        """
        Loads ViT (https://huggingface.co/docs/transformers/main/en/model_doc/vit).
        """

        super().__init__(loss_weight=loss_weight)
        self.model = ViTForImageClassification.from_pretrained(model_name)
        self.model.config.num_labels = num_classes
        self.transform = AutoImageProcessor.from_pretrained(model_name)

    def get_backbone(self) -> nn.Module:
        return self.model

    def preprocess_image(self, image: Image.Image | ImageFile.ImageFile) -> torch.Tensor:
        preprocessed_image = self.transform(image)
        return preprocessed_image

    def _classify_image(self, image: torch.Tensor) -> torch.Tensor:
        logits = self.model(image).logits
        return logits


class CLIPImageEncoderAsGAClassifier(BaseGAClassifier):
    """
    CLIP image encoder (https://doi.org/10.48550/arXiv.2103.00020) for Intra-GA Recommendation
    """

    def __init__(
        self,
        model_name: str,
        hidden_dim: int = 512,
        num_classes: int = 2,
        loss_weight: torch.Tensor = None,
    ):
        """
        Loads CLIP (https://github.com/openai/CLIP) and MLP head for image classification.
        """

        super().__init__(loss_weight=loss_weight)
        model, self.transform = clip.load(model_name, device='cpu')
        self.image_encoder = model.visual
        self.mlp = nn.Sequential(
            nn.Linear(self.image_encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def get_backbone(self):
        class _Backbone(nn.Module):
            def __init__(self, encoder, mlp):
                super().__init__()
                self.image_encoder = encoder
                self.mlp = mlp
        return _Backbone(self.image_encoder, self.mlp)

    def preprocess_image(self, image: Image.Image | ImageFile.ImageFile) -> torch.Tensor:
        preprocessed_image = self.transform(image)
        return preprocessed_image

    def _classify_image(self, image: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(image)
        logits = self.mlp(image_features)
        return logits


class EfficientNetV2AsGAClassifier(BaseGAClassifier):
    """
    EfficientNetV2 (https://doi.org/10.48550/arXiv.2104.00298) for Intra-GA Recommendation
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 2,
        loss_weight: torch.Tensor = None,
    ):
        """
        Loads EfficientNetV2 (https://pytorch.org/vision/stable/models/efficientnetv2.html).
        """

        super().__init__(loss_weight=loss_weight)
        weight_dict = {
            'efficientnet_v2_s': models.EfficientNet_V2_S_Weights.DEFAULT,
            'efficientnet_v2_m': models.EfficientNet_V2_M_Weights.DEFAULT,
            'efficientnet_v2_l': models.EfficientNet_V2_L_Weights.DEFAULT,
        }
        weights = weight_dict[model_name]
        model_fn = getattr(models, model_name)
        self.model: models.EfficientNet = model_fn(weights=weights)
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(num_features, num_classes)
        self.transform = weights.transforms()

    def get_backbone(self) -> nn.Module:
        return self.model

    def preprocess_image(self, image: Image.Image | ImageFile.ImageFile) -> torch.Tensor:
        preprocessed_image = self.transform(image)
        return preprocessed_image

    def _classify_image(self, image: torch.Tensor) -> torch.Tensor:
        logits = self.model(image).logits
        return logits


class ConvNeXtV2AsGAClassifier(BaseGAClassifier):
    """
    ConvNeXtV2 (https://doi.org/10.48550/arXiv.2301.00808) for Intra-GA Recommendation
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 2,
        loss_weight: torch.Tensor = None,
    ):
        """
        Loads ConvNeXtV2 (https://huggingface.co/docs/transformers/main/en/model_doc/convnextv2).
        """

        super().__init__(loss_weight=loss_weight)
        self.model = ConvNextV2ForImageClassification.from_pretrained(model_name, num_labels=num_classes, ignore_mismatched_sizes=True)
        self.transform = AutoImageProcessor.from_pretrained(model_name)

    def get_backbone(self) -> nn.Module:
        return self.model

    def preprocess_image(self, image: Image.Image | ImageFile.ImageFile) -> torch.Tensor:
        preprocessed_image = self.transform(image)
        return preprocessed_image

    def _classify_image(self, image: torch.Tensor) -> torch.Tensor:
        logits = self.model(image).logits
        return logits


class SwinTransformerV2AsGAClassifier(BaseGAClassifier):
    """
    SwinTransformerV2 (https://doi.org/10.48550/arXiv.2111.09883) for Intra-GA Recommendation
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = 2,
        loss_weight: torch.Tensor = None,
    ):
        """
        Loads SwinTransformerV2 (https://huggingface.co/docs/transformers/main/en/model_doc/swin).
        """

        super().__init__(loss_weight=loss_weight)
        self.model = SwinForImageClassification.from_pretrained(model_name, num_labels=num_classes, ignore_mismatched_sizes=True)
        self.transform = AutoImageProcessor.from_pretrained(model_name)

    def get_backbone(self) -> nn.Module:
        return self.model

    def preprocess_image(self, image: Image.Image | ImageFile.ImageFile) -> torch.Tensor:
        preprocessed_image = self.transform(image)
        return preprocessed_image

    def _classify_image(self, image: torch.Tensor) -> torch.Tensor:
        logits = self.model(image).logits
        return logits
