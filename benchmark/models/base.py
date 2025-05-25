import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from abc import ABC, abstractmethod
from transformers import BatchEncoding
from PIL import Image, ImageFile


# ════════════════════════════════════════════════════════════
# 📘 Intra-GA / 📙 Inter-GA Recommendation | (i) Abs2Cap
# ════════════════════════════════════════════════════════════

@dataclass
class Abs2CapMatcherOutput():
    """
    This class defines the output structure for abstract-to-caption matchers.

    Attributes:
        sim_abs2cap (List[float]): Similarity scores between abstract and captions. Shape = [m].
    """

    sim_abs2cap: list[float]


class BaseAbs2CapMatcher(ABC):
    """
    Base class for abstract-to-caption matchers.
    This class provides a unified interface for models that compute similarity scores between abstracts and captions,
    supporting flexible integration of various lexical overlap metrics (e.g., ROUGE, BM25, BERTScore).

    Subclasses must implement the following methods:
        - `_get_score()`: Compute similarity score between abstract and caption.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def match(self, abstract: str, captions: list[str]) -> Abs2CapMatcherOutput:
        """
        Compute similarity scores between an abstract and a list of captions.

        Args:
            abstract (str): Abstract string.
            captions (List[str]): A List of caption strings.

        Returns:
            Abs2CapMatcherOutput: Object containing the following attributes:
                - sim_abs2cap (List[float]): Similarity scores between abstract and captions. Shape = [m].
        """

        raise NotImplementedError('Subclasses should implement this method')


class BaseAbs2CapMatcherLoader(ABC):
    """
    Base class for loading Abstract-to-Caption matching models.
    This class provides a common interface for initializing and loading models used in abstract-to-caption matching tasks.
    Subclasses should implement the `_load()` method to support different lexical overlap metrics (e.g., ROUGE-L, BM25, BERTScore).

    Attributes:
        model_type (str): A identifier to specify the lexical overlap metric (to be set by subclasses).

    Example:
        >>> class SubAbs2CapMatcherLoader(BaseAbs2CapMatcherLoader):
        ...     model_type = 'ROUGE'
        ...
        ...     def __init__(self, **kwargs):
        ...         super().__init__()
        ...
        ...     def _load(self):
        ...         return SubAbs2CapMatcher()

        >>> MODEL_REGISTRY = {
        ...     model_loader.model_type: model_loader
        ...     for model_loader in BaseAbs2CapMatcherLoader.__subclasses__()
        ... }
        >>> loader = MODEL_REGISTRY['ROUGE']()
        >>> model = loader.get_model()
    """

    model_type: str
    default_model_name: str

    def __init__(self) -> None:
        self.model = self._load()

    @abstractmethod
    def _load(self) -> BaseAbs2CapMatcher:
        """
        Load the model for GA/non-GA binary classification (GA-BC)
        """

        raise NotImplementedError('Subclasses should implement this method')

    def get_model(self) -> BaseAbs2CapMatcher:
        """
        Get the loaded model instance.

        Returns:
            nn.Module: The loaded model instance.
        """

        return self.model


# ════════════════════════════════════════════════════════════
# 📘 Intra-GA Recommendation | (ii) GA-BC
# ════════════════════════════════════════════════════════════

@dataclass
class GAClassifierOutput():
    """
    This class defines the output structure for GA/non-GA binary classifier models designed for intra-GA Recommendation.

    Attributes:
        loss (torch.Tensor): Cross-entropy loss.
        prob (torch.Tensor): Predicted probabilities for each class. Values range from 0 to 1. 0 for non-GA, 1 for GA. Shape = [batch_size, num_classes].
        pred (torch.Tensor): Predicted binary class label. 0 for non-GA, 1 for GA. Shape = [batch_size].
    """

    loss: torch.Tensor
    prob: torch.Tensor
    pred: torch.Tensor


class BaseGAClassifier(nn.Module):
    """
    Base class for GA/non-GA binary classification models used in Intra-GA Recommendation.
    This class provides a unified interface for models that classify whether a figure is GA or non-GA,
    supporting flexible integration of various image encoder backbones (e.g., ViT, EfficientNet, ConvNext).

    Subclasses must implement the following methods:
        - `get_backbone()`: Get the backbone of the model.
        - `preprocess_image()`: Preprocess input images.
        - `_classify_image()`: Compute logits for the input image.
    """

    def __init__(self, loss_weight: torch.Tensor = None) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    @abstractmethod
    def get_backbone(self) -> nn.Module:
        """
        Get the backbone of the model.

        Returns:
            nn.Module: The backbone of the model.
        """

        return NotImplementedError('Subclasses should implement this method')

    @abstractmethod
    def preprocess_image(self, image: Image.Image | ImageFile.ImageFile) -> torch.Tensor:
        """
        Preprocess input images.

        Args:
            image (Union[PIL.Image.Image, PIL.ImageFile.ImageFile]): Input image to be preprocessed.

        Returns:
            torch.Tensor: Preprocessed image as a tensor. Shape = [num_channels, image_height, image_width].
        """

        return NotImplementedError('Subclasses should implement this method')

    @abstractmethod
    def _classify_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for the input image.

        Args:
            image (torch.Tensor): Preprocessed image. Shape = [batch_size, num_channels, image_height, image_width].

        Returns:
            torch.Tensor: Logits. Shape = [batch_size, num_classes].
        """

        raise NotImplementedError('Subclasses should implement this method')

    def forward(
        self,
        image: torch.Tensor,
        label: torch.Tensor = None,
    ) -> GAClassifierOutput:
        """
        Forward pass for the binary classifier.

        Args:
            image (torch.Tensor): Preprocessed image. Shape = [batch_size, num_channels, image_height, image_width].
            label (torch.Tensor, optional): Ground truth binary label indicating whether each input image is a GA. 0 for non-GA, 1 for GA. Shape = [batch_size].

        Returns:
            GAClassifierOutput: Object containing the following attributes:
                - 'prob' (torch.Tensor): Predicted probabilities for each class. Values range from 0 to 1. Shape = [batch_size, num_classes].
                - 'pred' (torch.Tensor): Predicted binary class label. 0 for non-GA, 1 for GA. Shape = [batch_size].
                - 'loss' (torch.Tensor): Cross-entropy loss.
        """

        logit = self._classify_image(image)

        prob = F.softmax(logit, dim=-1)
        pred = prob.argmax(dim=1, keepdim=True)
        loss = F.cross_entropy(logit, label, weight=self.loss_weight.to(logit.device)) if label is not None else None

        return GAClassifierOutput(
            loss=loss,
            prob=prob,
            pred=pred,
        )


class BaseGAClassifierLoader(ABC):
    """
    Base class for loading GA/non-GA binary classification models.
    This class provides a common interface for initializing and loading models used in GA binary classification tasks
    Subclasses should implement the `_load()` method to support different backbone architectures (e.g., ViT, EfficientNet, ConvNext).

    Attributes:
        model_type (str): A identifier to specify the backbone architecture (to be set by subclasses).
        default_model_name (str): The default pretrained model name, HuggingFace model identifier, or path (to be set by subclasses).

    Args:
        model_name (str, optional): Identifier used to load the model. Depending on the model type, this may refer to:
            - A HuggingFace repository name (e.g., 'google/vit-large-patch16-224-in21k' for ViT)
            - A local path to a pretrained checkpoint directory (e.g., `'./models/x2vlm_large_4m/'`)
            - A shorthand string used by the underlying model loader (e.g., 'ViT-L/14' for CLIP via openai/clip, 'efficientnet_v2_l' for torchvision)
        loss_weight (torch.Tensor, optional): Tensor representing class weighting for loss. Shape = [num_classes].

    Example:
        >>> class SubGAClassifierLoader(BaseGAClassifierLoader):
        ...     model_type = 'vit'
        ...     default_model_name = 'google/vit-large-patch16-224-in21k'
        ...
        ...     def __init__(self, model_name: str = None, loss_weight: torch.Tensor = None, **kwargs):
        ...         super().__init__(model_name, loss_weight)
        ...
        ...     def _load(self):
        ...         return SubGAClassifier(self.model_name)

        >>> MODEL_REGISTRY = {
        ...     model_loader.model_type: model_loader
        ...     for model_loader in BaseGAClassifierLoader.__subclasses__()
        ... }
        >>> loader = MODEL_REGISTRY['vit']()
        >>> model = loader.get_model()
    """

    model_type: str
    default_model_name: str

    def __init__(self, model_name: str = None, loss_weight: torch.Tensor = None) -> None:
        self.model_name = model_name or self.default_model_name
        self.loss_weight = loss_weight
        self.model = self._load()

    @abstractmethod
    def _load(self) -> BaseGAClassifier:
        """
        Load the model for GA/non-GA binary classification (GA-BC)
        """

        raise NotImplementedError('Subclasses should implement this method')

    def get_model(self) -> BaseGAClassifier:
        """
        Get the loaded model instance.

        Returns:
            nn.Module: The loaded model instance.
        """

        return self.model


# ════════════════════════════════════════════════════════════
# 📘 Intra-GA Recommendation | (iii - iv) Abs2Fig
# ════════════════════════════════════════════════════════════

@dataclass
class Abs2FigRetrieverOutputForIntraGARecommendation():
    """
    This class defines the output structure for abstract-to-figure retrieval models designed for intra-GA Recommendation.

    Attributes:
        intra_loss (torch.Tensor): Intra loss.
        sim_abs2fig (torch.Tensor): Similarity between abstracts and figures. Shape = [batch_size, m+1].
        abstract_embed (torch.Tensor): Encoded abstracts. Shape = [batch_size, 1, embedding_dim].
        figures_embed (torch.Tensor): Encoded figures. Shape = [batch_size, m, embedding_dim].
    """

    intra_loss: torch.Tensor
    sim_abs2fig: torch.Tensor
    abstract_embed: torch.Tensor
    figures_embed: torch.Tensor


class BaseAbs2FigRetrieverForIntraGARecommendation(nn.Module):
    """
    Base class for abstract-to-figure retrieval models used in Intra-GA Recommendation.
    This class provides a unified interface for models that retrieve GAs from abstracts,
    supporting flexible integration of various CLIP-like vision-language backbones (e.g., CLIP, Long-CLIP, BLIP-2).

    Subclasses must implement the following methods:
        - `get_backbone()`: Get the backbone of the model.
        - `tokenize()`: Tokenize input texts.
        - `preprocess_image()`: Preprocess input images.
        - `encode_text()`: Encode tokenized texts into embeddings.
        - `encode_image()`: Encode images into embeddings.
        - `_logit_scale()`: Return the logit scale parameter (used as the temperature τ in contrastive learning).
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_backbone(self) -> nn.Module:
        """
        Get the backbone of the model.

        Returns:
            nn.Module: The backbone of the model.
        """

        return NotImplementedError('Subclasses should implement this method')

    @abstractmethod
    def tokenize(self, text: str | list[str], batched: bool = True) -> torch.Tensor | BatchEncoding:
        """
        Tokenize input texts.

        Args:
            text (Union[str, List[str]]): Input text to be tokenized.
            batched (bool, optional): If True, the input is treated as a batch and the output will preserve the batch dimension. If False, assumes a single text and squeezes the batch dimension (e.g., shape [max_length]).

        Returns:
            Union[torch.Tensor, BatchEncoding]: Tokenized text as a tensor. Shape = [batch_size, max_length] or [max_length].
        """

        return NotImplementedError('Subclasses should implement this method')

    @abstractmethod
    def preprocess_image(self, image: Image.Image | ImageFile.ImageFile) -> torch.Tensor:
        """
        Preprocess input images.

        Args:
            image (Union[PIL.Image.Image, PIL.ImageFile.ImageFile]): Input image to be preprocessed.

        Returns:
            torch.Tensor: Preprocessed image as a tensor. Shape = [num_channels, image_height, image_width].
        """

        return NotImplementedError('Subclasses should implement this method')

    @abstractmethod
    def encode_text(self, text: torch.Tensor | BatchEncoding) -> torch.Tensor:
        """
        Encode tokenized texts into embeddings.

        Args:
            text (Union[torch.Tensor, BatchEncoding]): Tokenized text. Shape = [batch_size, max_length] or [m, max_length].

        Returns:
            Tensor: Encoded text. Shape = [batch_size, embedding_dim] or [m, embedding_dim].
        """

        raise NotImplementedError('Subclasses should implement this method')

    @abstractmethod
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode preprocessed images into embeddings.

        Args:
            image (torch.Tensor): Preprocessed image. Shape = [batch_size, num_channels, image_height, image_width].
        Returns:
            Tensor: Encoded image. Shape = [batch_size, embedding_dim].
        """

        raise NotImplementedError('Subclasses should implement this method')

    @abstractmethod
    def _logit_scale(self) -> torch.Tensor:
        """
        Return the logit scale parameter (used as the temperature τ in contrastive learning).

        Returns:
            torch.Tensor: Logit scale parameter.
        """

        return NotImplementedError('Subclasses should implement this method')

    def _encode_abstract(self, abstract: torch.Tensor | BatchEncoding) -> torch.Tensor:
        """
        Encode abstract text using the model and processor.

        Args:
            abstract (Union[torch.Tensor, BatchEncoding]): Tokenized abstract text. Shape = [batch_size, max_length].

        Returns:
            Tensor: Encoded abstract text. Shape = [batch_size, 1, embedding_dim].
        """

        abstract_embed = self.encode_text(abstract)
        abstract_embed = abstract_embed.unsqueeze(1)
        return abstract_embed

    def _encode_figures(self, figures: torch.Tensor, captions: torch.Tensor | BatchEncoding = None) -> torch.Tensor:
        """
        Encode figures using the model and processor.

        Args:
            figures (torch.Tensor): Preprocessed figures. Shape = [batch_size, m, num_channels, image_height, image_width].
            captions (torch.Tensor, optional): Tokenized captions for each figure. Shape = [batch_size, m, max_length].
        Returns:
            Tensor: Encoded figures. Shape = [batch_size, m, embedding_dim].
        """

        figures_embed = torch.stack([self.encode_image(fig) for fig in figures])
        if captions is None:
            return figures_embed

        caption_batch_size = captions.shape[0] if isinstance(captions, torch.Tensor) else captions['input_ids'].shape[0]
        if figures.shape[0] != caption_batch_size:
            raise ValueError('Batch size of figures and captions must match.')

        caption_sample_size = captions.shape[1] if isinstance(captions, torch.Tensor) else captions['input_ids'].shape[1]
        if figures.shape[1] != caption_sample_size:
            raise ValueError('The number of figures and figure captions per paper must match.')

        # NOTE: Convert BatchEncoding of shape [batch_size, m, max_length] into a list of dicts with shape [m, max_length] for each sample
        if isinstance(captions, BatchEncoding):
            captions = [{key: captions[key][i] for key in captions} for i in range(captions['input_ids'].shape[0])]

        captions_embed = torch.stack([self.encode_text(caption) for caption in captions])
        figures_embed = torch.stack([figures_embed[i] * captions_embed[i] for i in range(len(figures_embed))])
        return figures_embed

    def forward(
        self,
        abstract: torch.Tensor,
        figures: torch.Tensor,
        captions: torch.Tensor = None,
        label: torch.Tensor = None,
    ) -> Abs2FigRetrieverOutputForIntraGARecommendation:
        """
        Forward pass for contrastive learning.

        Args:
            abstract (torch.Tensor): Tokenized abstracts. Shape = [batch_size, max_length].
            figures (torch.Tensor): Preprocessed figures. Shape = [batch_size, m, num_channels, image_height, image_width], where m is the fixed number of figures per paper (with zero-padding if needed).
            captions (torch.Tensor, optional): Tokenized captions for each figure. Shape = [batch_size, m, max_length].
            label (torch.Tensor, optional): Ground truth indices for the GA within the figures. Shape = [batch_size], where each element is an integer in [0, m) indicating the index of the GA.

        Returns:
            Abs2FigRetrieverForIntraGARecommendationOutput: Object containing the following attributes:
                - intra_loss (float): Intra loss.
                - sim_abs2fig (torch.Tensor): Similarity between abstracts and figures. Shape = [batch_size, m+1].
                - abstract_embed (torch.Tensor): Encoded abstracts. Shape = [batch_size, 1, embedding_dim].
                - figures_embed (torch.Tensor): Encoded figures. Shape = [batch_size, m, embedding_dim].
        """

        # Encode text and images
        abstract_embed = self._encode_abstract(abstract)
        figures_embed = self._encode_figures(figures, captions)

        # Normalize embeddings
        normalized_abstract_embed = abstract_embed / abstract_embed.norm(dim=-1, keepdim=True)
        normalized_figures_embed = figures_embed / figures_embed.norm(dim=-1, keepdim=True)

        # Compute similarity
        sim_abs2fig = torch.matmul(normalized_abstract_embed, normalized_figures_embed.transpose(-2, -1))
        sim_abs2fig = sim_abs2fig.squeeze(1)

        # Scaling
        sim_abs2fig = self._logit_scale().exp() * sim_abs2fig

        # Intra Loss
        intra_loss = F.cross_entropy(sim_abs2fig, label) if label is not None else None

        return Abs2FigRetrieverOutputForIntraGARecommendation(
            intra_loss=intra_loss,
            sim_abs2fig=sim_abs2fig,
            abstract_embed=abstract_embed,
            figures_embed=figures_embed,
        )


class BaseAbs2FigRetrieverLoaderForIntraGARecommendation(ABC):
    """
    Base class for loading abstract-to-figure retrieval models designed for intra-GA recommendation.
    This class provides a common interface for initializing and loading models used in abstract-to-figure retrieval tasks.
    Subclasses should implement the `_load()` method to support different backbone architectures (e.g., CLIP, Long-CLIP, BLIP-2).

    Attributes:
        model_type (str): A identifier to specify the backbone architecture (to be set by subclasses).
        default_model_name (str): The default pretrained model name, HuggingFace model identifier, or path (to be set by subclasses).

    Args:
        model_name (str, optional): The model name, HuggingFace model identifier, or path to the pretrained model to load.

    Example:
        >>> class SubAbs2FigRetrieverLoader(BaseAbs2FigRetrieverForIntraGARecommendationLoader):
        ...     model_type = 'clip'
        ...     default_model_name = 'ViT-L/14'
        ...
        ...     def __init__(self, model_name: str = None, **kwargs):
        ...         super().__init__(model_name)
        ...
        ...     def _load(self):
        ...         return SubAbs2FigRetriever(self.model_name)

        >>> MODEL_REGISTRY = {
        ...     model_loader.model_type: model_loader
        ...     for model_loader in BaseAbs2FigRetrieverForIntraGARecommendationLoader.__subclasses__()
        ... }
        >>> loader = MODEL_REGISTRY['clip']()
        >>> model = loader.get_model()
    """

    model_type: str
    default_model_name: str

    def __init__(self, model_name: str = None) -> None:
        self.model_name = model_name or self.default_model_name
        self.model = self._load()

    @abstractmethod
    def _load(self) -> BaseAbs2FigRetrieverForIntraGARecommendation:
        """
        Load the model for abstract-to-figure retrieval.
        """

        raise NotImplementedError('Subclasses should implement this method')

    def get_model(self) -> BaseAbs2FigRetrieverForIntraGARecommendation:
        """
        Get the loaded model instance.

        Returns:
            nn.Module: The loaded model instance.
        """

        return self.model


# ════════════════════════════════════════════════════════════
# 📙 Inter-GA Recommendation | (iii - iv) Abs2Fig
# ════════════════════════════════════════════════════════════

@dataclass
class Abs2FigRetrieverOutputForInterGARecommendation():
    """
    This class defines the output structure for abstract-to-figure retrieval models designed for inter-GA Recommendation.

    Attributes:
        inter_loss (torch.Tensor): Inter loss.
        sim_abs2GA (torch.Tensor): Similarity between abstracts and GAs. Shape = [batch_size, batch_size].
        sim_GA2abs (torch.Tensor): Similarity between GAs and abstracts. Shape = [batch_size, batch_size].
        abstract_embed (torch.Tensor): Encoded abstracts. Shape = [batch_size, embedding_dim].
        GAs_embed (torch.Tensor): Encoded GAs. Shape = [batch_size, embedding_dim].
    """

    inter_loss: torch.Tensor
    sim_abs2GA: torch.Tensor
    sim_GA2abs: torch.Tensor
    abstract_embed: torch.Tensor
    GA_embed: torch.Tensor


class BaseAbs2FigRetrieverForInterGARecommendation(nn.Module):
    """
    Base class for abstract-to-figure retrieval models used in Inter-GA Recommendation.
    This class provides a unified interface for models that retrieve GAs from abstracts,
    supporting flexible integration of various CLIP-like vision-language backbones (e.g., CLIP, Long-CLIP, BLIP-2).

    Subclasses must implement the following methods:
        - `get_backbone()`: Get the backbone of the model.
        - `tokenize()`: Tokenize input texts.
        - `preprocess_image()`: Preprocess input images.
        - `encode_text()`: Encode tokenized texts into embeddings.
        - `encode_image()`: Encode images into embeddings.
        - `_logit_scale()`: Return the logit scale parameter (used as the temperature τ in contrastive learning).
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_backbone(self) -> nn.Module:
        """
        Get the backbone of the model.

        Returns:
            nn.Module: The backbone of the model.
        """

        return NotImplementedError('Subclasses should implement this method')

    @abstractmethod
    def tokenize(self, text: str | list[str], batched: bool = True) -> torch.Tensor | BatchEncoding:
        """
        Tokenize input texts.

        Args:
            text (Union[str, List[str]]): Input text to be tokenized.
            batched (bool, optional): If True, the input is treated as a batch and the output will preserve the batch dimension. If False, assumes a single text and squeezes the batch dimension (e.g., shape [max_length]).

        Returns:
            Union[torch.Tensor, BatchEncoding]: Tokenized text as a tensor. Shape = [batch_size, max_length] or [max_length].
        """

        return NotImplementedError('Subclasses should implement this method')

    @abstractmethod
    def preprocess_image(self, image: Image.Image | ImageFile.ImageFile) -> torch.Tensor:
        """
        Preprocess input images.

        Args:
            image (Union[PIL.Image.Image, PIL.ImageFile.ImageFile]): Input image to be preprocessed.

        Returns:
            torch.Tensor: Preprocessed image as a tensor. Shape = [num_channels, image_height, image_width].
        """

        return NotImplementedError('Subclasses should implement this method')

    @abstractmethod
    def encode_text(self, text: torch.Tensor | BatchEncoding) -> torch.Tensor:
        """
        Encode tokenized texts into embeddings.

        Args:
            text (Union[torch.Tensor, BatchEncoding]): Tokenized text. Shape = [batch_size, max_length].

        Returns:
            Tensor: Encoded text. Shape = [batch_size, embedding_dim].
        """

        raise NotImplementedError('Subclasses should implement this method')

    @abstractmethod
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode preprocessed images into embeddings.

        Args:
            image (torch.Tensor): Preprocessed image. Shape = [batch_size, num_channels, image_height, image_width].

        Returns:
            Tensor: Encoded image. Shape = [batch_size, embedding_dim].
        """

        raise NotImplementedError('Subclasses should implement this method')

    @abstractmethod
    def _logit_scale(self) -> torch.Tensor:
        """
        Return the logit scale parameter (used as the temperature τ in contrastive learning).

        Returns:
            torch.Tensor: Logit scale parameter.
        """

        return NotImplementedError('Subclasses should implement this method')

    def _encode_abstract(self, abstract: torch.Tensor | BatchEncoding) -> torch.Tensor:
        """
        Encode abstract text using the model and processor.

        Args:
            abstract (Union[torch.Tensor, BatchEncoding]): Tokenized abstract text. Shape = [batch_size, max_length].

        Returns:
            Tensor: Encoded abstract text. Shape = [batch_size, embedding_dim].
        """

        abstract_embed = self.encode_text(abstract)
        return abstract_embed

    def _encode_GA(self, GA: torch.Tensor, caption: torch.Tensor | BatchEncoding = None) -> torch.Tensor:
        """
        Encode GA using the model and processor.

        Args:
            GA (torch.Tensor): Preprocessed GAs. Shape = [batch_size, num_channels, image_height, image_width].
            caption (torch.Tensor, optional): Tokenized captions for each GA. Shape = [batch_size, max_length].

        Returns:
            Tensor: Encoded GA. Shape = [batch_size, embedding_dim].
        """

        figures_embed = self.encode_image(GA)
        if caption is None:
            return figures_embed

        caption_batch_size = caption.shape[0] if isinstance(caption, torch.Tensor) else caption['input_ids'].shape[0]
        if GA.shape[0] != caption_batch_size:
            raise ValueError('Batch size of figures and captions must match.')

        captions_embed = self.encode_text(caption)
        figures_embed = figures_embed * captions_embed
        return figures_embed

    def forward(
        self,
        abstract: torch.Tensor,
        GA: torch.Tensor,
        caption: torch.Tensor = None,
    ) -> Abs2FigRetrieverOutputForInterGARecommendation:
        """
        Forward pass for contrastive learning.

        Args:
            abstract (torch.Tensor): Tokenized abstracts. Shape = [batch_size, max_length].
            GA (torch.Tensor): Preprocessed GAs. Shape = [batch_size, num_channels, image_height, image_width].
            caption (torch.Tensor, optional): Tokenized captions for each GA. Shape = [batch_size, max_length].

        Returns:
            OutputOfAbs2FigRetrieverForInterGARecommendation: Object containing the following attributes:
                - inter_loss (float): Inter loss.
                - sim_abs2GA (torch.Tensor): Similarity between abstracts and GAs. Shape = [batch_size, batch_size].
                - sim_GA2abs (torch.Tensor): Similarity between GAs and abstracts. Shape = [batch_size, batch_size].
                - abstract_embed (torch.Tensor): Encoded abstracts. Shape = [batch_size, embedding_dim].
                - GA_embed (torch.Tensor): Encoded GAs. Shape = [batch_size, embedding_dim].
        """

        # Encode text and images
        abstract_embed = self._encode_abstract(abstract)
        GA_embed = self._encode_GA(GA, caption)

        # Normalize embeddings
        normalized_abstract_embed = abstract_embed / abstract_embed.norm(dim=-1, keepdim=True)
        normalized_GA_embed = GA_embed / GA_embed.norm(dim=-1, keepdim=True)

        # Compute similarity
        sim_abs2GA = torch.matmul(normalized_abstract_embed, normalized_GA_embed.T)
        sim_GA2abs = torch.matmul(normalized_GA_embed, normalized_abstract_embed.T)

        # Scaling
        sim_abs2GA = self._logit_scale().exp() * sim_abs2GA
        sim_GA2abs = self._logit_scale().exp() * sim_GA2abs

        # Inter Loss
        labels = torch.arange(GA_embed.size(0), dtype=torch.long).to(sim_abs2GA.device)
        inter_loss = (F.cross_entropy(sim_abs2GA, labels) +
                      F.cross_entropy(sim_GA2abs, labels)) / 2

        return Abs2FigRetrieverOutputForInterGARecommendation(
            inter_loss=inter_loss,
            sim_abs2GA=sim_abs2GA,
            sim_GA2abs=sim_GA2abs,
            abstract_embed=abstract_embed,
            GA_embed=GA_embed,
        )


class BaseAbs2FigRetrieverLoaderForInterGARecommendation(ABC):
    """
    Base class for loading abstract-to-figure retrieval models designed for Inter-GA Recommendation.
    This class provides a common interface for initializing and loading models used in abstract-to-figure retrieval tasks.
    Subclasses should implement the `_load()` method to support different backbone architectures (e.g., CLIP, Long-CLIP, BLIP-2).

    Attributes:
        model_type (str): A identifier to specify the backbone architecture (to be set by subclasses).
        default_model_name (str): The default pretrained model name, HuggingFace model identifier, or path (to be set by subclasses).

    Args:
        model_name (str, optional): The model name, HuggingFace model identifier, or path to the pretrained model to load.

    Example:
        >>> class SubAbs2FigRetrieverLoader(BaseAbs2FigRetrieverForInterGARecommendationLoader):
        ...     model_type = 'clip'
        ...     default_model_name = 'ViT-L/14'
        ...
        ...     def __init__(self, model_name: str = None, **kwargs):
        ...         super().__init__(model_name)
        ...
        ...     def _load(self):
        ...         return SubAbs2FigRetriever(self.model_name)

        >>> MODEL_REGISTRY = {
        ...     model_loader.model_type: model_loader
        ...     for model_loader in BaseAbs2FigRetrieverForInterGARecommendationLoader.__subclasses__()
        ... }
        >>> loader = MODEL_REGISTRY['clip']()
        >>> model = loader.get_model()
    """

    model_type: str
    default_model_name: str

    def __init__(self, model_name: str = None) -> None:
        self.model_name = model_name or self.default_model_name
        self.model = self._load()

    @abstractmethod
    def _load(self) -> BaseAbs2FigRetrieverForInterGARecommendation:
        """
        Load the model for abstract-to-figure retrieval.
        """

        raise NotImplementedError('Subclasses should implement this method')

    def get_model(self) -> BaseAbs2FigRetrieverForInterGARecommendation:
        """
        Get the loaded model instance.

        Returns:
            nn.Module: The loaded model instance.
        """

        return self.model
