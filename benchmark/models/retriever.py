import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import (
    BaseAbs2FigRetrieverForIntraGARecommendation,
    Abs2FigRetrieverOutputForIntraGARecommendation,
    BaseAbs2FigRetrieverForInterGARecommendation,
    Abs2FigRetrieverOutputForInterGARecommendation,
)
import clip
from benchmark.submodules.longclip import model as longclip
import open_clip
from transformers import Blip2ForImageTextRetrieval, AutoProcessor, BatchEncoding
from benchmark.submodules.x2vlm.models.model_retrieval import XVLMPlusForRetrieval
from benchmark.submodules.x2vlm.dataset import build_tokenizer
import yaml
from torchvision import transforms
from PIL import Image, ImageFile


# ════════════════════════════════════════════════════════════
# 📘 Intra-GA Recommendation | (iii - iv) Abs2Fig
# ════════════════════════════════════════════════════════════

class CLIPAsAbs2FigRetrieverForIntraGARecommendation(BaseAbs2FigRetrieverForIntraGARecommendation):
    """
    CLIP (https://doi.org/10.48550/arXiv.2103.00020) for Intra-GA Recommendation.
    """

    def __init__(
        self,
        model_name: str,
    ) -> None:
        """
        Loads CLIP (https://github.com/openai/CLIP).
        """

        super().__init__()
        self.model, self.processor = clip.load(model_name, device='cpu')

    def get_backbone(self) -> nn.Module:
        return self.model

    def tokenize(self, text: str | list[str], batched: bool = True) -> torch.Tensor:
        tokenized_text = clip.tokenize(text, truncate=True)
        tokenized_text = tokenized_text if batched else tokenized_text.squeeze(0)
        return tokenized_text

    def preprocess_image(self, image: Image.Image | ImageFile.ImageFile) -> torch.Tensor:
        preprocessed_image = self.processor(image)
        return preprocessed_image

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        encoded_text = self.model.encode_text(text)
        return encoded_text

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        encoded_image = self.model.encode_image(image)
        return encoded_image

    def _logit_scale(self):
        return self.model.logit_scale


class LongCLIPAsAbs2FigRetrieverForIntraGARecommendation(BaseAbs2FigRetrieverForIntraGARecommendation):
    """
    Long-CLIP (https://doi.org/10.48550/arXiv.2403.15378) for Intra-GA Recommendation.
    """

    def __init__(
        self,
        model_name: str,
    ) -> None:
        """
        Loads LongCLIP (https://github.com/beichenzbc/Long-CLIP).
        """

        super().__init__()
        self.model, self.processor = longclip.load(model_name, device='cpu')

    def get_backbone(self) -> nn.Module:
        return self.model

    def tokenize(self, text: str | list[str], batched: bool = True) -> torch.Tensor:
        tokenized_text = longclip.tokenize(text, truncate=True)
        tokenized_text = tokenized_text if batched else tokenized_text.squeeze(0)
        return tokenized_text

    def preprocess_image(self, image: Image.Image | ImageFile.ImageFile) -> torch.Tensor:
        preprocessed_image = self.processor(image)
        return preprocessed_image

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        encoded_text = self.model.encode_text(text)
        return encoded_text

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        encoded_image = self.model.encode_image(image)
        return encoded_image

    def _logit_scale(self):
        return self.model.logit_scale


class OpenCLIPAsAbs2FigRetrieverForIntraGARecommendation(BaseAbs2FigRetrieverForIntraGARecommendation):
    """
    OpenCLIP (https://doi.org/10.48550/arXiv.2212.07143) for Intra-GA Recommendation
    """

    def __init__(
        self,
        model_name: str,
    ) -> None:
        """
        Loads OpenCLIP (https://huggingface.co/docs/hub/open_clip).
        """

        super().__init__()
        self.model, self.processor = open_clip.create_model_from_pretrained(model_name, device='cpu')
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def get_backbone(self) -> nn.Module:
        return self.model

    def tokenize(self, text: str | list[str], batched: bool = True) -> torch.Tensor:
        tokenized_text = self.tokenizer(text)
        tokenized_text = tokenized_text if batched else tokenized_text.squeeze(0)
        return tokenized_text

    def preprocess_image(self, image: Image.Image | ImageFile.ImageFile) -> torch.Tensor:
        preprocessed_image = self.processor(image)
        return preprocessed_image

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        encoded_text = self.model.encode_text(text)
        return encoded_text

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        encoded_image = self.model.encode_image(image)
        return encoded_image

    def _logit_scale(self):
        return self.model.logit_scale


class BLIP2AsAbs2FigRetrieverForIntraGARecommendation(BaseAbs2FigRetrieverForIntraGARecommendation):
    """
    BLIP-2 (https://doi.org/10.48550/arXiv.2301.12597) for Intra-GA Recommendation
    """

    def __init__(
        self,
        model_name: str,
    ) -> None:
        """
        Loads BLIP-2 (https://huggingface.co/docs/transformers/main/en/model_doc/blip-2).
        """

        super().__init__()
        self.model = Blip2ForImageTextRetrieval.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def get_backbone(self) -> nn.Module:
        return self.model

    def tokenize(self, text: str | list[str], batched: bool = True) -> BatchEncoding:
        tokenized_text = self.processor(text=text, padding='max_length', truncation=True, return_tensors='pt')
        if not batched:
            for key in tokenized_text:
                tokenized_text[key] = tokenized_text[key].squeeze(0)

        return tokenized_text

    def preprocess_image(self, image: Image.Image | ImageFile.ImageFile) -> torch.Tensor:
        preprocessed_image = self.processor(images=image, return_tensors='pt')['pixel_values'].squeeze(0)
        return preprocessed_image

    def encode_text(self, text: BatchEncoding) -> torch.Tensor:
        query_embeds = self.model.embeddings(
            input_ids=text['input_ids'],
        )
        qformer_outputs = self.model.qformer(
            query_embeds=query_embeds,
            query_length=0,
            attention_mask=text['attention_mask'],
            return_dict=True,
        )
        qformer_hidden_states = qformer_outputs[0]
        encoded_text = self.model.text_projection(qformer_hidden_states[:, 0, :])
        return encoded_text

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        encoder_outputs = self.model.vision_model(pixel_values=image, return_dict=True)
        encoder_hidden_states = encoder_outputs[0]
        encoder_attention_mask = torch.ones(encoder_hidden_states.size()[:-1], dtype=torch.long, device=encoder_hidden_states.device)
        query_embeds = self.model.query_tokens.expand(encoder_hidden_states.shape[0], -1, -1)
        qformer_outputs = self.model.qformer(
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True
        )
        qformer_hidden_states = qformer_outputs[0]
        encoded_image = self.model.vision_projection(qformer_hidden_states)
        return encoded_image

    # NOTE: The encode_figures() function is overridden to adapt the q-former architecture of BLIP-2.
    def _encode_figures(self, figures: torch.Tensor, captions: BatchEncoding = None) -> torch.Tensor:
        """
        Encode figures using the model and processor.

        Args:
            figures (torch.Tensor): Preprocessed figures. Shape = [batch_size, m, num_channels, image_height, image_width].
            captions (torch.Tensor, optional): Tokenized captions for each figure. Shape = [batch_size, m, max_length].

        Returns:
            Tensor: Encoded figures. Shape = [batch_size, m, num_query_tokens, embedding_dim].
        """

        figures_embed = torch.stack([self.encode_image(fig) for fig in figures])
        if captions is None:
            return figures_embed

        if figures.shape[0] != captions['input_ids'].shape[0]:
            raise ValueError('Batch size of figures and captions must match.')
        if figures.shape[1] != captions['input_ids'].shape[1]:
            raise ValueError('The number of figures and figure captions per paper must match.')

        # NOTE: Convert BatchEncoding of shape [batch_size, m, max_length] into a list of dicts with shape [m, max_length] for each sample
        captions = [{key: captions[key][i] for key in captions} for i in range(captions['input_ids'].shape[0])]

        captions_embed = torch.stack([self.encode_text(caption).unsqueeze(1).repeat(1, 32, 1) for caption in captions])
        figures_embed = torch.stack([figures_embed[i] * captions_embed[i] for i in range(len(figures_embed))])
        return figures_embed

    def _logit_scale(self):
        return torch.tensor(np.log(1 / 0.07), device=self.model.device)

    # NOTE: The forward() function is overridden to adapt the q-former architecture of BLIP-2.
    def forward(
        self,
        abstract: torch.Tensor,
        figures: torch.Tensor,
        captions: torch.Tensor = None,
        labels: torch.Tensor = None,
    ) -> Abs2FigRetrieverOutputForIntraGARecommendation:
        # Encode text and images
        abstract_embed = self._encode_abstract(abstract)
        figures_embed = self._encode_figures(figures, captions)

        # Normalize embeddings
        normalized_abstract_embed = abstract_embed / abstract_embed.norm(dim=-1, keepdim=True)
        normalized_figures_embed = figures_embed / figures_embed.norm(dim=-1, keepdim=True)

        # Compute Similarity
        sim_abs2fig = torch.matmul(normalized_abstract_embed.unsqueeze(1), normalized_figures_embed.transpose(-2, -1))
        sim_abs2fig, _ = sim_abs2fig.max(dim=-1)
        sim_abs2fig = sim_abs2fig.squeeze(-1)

        # Scaling
        sim_abs2fig = self._logit_scale().exp() * sim_abs2fig

        # Intra loss
        intra_loss = F.cross_entropy(sim_abs2fig, labels) if labels is not None else None

        return Abs2FigRetrieverOutputForIntraGARecommendation(
            intra_loss=intra_loss,
            sim_abs2fig=sim_abs2fig,
            abstract_embed=abstract_embed,
            figures_embed=figures_embed,
        )


class X2VLMAsAbs2FigRetrieverForIntraGARecommendation(BaseAbs2FigRetrieverForIntraGARecommendation):
    """
    X2-VLM (https://doi.org/10.48550/arXiv.2211.12402) for Intra-GA Recommendation
    """

    def __init__(
        self,
        model_config_path: str,
        model_name: str,
        image_size: int = 224,
    ) -> None:
        """
        Loads X2-VLM (https://github.com/zengyan-97/X2-VLM)
        """

        super().__init__()
        self.config = yaml.load(open(model_config_path, 'r'), Loader=yaml.Loader)
        self.model = XVLMPlusForRetrieval(config=self.config)
        self.model.load_pretrained(model_name, self.config)
        self.tokenizer = build_tokenizer(self.config['text_encoder'])
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def get_backbone(self) -> nn.Module:
        return self.model

    def tokenize(self, text: str | list[str], batched: bool = True) -> BatchEncoding:
        tokenized_text = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.config['max_tokens'], return_tensors='pt')
        if not batched:
            for key in tokenized_text:
                tokenized_text[key] = tokenized_text[key].squeeze(0)

        return tokenized_text

    def preprocess_image(self, image: Image.Image | ImageFile.ImageFile) -> torch.Tensor:
        preprocessed_image = self.transform(image)
        return preprocessed_image

    def encode_text(self, text: BatchEncoding) -> torch.Tensor:
        text_embeds = self.model.get_text_embeds(text['input_ids'], text['attention_mask'])
        encoded_text = self.model.get_features(text_embeds=text_embeds)
        return encoded_text

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        vision_embeds, _ = self.model.get_vision_embeds(image)
        encoded_image = self.model.get_features(image_embeds=vision_embeds)
        return encoded_image

    def _logit_scale(self):
        device = next(self.model.parameters()).device
        return torch.tensor(np.log(1 / 0.07), device=device)


# ════════════════════════════════════════════════════════════
# 📙 Inter-GA Recommendation | (iii - iv) Abs2Fig
# ════════════════════════════════════════════════════════════

class CLIPAsAbs2FigRetrieverForInterGARecommendation(BaseAbs2FigRetrieverForInterGARecommendation):
    """
    CLIP (https://doi.org/10.48550/arXiv.2103.00020) for Inter-GA Recommendation.
    """

    def __init__(
        self,
        model_name: str,
    ) -> None:
        """
        Loads CLIP (https://github.com/openai/CLIP).
        """

        super().__init__()
        self.model, self.processor = clip.load(model_name, device='cpu')

    def get_backbone(self) -> nn.Module:
        return self.model

    def tokenize(self, text: str | list[str], batched: bool = True) -> torch.Tensor:
        tokenized_text = clip.tokenize(text, truncate=True)
        tokenized_text = tokenized_text if batched else tokenized_text.squeeze(0)
        return tokenized_text

    def preprocess_image(self, image: Image.Image | ImageFile.ImageFile) -> torch.Tensor:
        preprocessed_image = self.processor(image)
        return preprocessed_image

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        encoded_text = self.model.encode_text(text)
        return encoded_text

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        encoded_image = self.model.encode_image(image)
        return encoded_image

    def _logit_scale(self):
        return self.model.logit_scale


class LongCLIPAsAbs2FigRetrieverForInterGARecommendation(BaseAbs2FigRetrieverForInterGARecommendation):
    """
    Long-CLIP (https://doi.org/10.48550/arXiv.2403.15378) for Inter-GA Recommendation.
    """

    def __init__(
        self,
        model_name: str,
    ) -> None:
        """
        Loads LongCLIP (https://github.com/beichenzbc/Long-CLIP).
        """

        super().__init__()
        self.model, self.processor = longclip.load(model_name, device='cpu')

    def get_backbone(self) -> nn.Module:
        return self.model

    def tokenize(self, text: str | list[str], batched: bool = True) -> torch.Tensor:
        tokenized_text = longclip.tokenize(text, truncate=True)
        tokenized_text = tokenized_text if batched else tokenized_text.squeeze(0)
        return tokenized_text

    def preprocess_image(self, image: Image.Image | ImageFile.ImageFile) -> torch.Tensor:
        preprocessed_image = self.processor(image)
        return preprocessed_image

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        encoded_text = self.model.encode_text(text)
        return encoded_text

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        encoded_image = self.model.encode_image(image)
        return encoded_image

    def _logit_scale(self):
        return self.model.logit_scale


class OpenCLIPAsAbs2FigRetrieverForInterGARecommendation(BaseAbs2FigRetrieverForInterGARecommendation):
    """
    OpenCLIP (https://doi.org/10.48550/arXiv.2212.07143) for Inter-GA Recommendation
    """

    def __init__(
        self,
        model_name: str,
    ) -> None:
        """
        Loads OpenCLIP (https://huggingface.co/docs/hub/open_clip).
        """

        super().__init__()
        self.model, self.processor = open_clip.create_model_from_pretrained(model_name, device='cpu')
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def get_backbone(self) -> nn.Module:
        return self.model

    def tokenize(self, text: str | list[str], batched: bool = True) -> torch.Tensor:
        tokenized_text = self.tokenizer(text)
        tokenized_text = tokenized_text if batched else tokenized_text.squeeze(0)
        return tokenized_text

    def preprocess_image(self, image: Image.Image | ImageFile.ImageFile) -> torch.Tensor:
        preprocessed_image = self.processor(image)
        return preprocessed_image

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        encoded_text = self.model.encode_text(text)
        return encoded_text

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        encoded_image = self.model.encode_image(image)
        return encoded_image

    def _logit_scale(self):
        return self.model.logit_scale


class BLIP2AsAbs2FigRetrieverForInterGARecommendation(BaseAbs2FigRetrieverForInterGARecommendation):
    """
    BLIP-2 (https://doi.org/10.48550/arXiv.2301.12597) for Inter-GA Recommendation
    """

    def __init__(
        self,
        model_name: str,
    ) -> None:
        """
        Loads BLIP-2 (https://huggingface.co/docs/transformers/main/en/model_doc/blip-2).
        """

        super().__init__()
        self.model = Blip2ForImageTextRetrieval.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def get_backbone(self) -> nn.Module:
        return self.model

    def tokenize(self, text: str | list[str], batched: bool = True) -> BatchEncoding:
        tokenized_text = self.processor(text=text, padding='max_length', truncation=True, return_tensors='pt')
        if not batched:
            for key in tokenized_text:
                tokenized_text[key] = tokenized_text[key].squeeze(0)

        return tokenized_text

    def preprocess_image(self, image: Image.Image | ImageFile.ImageFile) -> torch.Tensor:
        preprocessed_image = self.processor(images=image, return_tensors='pt')['pixel_values'].squeeze(0)
        return preprocessed_image

    def encode_text(self, text: BatchEncoding) -> torch.Tensor:
        """
        Encode text using the model and processor.

        Args:
            text (torch.Tensor): Tokenized text. {"input_ids": Shape = [batch_size, max_length], "attention_mask": Shape = [batch_size, max_length]}.

        Returns:
            Tensor: Encoded text. Shape = [batch_size, embedding_dim].
        """

        query_embeds = self.model.embeddings(
            input_ids=text['input_ids'],
        )
        qformer_outputs = self.model.qformer(
            query_embeds=query_embeds,
            query_length=0,
            attention_mask=text['attention_mask'],
            return_dict=True,
        )
        qformer_hidden_states = qformer_outputs[0]
        encoded_text = self.model.text_projection(qformer_hidden_states[:, 0, :])
        return encoded_text

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image using the model and processor.

        Args:
            image (torch.Tensor): Preprocessed image. Shape = [(batch_size | m), num_channels, image_height, image_width].

        Returns:
            Tensor: Encoded image. Shape = [(batch_size | m), num_query_tokens, embedding_dim].
        """

        encoder_outputs = self.model.vision_model(pixel_values=image, return_dict=True)
        encoder_hidden_states = encoder_outputs[0]
        encoder_attention_mask = torch.ones(encoder_hidden_states.size()[:-1], dtype=torch.long, device=encoder_hidden_states.device)
        query_embeds = self.model.query_tokens.expand(encoder_hidden_states.shape[0], -1, -1)
        qformer_outputs = self.model.qformer(
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True
        )
        qformer_hidden_states = qformer_outputs[0]
        encoded_image = self.model.vision_projection(qformer_hidden_states)
        return encoded_image

    # NOTE: The encode_GA() function is overridden to adapt the q-former architecture of BLIP-2.
    def _encode_GA(self, GA: torch.Tensor, caption: BatchEncoding = None) -> torch.Tensor:
        """
        Encode GA using the model and processor.

        Args:
            GA (torch.Tensor): Preprocessed GAs. Shape = [batch_size, num_channels, image_height, image_width].
            caption (torch.Tensor, optional): Tokenized captions for each GA. Shape = [batch_size, max_length].

        Returns:
            Tensor: Encoded GA. Shape = [batch_size, num_query_tokens, embedding_dim].
        """
        figures_embed = self.encode_image(GA)
        if caption is None:
            return figures_embed

        if GA.shape[0] != caption['input_ids'].shape[0]:
            raise ValueError('Batch size of figures and captions must match.')

        captions_embed = self.encode_text(caption).unsqueeze(1).repeat(1, 32, 1)
        figures_embed = figures_embed * captions_embed
        return figures_embed

    def _logit_scale(self):
        return torch.tensor(np.log(1 / 0.07), device=self.model.device)

    # NOTE: The forward() function is overridden to adapt the q-former architecture of BLIP-2.
    def forward(
        self,
        abstract: torch.Tensor,
        GAs: torch.Tensor,
        captions: torch.Tensor = None,
    ) -> Abs2FigRetrieverOutputForInterGARecommendation:
        # Encode text and images
        abstract_embed = self._encode_abstract(abstract)
        GA_embed = self._encode_GA(GAs, captions)

        # Normalize embeddings
        normalized_abstract_embed = abstract_embed / abstract_embed.norm(dim=-1, keepdim=True)
        normalized_GA_embed = GA_embed / GA_embed.norm(dim=-1, keepdim=True)

        # Compute similarity
        sim_GA2abs = torch.matmul(normalized_abstract_embed, normalized_GA_embed.transpose(-2, -1))
        sim_GA2abs, _ = sim_GA2abs.max(dim=-1)
        sim_abs2GA = sim_GA2abs.T

        # Scaling
        sim_GA2abs = self._logit_scale().exp() * sim_GA2abs
        sim_abs2GA = self._logit_scale().exp() * sim_abs2GA

        # Inter loss
        labels = torch.arange(GA_embed.size(0), dtype=torch.long).to(sim_GA2abs.device)
        inter_loss = (F.cross_entropy(sim_GA2abs, labels) +
                      F.cross_entropy(sim_abs2GA, labels)) / 2

        return Abs2FigRetrieverOutputForInterGARecommendation(
            inter_loss=inter_loss,
            sim_abs2GA=sim_abs2GA,
            sim_GA2abs=sim_GA2abs,
            abstract_embed=abstract_embed,
            GA_embed=GA_embed,
        )


class X2VLMAsAbs2FigRetrieverForInterGARecommendation(BaseAbs2FigRetrieverForInterGARecommendation):
    """
    X2-VLM (https://doi.org/10.48550/arXiv.2211.12402) for Inter-GA Recommendation
    """

    def __init__(
        self,
        model_config_path: str,
        model_name: str,
        image_size: int = 224,
    ) -> None:
        """
        Loads X2-VLM (https://github.com/zengyan-97/X2-VLM)
        """

        super().__init__()
        self.config = yaml.load(open(model_config_path, 'r'), Loader=yaml.Loader)
        self.model = XVLMPlusForRetrieval(config=self.config)
        self.model.load_pretrained(model_name, self.config)
        self.tokenizer = build_tokenizer(self.config['text_encoder'])
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def get_backbone(self) -> nn.Module:
        return self.model

    def tokenize(self, text: str | list[str], batched: bool = True) -> BatchEncoding:
        tokenized_text = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.config['max_tokens'], return_tensors='pt')
        if not batched:
            for key in tokenized_text:
                tokenized_text[key] = tokenized_text[key].squeeze(0)

        return tokenized_text

    def preprocess_image(self, image: Image.Image | ImageFile.ImageFile) -> torch.Tensor:
        preprocessed_image = self.transform(image)
        return preprocessed_image

    def encode_text(self, text: BatchEncoding) -> torch.Tensor:
        text_embeds = self.model.get_text_embeds(text['input_ids'], text['attention_mask'])
        encoded_text = self.model.get_features(text_embeds=text_embeds)
        return encoded_text

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        vision_embeds, _ = self.model.get_vision_embeds(image)
        encoded_image = self.model.get_features(image_embeds=vision_embeds)
        return encoded_image

    def _logit_scale(self):
        device = next(self.model.parameters()).device
        return torch.tensor(np.log(1 / 0.07), device=device)
