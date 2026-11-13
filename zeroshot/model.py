import time
from collections.abc import Sequence

import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from transformers import CLIPModel

from . import config


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class ZeroShotCLIP:
    def __init__(self, model_id: str = config.MODEL_ID, device: torch.device | None = None) -> None:
        self.device = device or get_device()
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.tokenizer = Tokenizer.from_file(hf_hub_download(model_id, "tokenizer.json"))
        eos_id = self.tokenizer.token_to_id("<|endoftext|>")
        self.tokenizer.enable_padding(pad_id=eos_id, pad_token="<|endoftext|>")
        self.tokenizer.enable_truncation(max_length=self.model.config.text_config.max_position_embeddings)

    @torch.no_grad()
    def encode_text(self, prompts: Sequence[str]) -> torch.Tensor:
        encodings = self.tokenizer.encode_batch(list(prompts))
        input_ids = torch.tensor([e.ids for e in encodings], device=self.device)
        attention_mask = torch.tensor([e.attention_mask for e in encodings], device=self.device)
        features = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return features / features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def build_classifier(self, classnames: Sequence[str], templates: Sequence[str]) -> torch.Tensor:
        prompts = [template.format(name) for template in templates for name in classnames]
        embeddings = self.encode_text(prompts).view(len(templates), len(classnames), -1)
        mean = embeddings.mean(dim=0)
        return mean / mean.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_images(self, pixel_values: torch.Tensor, batch_size: int = config.BATCH_SIZE) -> torch.Tensor:
        chunks = []
        for batch in pixel_values.split(batch_size):
            features = self.model.get_image_features(pixel_values=batch.to(self.device)).pooler_output
            chunks.append(features / features.norm(dim=-1, keepdim=True))
        return torch.cat(chunks)

    @torch.no_grad()
    def predict(self, pixel_values: torch.Tensor, classifier: torch.Tensor) -> torch.Tensor:
        image_features = self.encode_images(pixel_values)
        logits = self.model.logit_scale.exp() * image_features @ classifier.T
        return logits.argmax(dim=-1).cpu()

    def timed_classifier(self, classnames: Sequence[str], templates: Sequence[str]) -> tuple[torch.Tensor, float]:
        _synchronize(self.device)
        start = time.perf_counter()
        classifier = self.build_classifier(classnames, templates)
        _synchronize(self.device)
        return classifier, time.perf_counter() - start

    def timed_predict(self, pixel_values: torch.Tensor, classifier: torch.Tensor) -> tuple[torch.Tensor, float]:
        _synchronize(self.device)
        start = time.perf_counter()
        predictions = self.predict(pixel_values, classifier)
        _synchronize(self.device)
        return predictions, time.perf_counter() - start
