"""CLIP 로드, 프롬프트, 이미지·문장 임베딩."""
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

from .config import MODEL_NAME, NAMES

# 프롬프트 비교에 쓰는 세 가지 템플릿. 기본은 "street".
PROMPT_TEMPLATES = {
    "name": "{}",
    "photo": "a photo of {}",
    "street": "a street photo taken in {}",
}


def load_clip(device, model_name: str = MODEL_NAME):
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def build_prompts(template: str = "street", names=NAMES) -> list[str]:
    return [PROMPT_TEMPLATES[template].format(n) for n in names]


# get_text_features/get_image_features는 transformers 버전에 따라 반환 형식이 달라서
# (4.x: 텐서, 5.x: 출력 객체) 인코더 출력에 projection을 직접 적용한다. 계산은 같다.
def text_features(model, **tokens) -> torch.Tensor:
    return model.text_projection(model.text_model(**tokens).pooler_output)


def image_features(model, pixel_values) -> torch.Tensor:
    return model.visual_projection(model.vision_model(pixel_values=pixel_values).pooler_output)


@torch.no_grad()
def encode_texts(model, processor, texts, device) -> torch.Tensor:
    """정규화된 문장 임베딩 (N, 512)."""
    tokens = processor(text=texts, return_tensors="pt", padding=True).to(device)
    return F.normalize(text_features(model, **tokens), dim=-1)


@torch.no_grad()
def encode_images(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """정규화된 이미지 임베딩 (M, 512)과 라벨 (M,). CPU 텐서로 돌려준다."""
    embs, labels = [], []
    for x, y in tqdm(loader, desc="encode"):
        emb = image_features(model, x.to(device))
        embs.append(F.normalize(emb, dim=-1).cpu())
        labels.append(y)
    return torch.cat(embs), torch.cat(labels)


def zeroshot_probs(image_emb: torch.Tensor, text_emb: torch.Tensor, logit_scale: float) -> torch.Tensor:
    """코사인 유사도 x logit_scale -> softmax. CLIP 학습 시와 같은 온도(약 100)를 쓴다."""
    logits = logit_scale * image_emb @ text_emb.T
    return logits.softmax(dim=-1)
