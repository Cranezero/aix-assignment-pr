"""분할 CSV 로드, Dataset, transform."""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .config import CODE_TO_IDX, DRIVE_DIR, LOCAL_IMAGE_DIR, SPLITS_DIR

IMAGES_ZIP = "images.zip"


def prepare_images(drive_dir: Path = DRIVE_DIR, local_dir: Path = LOCAL_IMAGE_DIR) -> Path:
    """Drive의 images.zip(01_data_prep.ipynb가 만든 것)을 Colab 로컬 디스크에 푼다.
    Drive에서 작은 파일을 한 장씩 읽으면 느리므로 로컬에 풀어서 쓴다."""
    import zipfile
    local_dir = Path(local_dir)
    if local_dir.exists() and any(local_dir.glob("*.jpg")):
        return local_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(Path(drive_dir) / IMAGES_ZIP) as z:
        z.extractall(local_dir)
    return local_dir

# CLIPProcessor와 같은 정규화 값
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGE_SIZE = 224


def load_split(split: str, splits_dir: Path = SPLITS_DIR) -> pd.DataFrame:
    """split: 'train' | 'val' | 'test'. 열: id, country, label 등."""
    df = pd.read_csv(Path(splits_dir) / f"{split}.csv", dtype={"id": str})
    df["label"] = df["country"].map(CODE_TO_IDX)
    if df["label"].isna().any():
        missing = sorted(df.loc[df["label"].isna(), "country"].unique())
        raise ValueError(f"config.COUNTRIES에 없는 국가 코드: {missing}")
    df["label"] = df["label"].astype(int)
    return df


def get_transform(train: bool = False) -> T.Compose:
    """평가: CLIPProcessor와 같은 처리(짧은 변 224로 resize 후 중앙 crop).
    학습: 무작위 자르기와 밝기 조정만 쓴다. 좌우 반전은 통행 방향과 글자를 뒤집으므로 쓰지 않는다.
    """
    if train:
        return T.Compose([
            T.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0), ratio=(3 / 4, 4 / 3),
                                interpolation=T.InterpolationMode.BICUBIC),
            T.ColorJitter(brightness=0.2),
            T.ToTensor(),
            T.Normalize(CLIP_MEAN, CLIP_STD),
        ])
    return T.Compose([
        T.Resize(IMAGE_SIZE, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])


class StreetDataset(Dataset):
    """image_dir/{id}.jpg 를 읽어 (이미지 텐서, 라벨)을 돌려준다."""

    def __init__(self, df: pd.DataFrame, image_dir: Path, transform=None):
        self.ids = df["id"].tolist()
        self.labels = df["label"].tolist()
        self.image_dir = Path(image_dir)
        self.transform = transform or get_transform(train=False)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i):
        img = Image.open(self.image_dir / f"{self.ids[i]}.jpg").convert("RGB")
        return self.transform(img), torch.tensor(self.labels[i])
