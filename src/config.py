"""경로, seed, 모델 이름, 국가 목록 등 공통 설정."""
from pathlib import Path

SEED = 42
MODEL_NAME = "openai/clip-vit-base-patch32"

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "splits"
RESULTS_DIR = REPO_ROOT / "results"
ASSETS_DIR = REPO_ROOT / "assets"

# Colab 기준 경로. Drive에는 축소 이미지 zip과 체크포인트만 둔다.
DRIVE_DIR = Path("/content/drive/MyDrive/aix-osv5m")
LOCAL_IMAGE_DIR = Path("/content/images")

# 국가 코드(ISO 3166-1 alpha-2) -> 프롬프트용 영어 국가명. 순서가 클래스 인덱스가 된다.
COUNTRIES = {
    # 아시아
    "KR": "South Korea",
    "JP": "Japan",
    "TH": "Thailand",
    "ID": "Indonesia",
    "IN": "India",
    # 유럽
    "ES": "Spain",
    "PT": "Portugal",
    "FR": "France",
    "DE": "Germany",
    "GB": "the United Kingdom",
    "NO": "Norway",
    "SE": "Sweden",
    "RU": "Russia",
    # 아메리카
    "US": "the United States",
    "CA": "Canada",
    "MX": "Mexico",
    "BR": "Brazil",
    "AR": "Argentina",
    # 오세아니아·아프리카
    "AU": "Australia",
    "ZA": "South Africa",
}

# 국가당 장수 (모든 국가 동일). train:val = 85:15
N_TRAIN, N_VAL, N_TEST = 680, 120, 200
# 한 sequence(같은 사용자가 연속 촬영한 묶음)에서 뽑는 최대 장수
SEQ_CAP = 5

CODES = list(COUNTRIES)
NAMES = [COUNTRIES[c] for c in CODES]
CODE_TO_IDX = {c: i for i, c in enumerate(CODES)}
