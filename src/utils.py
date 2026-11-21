import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """python, numpy, torch 난수를 모두 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def export_outputs(drive_dir=None) -> None:
    """Colab에서 만든 results/와 assets/를 Drive의 outputs/로 복사한다.
    Colab 런타임이 끝나면 clone한 저장소가 사라지므로 노트북 마지막에 실행한다."""
    import shutil
    from .config import ASSETS_DIR, DRIVE_DIR, RESULTS_DIR
    out = (drive_dir or DRIVE_DIR) / "outputs"
    for src in (RESULTS_DIR, ASSETS_DIR):
        if src.exists():
            shutil.copytree(src, out / src.name, dirs_exist_ok=True)
    print(f"복사 완료: {out}")


def restore_outputs(drive_dir=None) -> None:
    """Drive의 outputs/에서 저장소의 results/, assets/에 없는 파일만 가져온다.
    앞 노트북을 다른 런타임에서 실행했을 때 그 결과를 이어서 쓰기 위함이다."""
    import shutil
    from .config import ASSETS_DIR, DRIVE_DIR, RESULTS_DIR
    out = (drive_dir or DRIVE_DIR) / "outputs"
    for dst in (RESULTS_DIR, ASSETS_DIR):
        dst.mkdir(parents=True, exist_ok=True)
        src = out / dst.name
        if not src.exists():
            continue
        for f in src.iterdir():
            if f.is_file() and not (dst / f.name).exists():
                shutil.copy2(f, dst / f.name)
