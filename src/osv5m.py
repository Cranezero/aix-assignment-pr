"""OSV-5M 메타데이터 로드, 표본 추출, 필요한 이미지만 내려받기.

OSV-5M 이미지 zip은 압축 없이(stored) 저장되어 있다. zip 목차(central directory)만 읽어
각 이미지의 바이트 위치를 얻고, HTTP Range 요청으로 필요한 이미지만 받는다.
train 전체(98개 zip, 약 246GB)를 내려받지 않아도 된다.
"""
import io
import struct
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm.auto import tqdm

HF_BASE = "https://huggingface.co/datasets/osv5m/osv5m/resolve/main"
N_ZIPS = {"train": 98, "test": 5}
META_COLS = ["id", "latitude", "longitude", "country", "sequence", "captured_at",
             "drive_side", "creator_id"]


# ---------------------------------------------------------------- 메타데이터
def download_metadata(split: str, out_dir: Path) -> Path:
    """train.csv(약 2.9GB) / test.csv(약 116MB)를 out_dir에 받는다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split}.csv"
    if path.exists():
        return path
    with requests.get(f"{HF_BASE}/{split}.csv", stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(path.with_suffix(".part"), "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                                              desc=f"{split}.csv") as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    path.with_suffix(".part").rename(path)
    return path


def load_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=META_COLS, dtype={"id": str, "sequence": str, "creator_id": str})
    # sequence가 비어 있으면 사진 한 장을 하나의 sequence로 본다.
    df["sequence"] = df["sequence"].fillna("img_" + df["id"])
    return df


# ---------------------------------------------------------------- 표본 추출
def sample_capped(df: pd.DataFrame, n: int, rng: np.random.Generator, cap: int) -> pd.DataFrame:
    """한 sequence에서 최대 cap장까지만 뽑아 n장을 고른다 (비슷한 사진이 몰리지 않게)."""
    shuffled = df.sample(frac=1, random_state=rng.integers(1 << 31))
    capped = shuffled.groupby("sequence", sort=False).head(cap)
    if len(capped) < n:
        raise ValueError(f"{df['country'].iloc[0]}: sequence당 {cap}장 제한으로 {len(capped)}장뿐 (필요 {n})")
    return capped.head(n)


def split_train_val(df: pd.DataFrame, n_train: int, n_val: int, rng: np.random.Generator,
                    cap: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """한 국가의 공식 train 풀에서 val과 train을 sequence 단위로 나눈다.
    같은 sequence가 train과 val에 동시에 들어가지 않는다."""
    seqs = df["sequence"].drop_duplicates().sample(frac=1, random_state=rng.integers(1 << 31)).tolist()
    # val에 쓸 sequence를 먼저 정한다. val 후보가 n_val의 2배가 될 때까지 sequence를 모은다.
    capped_sizes = df.groupby("sequence").size().clip(upper=cap)
    val_seqs, acc = [], 0
    for s in seqs:
        val_seqs.append(s)
        acc += capped_sizes[s]
        if acc >= 2 * n_val:
            break
    is_val = df["sequence"].isin(val_seqs)
    val = sample_capped(df[is_val], n_val, rng, cap)
    train = sample_capped(df[~is_val], n_train, rng, cap)
    return train, val


def make_splits(train_meta: pd.DataFrame, test_meta: pd.DataFrame, countries: list[str],
                n_train: int, n_val: int, n_test: int, seed: int = 42, cap: int = 5) -> dict:
    """국가별로 같은 장수를 뽑아 {'train','val','test'} DataFrame을 돌려준다."""
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for c in countries:
        tr, va = split_train_val(train_meta[train_meta["country"] == c], n_train, n_val, rng, cap)
        te = sample_capped(test_meta[test_meta["country"] == c], n_test, rng, cap)
        out["train"].append(tr)
        out["val"].append(va)
        out["test"].append(te)
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


SPLIT_COLS = ["id", "country", "sequence", "latitude", "longitude", "drive_side", "captured_at"]


def save_splits(splits: dict, splits_dir: Path) -> None:
    """data/splits/{train,val,test}.csv 저장. 국가 코드 순서, id 순으로 정렬해 diff가 안정적이게 한다."""
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df[SPLIT_COLS].sort_values(["country", "id"]).to_csv(splits_dir / f"{name}.csv", index=False)


def check_splits(splits: dict) -> None:
    """id 중복, train-val sequence 중복이 없는지 확인한다."""
    ids = [set(df["id"]) for df in splits.values()]
    assert all(len(a & b) == 0 for i, a in enumerate(ids) for b in ids[i + 1:]), "split 간 id 중복"
    seq_tr, seq_va = set(splits["train"]["sequence"]), set(splits["val"]["sequence"])
    assert not (seq_tr & seq_va), "train-val sequence 중복"
    for name, df in splits.items():
        counts = df["country"].value_counts()
        assert counts.nunique() == 1, f"{name}: 국가별 장수가 다르다"


# ---------------------------------------------------------------- zip 목차
def zip_url(split: str, k: int) -> str:
    return f"{HF_BASE}/images/{split}/{k:02d}.zip"


class _HTTPFile(io.RawIOBase):
    """zipfile이 목차를 읽을 수 있도록 Range 요청으로 동작하는 읽기 전용 파일."""

    def __init__(self, url: str, session: requests.Session):
        self.url, self.s = url, session
        self.size = int(session.head(url, allow_redirects=True, timeout=60).headers["content-length"])
        self.pos = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        end = self.size - 1 if n is None or n < 0 else min(self.pos + n, self.size) - 1
        if end < self.pos:
            return b""
        data = _get_range(self.s, self.url, self.pos, end)
        self.pos += len(data)
        return data


def _get_range(session, url, start, end, retries: int = 5) -> bytes:
    for attempt in range(retries):
        try:
            r = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=60)
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def index_zip(split: str, k: int, session: requests.Session = None) -> pd.DataFrame:
    """zip 하나의 목차를 읽어 id, zip 번호, 오프셋, 크기를 돌려준다."""
    session = session or requests.Session()
    with zipfile.ZipFile(_HTTPFile(zip_url(split, k), session)) as z:
        rows = [(Path(i.filename).stem, k, i.header_offset, i.compress_size, i.compress_type)
                for i in z.infolist() if i.filename.endswith(".jpg")]
    return pd.DataFrame(rows, columns=["id", "zip", "offset", "size", "compress"])


def build_index(split: str, ids: set, session: requests.Session = None) -> pd.DataFrame:
    """모든 zip 목차를 읽고 필요한 id의 위치만 남긴다."""
    session = session or requests.Session()
    parts = []
    for k in tqdm(range(N_ZIPS[split]), desc=f"index {split}"):
        idx = index_zip(split, k, session)
        parts.append(idx[idx["id"].isin(ids)])
    index = pd.concat(parts, ignore_index=True)
    missing = set(ids) - set(index["id"])
    if missing:
        raise ValueError(f"{split}: zip에서 찾지 못한 id {len(missing)}개")
    if (index["compress"] != zipfile.ZIP_STORED).any():
        raise ValueError("압축된 항목이 있다. 이 방식은 stored 항목만 지원한다.")
    return index.drop(columns="compress").assign(split=split)


INDEX_FILE = "image_index.csv"


def load_index(splits_dir: Path) -> pd.DataFrame:
    """data/splits/image_index.csv: id, zip, offset, size, split(공식 split 이름: train/test)."""
    return pd.read_csv(Path(splits_dir) / INDEX_FILE, dtype={"id": str})


# ---------------------------------------------------------------- 내려받기·축소
def resize_short_side(img: Image.Image, size: int = 224) -> Image.Image:
    """짧은 변을 size로 맞춘다 (비율 유지)."""
    w, h = img.size
    s = size / min(w, h)
    return img.resize((round(w * s), round(h * s)), Image.BICUBIC)


def download_images(split: str, index: pd.DataFrame, out_dir: Path, size: int = 224,
                    workers: int = 16) -> list[str]:
    """index의 이미지를 받아 짧은 변 size로 축소해 out_dir/{id}.jpg로 저장한다.
    이미 있는 파일은 건너뛴다. 실패한 id 목록을 돌려준다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = index[~index["id"].map(lambda i: (out_dir / f"{i}.jpg").exists())]
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers)
    session.mount("https://", adapter)

    failed = []
    for k, group in todo.groupby("zip"):
        # 서명된 CDN 주소는 시간이 지나면 만료되므로 zip마다 새로 얻는다.
        url = session.head(zip_url(split, k), allow_redirects=True, timeout=60).url

        def fetch(row):
            try:
                # 로컬 헤더(30바이트 + 파일명 + extra) 뒤에 데이터가 있다. extra 길이를 몰라 여유를 둔다.
                raw = _get_range(session, url, row.offset, row.offset + 30 + 512 + row.size)
                n, m = struct.unpack("<HH", raw[26:30])
                data = raw[30 + n + m: 30 + n + m + row.size]
                img = Image.open(io.BytesIO(data)).convert("RGB")
                resize_short_side(img, size).save(out_dir / f"{row.id}.jpg", quality=90)
                return None
            except Exception:
                return row.id

        with ThreadPoolExecutor(workers) as ex:
            res = list(tqdm(ex.map(fetch, group.itertuples()), total=len(group),
                            desc=f"{split}/{k:02d}.zip", leave=False))
        failed += [r for r in res if r is not None]
    return failed
