# OpenAI CLIP 모델 전이학습 단계에 따른 OSV-5M 데이터셋 거리 사진 국가 분류

**Members**

| 이름 | 학과 | 이메일 |
| --- | --- | --- |
| 이학영 | 경영학부 | cranezero1103@gmail.com |
| 장윤도 | 화학과 | zhangrt2024@yeah.net |
| 류현성 | 전자공학부 | ksi7080z@naver.com |
| 이름4 | 학과 | email@hanyang.ac.kr |

**Video**: [YouTube 링크](https://www.youtube.com/) <!-- 최종 제출 시 실제 링크로 교체 -->

**Table of Contents**
[I. Proposal](#i-proposal) · [II. Datasets](#ii-datasets) · [III. Methodology](#iii-methodology) · [IV. Evaluation & Analysis](#iv-evaluation--analysis) · [V. Related Work](#v-related-work) · [VI. Conclusion](#vi-conclusion)

---

## I. Proposal

### Motivation
GeoGuessr에서 사람은 표지판 문자, 차선, 식생 등의 단서로 사진이 촬영된 국가를 추론한다. 본 프로젝트는 이미지–텍스트 쌍으로 사전학습된 CLIP이 같은 과제를 어느 수준까지 수행하는지 검증한다.

### Goal
1. 학습 없는 제로샷 분류의 성능을 측정한다.
2. 모델에 대한 개입 깊이(분류기만 학습 → 인코더 일부 학습)에 따른 성능 변화를 비교한다.
3. 국가 간 혼동 양상과 그 원인을 분석한다.

### Dataset
[OSV-5M](https://huggingface.co/datasets/osv5m/osv5m)의 거리 사진 중 15~20개국을 선정하여 국가당 500~1,000장을 사용한다. 테스트 세트는 공식 test split에서 추출한다.

### Approach

| 단계 | 방법 | 학습 대상 |
| --- | --- | --- |
| 1 | 제로샷 분류 | 없음 |
| 2 | 선형 프로브 | 분류기 |
| 3 | 부분 파인튜닝 | 인코더 마지막 블록 + 분류층 |

---

## II. Datasets

### Source
[OSV-5M](https://huggingface.co/datasets/osv5m/osv5m): 전 세계 거리 사진 약 510만 장, 225개 국가·지역, 사진별 국가·위경도 라벨 (CC-BY-SA-4.0).

### Country Selection
<!-- 메타데이터 국가별 장수 집계 결과와 선정 기준, 최종 국가 목록 -->

| 국가 | 코드 | Train | Val | Test |
| --- | --- | --- | --- | --- |
| — | — | — | — | — |

### Split & Preprocessing
<!-- 공식 test split 사용 근거(누수 방지), train:val = 85:15, 224×224 리사이즈, 국가당 장수 균등화 -->

### Exploration
<!-- 국가별 예시 그리드, 사람이 쓰는 단서(문자·통행 방향·식생) 정리 -->

![국가별 예시 이미지](assets/eda_grid.png)

---

## III. Methodology

### Why CLIP
<!-- 이미지·문장의 공동 임베딩 공간, 제로샷 분류 원리, 수작업 특징 없음 -->

### Common Setup
- 모델: `openai/clip-vit-base-patch32`
- 라이브러리: Hugging Face transformers, PyTorch, scikit-learn
- 환경: Google Colab (GPU), seed 고정

### Method 1 — Zero-shot Classification
<!-- 프롬프트 생성 → 전처리 → 유사도 → Top-1/Top-5 예측. 핵심 코드 일부 -->

### Method 2 — Linear Probe
<!-- 고정 인코더 임베딩 추출 → 로지스틱 회귀, 검증 세트로 정규화 강도 선택 -->

### Method 3 — Partial Fine-tuning
<!-- 마지막 블록 + 분류층 학습, 분류층을 문장 임베딩으로 초기화, 하이퍼파라미터, 증강(좌우 반전 제외 이유) -->

---

## IV. Evaluation & Analysis

### Overall Comparison

| 방법 | Top-1 | Top-5 | Macro Acc. | 학습 시간 |
| --- | --- | --- | --- | --- |
| 제로샷 | — | — | — | — |
| 선형 프로브 | — | — | — | — |
| 부분 파인튜닝 | — | — | — | — |

### Prompt Comparison (Method 1)

| 프롬프트 | Top-1 |
| --- | --- |
| `Korea` | — |
| `a photo of Korea` | — |
| `a street photo taken in Korea` | — |

### Learning Curve (Method 3)
![학습 곡선](assets/learning_curve.png)

### Confusion Matrix
![혼동행렬: 제로샷 vs 파인튜닝](assets/confusion_matrix.png)

### Case Study
<!-- 제로샷 오답·파인튜닝 정답 사례, 전 방법 오답 사례와 해석 -->

---

## V. Related Work

- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [OpenStreetView-5M](https://huggingface.co/papers/2404.18873)
- [OSV-5M Dataset](https://huggingface.co/datasets/osv5m/osv5m)
- [Hugging Face CLIP Documentation](https://huggingface.co/docs/transformers/model_doc/clip)

---

## VI. Conclusion

### Discussion
<!-- 학습 없이 도달한 성능, 개입 깊이에 따라 개선된 점과 개선되지 않은 점 -->

### Limitations
<!-- 일부 국가·거리 사진 한정, 소규모 데이터의 과적합 가능성, CLIP 사전학습 데이터와의 중복 가능성 -->

### Contributions
<!-- 팀원별 역할 (과제 Task 5) -->

---

## Appendix: Reproduction

### Repository Structure

```
notebooks/      실험 노트북 (아래 순서대로 실행)
src/            노트북 공통 모듈 (데이터 로드, CLIP 유틸, 평가 지표)
data/splits/    train/val/test 분할 CSV (이미지 ID–국가)
results/        방법별 예측·지표
assets/         블로그 게시용 그림
requirements.txt
```

### Setup (Google Colab)

```bash
git clone https://github.com/<user>/<repo>.git  # 공개 저장소 주소로 교체
cd <repo>
pip install -r requirements.txt
```

| 순서 | 노트북 | 내용 |
| --- | --- | --- |
| 1 | [`01_data_prep.ipynb`](notebooks/01_data_prep.ipynb) | 데이터 다운로드·샘플링·분할 |
| 2 | [`02_eda.ipynb`](notebooks/02_eda.ipynb) | 데이터 탐색 |
| 3 | [`03_zeroshot.ipynb`](notebooks/03_zeroshot.ipynb) | 방법 1 |
| 4 | [`04_linear_probe.ipynb`](notebooks/04_linear_probe.ipynb) | 방법 2 |
| 5 | [`05_finetune.ipynb`](notebooks/05_finetune.ipynb) | 방법 3 |
| 6 | [`06_analysis.ipynb`](notebooks/06_analysis.ipynb) | 결과 분석 |

본 저장소에는 원본 이미지와 모델 가중치를 포함하지 않는다. 게시된 예시 이미지의 출처는 OSV-5M (CC-BY-SA-4.0)이다.