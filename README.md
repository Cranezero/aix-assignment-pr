# OpenAI CLIP 모델 전이학습 단계에 따른 OSV-5M 데이터셋 거리 사진 국가 분류

**Members**

| 이름 | 학과 | 이메일 |
| --- | --- | --- |
| 이학영 | 경영학부 | cranezero1103@gmail.com |
| 장윤도 | 화학과 | zhangrt2024@yeah.net |
| 류현성 | 전자공학부 | ksi7080z@naver.com |
| 정재호 | 컴퓨터소프트웨어학부 | jaeho041005@outlook.com |

**Video**: (작성중)

## Table of Contents

- [I. Proposal](#i-proposal)
  - [Motivation](#motivation)
  - [Goal](#goal)
  - [Dataset](#dataset)
  - [Approach](#approach)
- [II. Datasets](#ii-datasets)
  - [Source](#source)
  - [Country Selection](#country-selection)
  - [Split & Preprocessing](#split--preprocessing)
  - [Exploration](#exploration)
- [III. Methodology](#iii-methodology)
  - [Why CLIP](#why-clip)
  - [Common Setup](#common-setup)
  - [Method 1: Zero-shot Classification](#method-1-zero-shot-classification)
  - [Method 2: Linear Probe](#method-2-linear-probe)
  - [Method 3: Partial Fine-tuning](#method-3-partial-fine-tuning)
- [IV. Evaluation & Analysis](#iv-evaluation--analysis)
  - [Overall Comparison](#overall-comparison)
  - [Prompt Comparison](#prompt-comparison)
  - [Learning Curve](#learning-curve)
  - [Confusion Matrix](#confusion-matrix)
  - [Case Study](#case-study)
- [V. Related Work](#v-related-work)
- [VI. Conclusion](#vi-conclusion)
  - [Discussion](#discussion)
  - [Limitations](#limitations)
  - [Contributions](#contributions)
- [Appendix: Reproduction](#appendix-reproduction)

## I. Proposal

### Motivation
GeoGuessr 플레이어는 표지판 문자, 통행 방향, 전신주, 식생 같은 단서로 거리 사진의 국가를 맞힌다. CLIP은 이미지-텍스트 쌍 약 4억 개로 사전학습되었고, 사진 설명에는 지명이 자주 등장한다. 따라서 CLIP이 국가별 시각적 특징을 이미 학습했을 수 있다.

본 프로젝트는 CLIP이 거리 사진의 국가를 얼마나 맞히는지, 그리고 선형 프로브와 부분 파인튜닝으로 학습 범위를 넓히면 성능이 어떻게 달라지는지 탐구한다.

### Goal
1. **제로샷**: 학습 없이 국가를 얼마나 맞히는지, 프롬프트 표현에 따라 결과가 달라지는지 측정한다.
2. **선형 프로브와 부분 파인튜닝**: 분류기만 학습할 때와 인코더 마지막 블록까지 학습할 때 성능이 제로샷 대비 얼마나 변하는지 비교한다.
3. **혼동 분석**: 어떤 국가끼리 혼동하는지, 학습 후 혼동이 줄어드는지 분석한다.

### Dataset
[OSV-5M](https://huggingface.co/datasets/osv5m/osv5m)에서 15~20개국, 국가당 500~1,000장을 사용한다. 한국-일본, 스페인-포르투갈처럼 비슷한 국가 쌍을 포함한다. 테스트 세트는 공식 test split에서 추출한다.

### Approach
세 방법 모두 같은 사전학습 CLIP(ViT-B/32)에서 시작하고, 학습 범위만 다르다.

| 단계 | 방법 | 학습 대상 | 방식 |
| --- | --- | --- | --- |
| 1 | 제로샷 분류 | 없음 | 사진과 "a street photo taken in {국가}" 문장의 유사도로 분류 |
| 2 | 선형 프로브 | 분류기 | 고정된 이미지 임베딩으로 로지스틱 회귀 학습 |
| 3 | 부분 파인튜닝 | 인코더 마지막 블록 + 분류층 | 인코더 일부를 우리 데이터로 학습 |

- 모든 방법을 같은 테스트 세트로 평가한다.
- 부분 파인튜닝의 분류층은 제로샷의 국가별 문장 임베딩으로 초기화한다. 제로샷과 같은 지점에서 학습을 시작하므로 성능 차이가 곧 학습 효과다.
- 지표는 Top-1, Top-5, macro 정확도, 혼동행렬이다.

## II. Datasets

### Source
[OSV-5M](https://huggingface.co/datasets/osv5m/osv5m): 전 세계 거리 사진 약 510만 장, 225개 국가·지역, 사진별 국가·위경도 라벨 (CC-BY-SA-4.0).

### Country Selection
(작성중)

### Split & Preprocessing
(작성중)

### Exploration
(작성중)

## III. Methodology

### Why CLIP
(작성중)

### Common Setup
- 모델: `openai/clip-vit-base-patch32`
- 라이브러리: Hugging Face transformers, PyTorch, scikit-learn
- 환경: Google Colab (GPU), seed 고정

### Method 1: Zero-shot Classification
(작성중)

### Method 2: Linear Probe
(작성중)

### Method 3: Partial Fine-tuning
(작성중)

## IV. Evaluation & Analysis

### Overall Comparison
(작성중)

### Prompt Comparison
(작성중)

### Learning Curve
(작성중)

### Confusion Matrix
(작성중)

### Case Study
(작성중)

## V. Related Work

- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [OpenStreetView-5M](https://huggingface.co/papers/2404.18873)
- [OSV-5M Dataset](https://huggingface.co/datasets/osv5m/osv5m)
- [Hugging Face CLIP Documentation](https://huggingface.co/docs/transformers/model_doc/clip)

## VI. Conclusion

### Discussion
(작성중)

### Limitations
(작성중)

### Contributions
(작성중)

## Appendix: Reproduction

### Repository Structure

```
notebooks/      실험 노트북 (아래 순서대로 실행)
src/            노트북 공통 모듈 (데이터 로드, CLIP 유틸, 평가 지표)
data/splits/    train/val/test 분할 CSV (이미지 ID, 국가)
results/        방법별 예측·지표
assets/         블로그 게시용 그림
requirements.txt
```

### Setup (Google Colab)

저장소 clone: (작성중)

```bash
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
