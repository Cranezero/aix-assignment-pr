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
GeoGuessr는 Google 스트리트뷰의 임의 위치에 플레이어를 떨어뜨리고, 그곳이 어디인지 세계 지도에 표시하게 하는 웹 게임이다. 실제 위치와 가까이 찍을수록 점수가 높다. 국가만 맞히는 모드도 있다.

사진에는 위치 정보가 없다. 플레이어는 화면 속 단서로 위치를 추론한다.

- **문자와 언어**: 표지판과 간판의 문자 체계, 철자
- **도로**: 통행 방향, 차선 색, 도로변 기둥(볼라드) 모양
- **시설물**: 전신주 형태, 번호판 색과 모양
- **자연환경**: 식생, 토양 색, 지형, 하늘과 태양의 방향

숙련된 플레이어는 이런 단서를 조합해 사진 몇 초 만에 국가를 맞힌다. 이 지식은 수많은 사진을 보며 쌓은 경험에서 나온다.

CLIP도 비슷한 경험을 가지고 있을 수 있다. CLIP은 인터넷에서 수집한 이미지-텍스트 쌍 약 4억 개로 사전학습되었다. 여행 사진이나 풍경 사진의 설명에는 지명이 자주 함께 등장한다. 따라서 CLIP이 국가별 시각적 특징을 이미 학습했을 수 있다.

본 프로젝트는 CLIP이 거리 사진의 국가를 얼마나 맞히는지 탐구한다. 학습 없는 제로샷에서 시작해 선형 프로브, 부분 파인튜닝 순으로 학습 범위를 넓히며 성능이 어떻게 달라지는지 비교한다.

### Goal
1. **제로샷**: 학습 없이 국가를 얼마나 맞히는지 측정한다. 프롬프트 표현(`Korea`, `a photo of Korea`, `a street photo taken in Korea`)에 따라 결과가 달라지는지 확인한다.
2. **선형 프로브와 부분 파인튜닝**: 분류기만 학습할 때와 인코더 마지막 블록까지 학습할 때 성능이 제로샷 대비 얼마나 변하는지 비교한다. 학습 시간도 함께 비교한다.
3. **혼동 분석**: 어떤 국가끼리 혼동하는지, 학습 후 혼동이 줄어드는지 분석한다. 틀린 사진을 직접 보고 위의 단서가 사진에 있었는지 확인한다.

### Dataset
[OSV-5M](https://huggingface.co/datasets/osv5m/osv5m)은 전 세계 거리 사진 약 510만 장으로 구성된 데이터셋이다. 사진마다 국가와 위경도 라벨이 있다. GeoGuessr와 같은 거리 사진이므로 이 과제에 맞다.

이 중 15\~20개국, 국가당 500\~1,000장을 사용한다. 한국-일본, 스페인-포르투갈처럼 비슷해 보이는 국가 쌍을 포함한다. 한국과 일본은 문자와 통행 방향이 다르고, 스페인과 포르투갈은 언어가 비슷하다. 모델이 이런 차이를 구분하는지 확인하기 위함이다. 테스트 세트는 공식 test split에서 추출한다.

### Approach
세 방법 모두 같은 사전학습 CLIP(ViT-B/32)에서 시작하고, 학습 범위만 다르다. 사진의 특징은 CLIP 이미지 인코더가 추출한다. 수작업 특징은 만들지 않는다.

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
[OSV-5M](https://huggingface.co/datasets/osv5m/osv5m): 전 세계 거리 사진 약 510만 장(train 4,894,685장, test 210,122장), 225개 국가·지역, 사진별 국가·위경도 라벨 (CC-BY-SA-4.0). 사진은 Mapillary에 사용자들이 올린 거리 사진이며, 대부분 차량에서 촬영되었다. 메타데이터 CSV에는 국가 코드, 위경도, 촬영 시각, sequence ID(같은 사용자가 연속 촬영한 묶음) 등이 있다.

### Country Selection
20개국을 사용한다. 선정 기준은 다음과 같다.

- **장수**: 공식 test split에서 국가당 200장을 뽑을 수 있어야 한다. 비슷한 사진이 몰리지 않도록 sequence 하나에서는 최대 5장만 뽑으므로 실제 필요한 장수는 더 많다. 20개국 중 가장 적은 국가는 한국(test 616장)이다.
- **지역**: 아시아 5개국, 유럽 8개국, 아메리카 5개국, 오세아니아·아프리카 2개국을 포함한다.
- **혼동 쌍**: 서로 비슷해 보일 것으로 예상한 쌍을 포함한다.
  - 한국-일본: 도로 모습이 비슷하지만 문자와 통행 방향이 다르다.
  - 스페인-포르투갈: 언어와 풍경이 비슷하다.
  - 미국-캐나다, 노르웨이-스웨덴: 도로 표지와 식생이 비슷하다.
  - 호주-남아프리카공화국: 둘 다 좌측통행이고 건조한 풍경이 많다.

| 국가 | 코드 | 프롬프트 국가명 | OSV-5M train | OSV-5M test |
| --- | --- | --- | ---: | ---: |
| 한국 | KR | South Korea | 3,316 | 616 |
| 일본 | JP | Japan | 209,205 | 2,128 |
| 태국 | TH | Thailand | 74,763 | 2,682 |
| 인도네시아 | ID | Indonesia | 36,582 | 2,628 |
| 인도 | IN | India | 50,604 | 6,525 |
| 스페인 | ES | Spain | 96,508 | 2,470 |
| 포르투갈 | PT | Portugal | 51,551 | 689 |
| 프랑스 | FR | France | 242,074 | 2,047 |
| 독일 | DE | Germany | 272,282 | 1,204 |
| 영국 | GB | the United Kingdom | 137,648 | 1,491 |
| 노르웨이 | NO | Norway | 27,940 | 2,771 |
| 스웨덴 | SE | Sweden | 131,816 | 2,105 |
| 러시아 | RU | Russia | 222,351 | 14,984 |
| 미국 | US | the United States | 1,218,664 | 23,875 |
| 캐나다 | CA | Canada | 84,299 | 9,580 |
| 멕시코 | MX | Mexico | 53,468 | 4,759 |
| 브라질 | BR | Brazil | 140,599 | 10,425 |
| 아르헨티나 | AR | Argentina | 41,581 | 4,260 |
| 호주 | AU | Australia | 162,138 | 11,052 |
| 남아프리카공화국 | ZA | South Africa | 53,443 | 2,969 |

OSV-5M train/test 열은 원본 데이터셋 전체의 국가별 장수다. 이 중 국가마다 train 680장, val 120장, test 200장, 합계 1,000장을 뽑는다. 전체는 train 13,600장, val 2,400장, test 4,000장이다.

### Split & Preprocessing
**test는 공식 test split에서 뽑는다.** 같은 경로를 연속 촬영한 사진은 서로 매우 비슷하다. 이런 사진이 학습과 테스트에 나뉘어 들어가면 모델이 장소를 외워서 맞힐 수 있다. OSV-5M 논문은 공식 test를 만들 때 train 이미지와 1km 이내에 있거나 sequence ID를 공유하는 사진을 제외했다. 따라서 공식 test에서 뽑으면 이 누수를 피할 수 있다.

**train/val은 공식 train split에서 뽑아 85:15로 나눈다.** 이때도 같은 sequence가 train과 val에 함께 들어가지 않도록 sequence 단위로 나눈다. val은 선형 프로브의 정규화 강도 선택과 부분 파인튜닝의 조기 종료에 쓰므로, val이 train과 겹치면 두 결정이 낙관적으로 치우친다.

**국가당 장수를 같게 맞춘다.** 원본은 미국이 한국의 수백 배다. 장수를 맞추면 정확도가 특정 국가에 좌우되지 않는다.

분할 절차는 다음과 같다.

1. 국가별로 공식 train에서 sequence 목록을 섞고, val용 sequence를 먼저 고른다.
2. val용 sequence에서 120장, 나머지 sequence에서 680장을 뽑는다. 공식 test에서 200장을 뽑는다.
3. 모든 단계에서 sequence 하나당 최대 5장까지만 뽑는다. seed는 42로 고정한다.
4. 결과를 `data/splits/{train,val,test}.csv`(이미지 ID, 국가, sequence, 위경도 등)로 저장한다. 모든 방법이 이 파일을 그대로 쓴다.

**전처리**: 원본 이미지는 높이 512px이고 가로 길이는 사진마다 다르다. 짧은 변을 224px로 줄이고 비율은 유지해 저장한다. 정사각형으로 자르는 것은 모델 입력 단계에서 한다. 평가 때는 가운데를 224×224로 자르고, 부분 파인튜닝 학습 때는 무작위로 자른다.

**내려받기**: OSV-5M 이미지는 train 98개, test 5개 zip(합계 약 256GB)으로 배포된다. zip 안의 파일이 압축 없이 저장되어 있으므로, zip 목차에서 필요한 사진의 바이트 위치를 찾아 그 구간만 HTTP Range 요청으로 받는다. 이렇게 하면 2만 장(축소 후 약 350MB)만 받으면 된다. 위치 정보는 `data/splits/image_index.csv`에 저장했다.

### Exploration
국가별 예시 이미지: (작성중)

사람이 국가를 구분할 때 쓰는 단서 중 문자와 통행 방향을 정리하면 다음과 같다.

| 국가 | 문자 | 통행 방향 |
| --- | --- | --- |
| 한국 | 한글 | 우측 |
| 일본 | 한자, 가나 | 좌측 |
| 태국 | 타이 문자 | 좌측 |
| 인도네시아 | 라틴 문자 | 좌측 |
| 인도 | 데바나가리 등, 라틴 문자 | 좌측 |
| 스페인, 포르투갈, 프랑스, 독일, 노르웨이, 스웨덴 | 라틴 문자 (언어별 철자 차이) | 우측 |
| 영국 | 라틴 문자 | 좌측 |
| 러시아 | 키릴 문자 | 우측 |
| 미국, 캐나다 | 라틴 문자 (캐나다 일부 지역은 프랑스어) | 우측 |
| 멕시코, 브라질, 아르헨티나 | 라틴 문자 (브라질은 포르투갈어) | 우측 |
| 호주, 남아프리카공화국 | 라틴 문자 | 좌측 |

문자는 한국, 일본, 태국, 러시아처럼 고유 문자를 쓰는 국가에서 강한 단서다. 라틴 문자를 쓰는 국가끼리는 철자 차이나 도로 시설, 식생 같은 단서가 필요하다. 통행 방향은 20개국을 좌측통행 7개국과 우측통행 13개국으로 나눈다. 다만 사진 한 장에서 통행 방향이 항상 보이지는 않는다. 차가 없는 시골 도로에서는 판단이 어렵다.

OSV-5M 메타데이터에도 통행 방향 열(`drive_side`)이 있지만, 영국이 우측통행으로 잘못 표시되어 있어 위 표는 실제 통행 방향으로 적었다.

## III. Methodology

### Why CLIP
CLIP은 이미지 인코더와 텍스트 인코더로 구성된다. 인터넷에서 모은 이미지-텍스트 쌍 약 4억 개로, 짝이 맞는 이미지와 문장의 임베딩은 가깝게, 짝이 아닌 것은 멀게 학습했다. 그 결과 이미지와 문장이 같은 512차원 공간에 놓인다.

이 구조 덕분에 학습 없이 분류할 수 있다. "a street photo taken in Japan"처럼 클래스마다 문장을 만들고, 사진과 가장 가까운 문장을 고르면 된다. 이것이 제로샷 분류다.

또한 세 방법을 같은 출발점에서 비교할 수 있다. 제로샷은 CLIP을 그대로 쓰고, 선형 프로브는 같은 이미지 임베딩 위에 분류기만 학습하며, 부분 파인튜닝은 이미지 인코더 일부까지 학습한다. 학습 범위만 다르므로 성능 차이를 학습 범위의 효과로 해석할 수 있다.

사진의 특징은 CLIP 이미지 인코더가 추출한다. 색 히스토그램이나 에지 같은 수작업 특징은 쓰지 않는다.

사용하는 모델은 ViT-B/32다. 이미지를 32×32 패치 49개로 나눠 12층 Transformer로 처리한다. 파라미터는 이미지 인코더 약 8,800만 개, 텍스트 인코더 약 6,300만 개다. CLIP 중 작은 모델이라 Colab GPU에서 파인튜닝까지 할 수 있다.

### Common Setup
- 모델: `openai/clip-vit-base-patch32` (`CLIPModel`, `CLIPProcessor`)
- 라이브러리: Hugging Face transformers, PyTorch, scikit-learn
- 환경: Google Colab (GPU)
- seed: 42 (분할, 학습, 데이터 로더 모두)
- 입력: 짧은 변 224px로 축소된 이미지를 224×224로 자르고 CLIP의 평균·표준편차로 정규화한다.
- 평가: 세 방법 모두 같은 test 4,000장. 지표는 Top-1, Top-5, macro 정확도(국가별 정확도의 평균), 혼동행렬이다. 국가당 test 장수가 같으므로 Top-1과 macro 정확도는 같은 값이 된다. 국가별 정확도는 따로 표시한다.
- 공통 코드는 `src/`에 있다. 모든 방법의 테스트 예측을 같은 형식(`results/{method}_test_pred.csv`)으로 저장한다.

### Method 1: Zero-shot Classification
학습 없이 사진과 국가별 문장의 유사도로 분류한다.

1. 국가마다 문장 "a street photo taken in {국가명}"을 만든다. 20개 문장이다.
2. 텍스트 인코더로 문장 임베딩을, 이미지 인코더로 사진 임베딩을 구하고 각각 길이 1로 정규화한다.
3. 사진 임베딩과 20개 문장 임베딩의 코사인 유사도를 구한다. CLIP이 학습한 온도 값(logit scale, 약 100)을 곱한 뒤 softmax로 확률로 바꾼다.
4. 확률이 가장 높은 국가를 예측으로 하고, 상위 5개도 함께 기록한다.

```python
prompts = [f"a street photo taken in {name}" for name in NAMES]
tokens = processor(text=prompts, return_tensors="pt", padding=True)
text_emb = F.normalize(model.text_projection(model.text_model(**tokens).pooler_output), dim=-1)
img_emb = F.normalize(model.visual_projection(model.vision_model(pixel_values=x).pooler_output), dim=-1)

probs = (model.logit_scale.exp() * img_emb @ text_emb.T).softmax(dim=-1)  # (사진 수, 20)
top5 = probs.topk(5, dim=-1).indices
```

프롬프트 표현에 따라 결과가 달라지는지도 확인한다. 같은 사진 임베딩에 세 가지 문장 템플릿을 적용해 비교한다.

| 템플릿 | 예시 |
| --- | --- |
| `{국가명}` | South Korea |
| `a photo of {국가명}` | a photo of South Korea |
| `a street photo taken in {국가명}` | a street photo taken in South Korea |

### Method 2: Linear Probe
이미지 인코더를 고정하고, 그 임베딩 위에 로지스틱 회귀 분류기만 학습한다.

1. 고정된 이미지 인코더로 train, val, test 사진의 임베딩(512차원, 길이 1로 정규화)을 한 번만 추출한다. 증강은 쓰지 않는다.
2. train 임베딩으로 다항 로지스틱 회귀(scikit-learn `LogisticRegression`)를 학습한다.
3. 정규화 강도 C를 0.01부터 1000까지 로그 간격 11개 값으로 바꿔 가며 학습하고, val 정확도가 가장 높은 C를 고른다.
4. 고른 C로 학습한 분류기로 test를 평가한다.

제로샷의 분류기는 사람이 쓴 문장 20개이고, 선형 프로브의 분류기는 데이터로 학습한 가중치다. 이미지 표현은 둘이 같다. 따라서 두 방법의 차이는 분류기를 데이터로 학습한 효과다. 인코더를 학습하지 않으므로 학습이 몇 초 안에 끝난다.

### Method 3: Partial Fine-tuning
이미지 인코더의 마지막 Transformer 블록(12층 중 12번째)과 분류층만 학습하고 나머지는 고정한다.

**분류층 초기화**: 분류층은 512차원 입력, 20개 출력의 선형층이다. 가중치를 방법 1의 국가별 문장 임베딩으로 초기화하고, 출력에 CLIP의 logit scale을 곱한다. 이렇게 하면 학습 전 모델의 식이 제로샷과 똑같다. 학습 전 test 정확도가 제로샷과 같은지 확인한 뒤 학습을 시작한다. 같은 지점에서 출발하므로 학습 후 성능 차이가 곧 학습 효과다.

```python
class CLIPClassifier(nn.Module):
    def __init__(self, clip, text_emb, logit_scale):
        super().__init__()
        self.vision = clip.vision_model
        self.proj = clip.visual_projection
        self.head = nn.Linear(512, 20, bias=False)
        self.head.weight.data.copy_(text_emb)      # 제로샷 문장 임베딩으로 초기화
        self.register_buffer("scale", torch.tensor(logit_scale))

    def forward(self, x):
        feat = self.proj(self.vision(pixel_values=x).pooler_output)
        return self.scale * self.head(F.normalize(feat, dim=-1))
```

**학습 대상**: 마지막 블록(약 710만 개)과 분류층(약 1만 개). 전체 이미지 모델 약 8,800만 개 중 8%다. 앞쪽 층은 선, 질감 같은 일반적인 특징을 담당하므로 고정하고, 고수준 특징을 만드는 마지막 블록만 우리 데이터에 맞춘다. 데이터가 13,600장으로 적어 전체를 학습하면 과적합 위험이 크다.

**하이퍼파라미터**

| 항목 | 값 |
| --- | --- |
| 옵티마이저 | AdamW |
| 학습률 | 마지막 블록 1e-5, 분류층 1e-3 |
| 배치 크기 | 64 |
| epoch | 최대 10, val 손실이 2 epoch 연속 줄지 않으면 조기 종료 |
| 손실 | 교차 엔트로피 |
| 정밀도 | float16 혼합 정밀도 |

학습률은 사전학습된 블록이 크게 망가지지 않도록 작게, 새로 학습하는 분류층은 크게 둔다. 테스트에는 val 손실이 가장 낮았던 epoch의 가중치를 쓴다.

**데이터 증강**: 무작위 자르기(원본 면적의 60\~100%)와 밝기 조정(±20%)만 쓴다. 좌우 반전은 쓰지 않는다. 좌우를 뒤집으면 우측통행 도로가 좌측통행처럼 보이고 표지판 글자도 뒤집힌다. 통행 방향과 문자는 국가를 구분하는 핵심 단서이므로 반전은 라벨을 바꾸는 것과 같다.

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
data/splits/    train/val/test 분할 CSV (이미지 ID, 국가), 이미지 위치 색인
results/        방법별 예측·지표
assets/         블로그 게시용 그림
requirements.txt
```

### Setup (Google Colab)

각 노트북의 첫 셀이 아래 작업을 한다. Colab에서 노트북을 열고 위에서부터 실행하면 된다.

```bash
git clone https://github.com/Cranezero/aix-assignment-pr.git
cd aix-assignment-pr
pip install -r requirements.txt
```

첫 셀은 Google Drive도 연결한다. 축소 이미지(`images.zip`), 체크포인트, 임베딩은 Drive의 `MyDrive/aix-osv5m/`에 저장된다. 각 노트북은 마지막 셀에서 `results/`와 `assets/`를 Drive의 `MyDrive/aix-osv5m/outputs/`로 복사한다.

| 순서 | 노트북 | 내용 |
| --- | --- | --- |
| 1 | [`01_data_prep.ipynb`](notebooks/01_data_prep.ipynb) | 이미지 다운로드·축소 (분할을 다시 만들 수도 있다) |
| 2 | [`02_eda.ipynb`](notebooks/02_eda.ipynb) | 데이터 탐색 |
| 3 | [`03_zeroshot.ipynb`](notebooks/03_zeroshot.ipynb) | 방법 1 |
| 4 | [`04_linear_probe.ipynb`](notebooks/04_linear_probe.ipynb) | 방법 2 |
| 5 | [`05_finetune.ipynb`](notebooks/05_finetune.ipynb) | 방법 3 |
| 6 | [`06_analysis.ipynb`](notebooks/06_analysis.ipynb) | 결과 분석 |

본 저장소에는 원본 이미지와 모델 가중치를 포함하지 않는다. 게시된 예시 이미지의 출처는 OSV-5M (CC-BY-SA-4.0)이다.
