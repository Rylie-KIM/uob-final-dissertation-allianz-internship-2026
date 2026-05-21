# Dataset Guide — SFP Loop Research
*모든 데이터셋의 구조, 컬럼 설명, 통계, 사용법*

---

## 전체 파일 목록

```
datasets/
├── synthetic/                           ← 직접 생성한 합성 보험 청구 데이터
│   ├── claims_small.csv                 3,000행 × 11컬럼   (166 KB)
│   ├── claims_medium.csv               10,000행 × 11컬럼   (554 KB)
│   ├── claims_large.csv                50,000행 × 11컬럼  (2.7 MB)
│   ├── claims_with_versions.csv        10,000행 × 22컬럼  (1.3 MB)  ← 핵심
│   ├── claims_with_versions_eps.csv    10,000행 × 22컬럼  (1.3 MB)  ← ε=0.10 탐색
│   ├── loop_metrics.csv                     3행 × 9컬럼   (0.3 KB)
│   ├── loop_metrics_eps.csv                 3행 × 9컬럼   (0.3 KB)
│   └── README.txt                                          (3.5 KB)
│
└── real/                               ← 실제 보험 데이터 (공개 데이터셋)
    ├── coil2000.csv                    9,822행 × 86컬럼   (410 KB)  ← 원본
    ├── coil2000_with_sfp_loop.csv      9,822행 × 96컬럼   (598 KB)  ← SFP 추가
    └── porto_seguro_sample.csv        10,000행 × 39컬럼  (1.1 MB)
```

---

## 1. 합성 데이터 — `claims_small/medium/large.csv`

### 용도
빠른 단위 테스트 및 기능 검증용 기반 데이터.
모델 버전 정보 없이 **청구 특성 + 실제 사기 레이블만** 포함.

### 데이터 생성 원리 (Data Generating Process)

사기 확률은 다음 로지스틱 모델로 생성됨:

```
logit(P(사기)) = -2.5
               + 0.8 × high_amount       (대규모 청구 → 사기 위험↑)
               + 0.6 × night_claim       (야간 접수 → 사기 위험↑)
               + 1.2 × high_postcode     (고위험 우편번호 → 사기 위험↑)
               + 0.4 × prior_claims      (과거 청구 횟수 → 사기 위험↑)
               + N(0, 0.3)              (개인별 노이즈)
```

**실제 UK 모터 보험 사기율 5–12%에 맞춰 기저율 약 17% 설계.**

### 스키마

| 컬럼 | 타입 | 설명 | 예시 |
|------|------|------|------|
| `claim_id` | str | 고유 청구 ID (CLM000001 형식) | `CLM000000` |
| `claim_amount` | float | 청구 금액(GBP), 로그정규분포 (μ=7.5) | `2606.23` |
| `claim_hour` | int | 청구 접수 시각 (0–23) | `12` |
| `postcode_risk` | float | 우편번호별 사기 위험 지수 (0=저위험, 1=고위험) | `0.1146` |
| `prior_claims` | int | 동일 청구인의 과거 청구 건수 (0–5) | `0` |
| `claim_type` | str | 청구 유형: `theft`, `damage`, `injury` | `injury` |
| `high_amount` | int | 청구액 > 75th 백분위이면 1, 아니면 0 | `0` |
| `night_claim` | int | 접수 시각이 22시–5시이면 1, 아니면 0 | `0` |
| `high_postcode` | int | postcode_risk > 0.6이면 1, 아니면 0 | `0` |
| `true_fraud` | int | **실제 사기 여부 (Oracle)** — 0 또는 1 | `0` |
| `true_fraud_prob` | float | 실제 사기 확률 (Oracle) | `0.0988` |

### 핵심 통계

| 파일 | 행 수 | 사기율(true) | 결측값 |
|------|-------|-------------|--------|
| claims_small.csv | 3,000 | 17.7% | 없음 |
| claims_medium.csv | 10,000 | 17.2% | 없음 |
| claims_large.csv | 50,000 | 16.5% | 없음 |

### 샘플 행

```
claim_id   claim_amount  claim_hour  postcode_risk  prior_claims  claim_type  high_amount  night_claim  high_postcode  true_fraud  true_fraud_prob
CLM000000  2606.23       12          0.1146         0             injury      0            0            0              0           0.0988
CLM000001  519.06        23          0.1832         0             theft       0            1            0              0           0.1210
```

---

## 2. SFP Loop 시뮬레이션 데이터 — `claims_with_versions.csv`

> **이 프로젝트의 핵심 데이터셋.** 3세대 모델 버전과 SFP loop 효과가 모두 포함됨.

### 용도
- Loop Detection (Build 02)
- Unbiased Evaluation (Build 03)
- Intervention Analysis (Build 04)
- Causal Mitigation (Build 06)

### Loop 생성 메커니즘

```
[생성 과정]
1. 10,000건 청구 생성 (진짜 사기 레이블 = oracle)
2. 무작위 5% 시드 배치로 Model v1 훈련 (무편향)
3. v1 점수 상위 25% 청구만 조사 → 조사된 곳에서만 사기 발견
4. 편향된 레이블로 Model v2 훈련
5. v2로 다시 상위 25% 조사 → Model v3 훈련
6. 각 세대의 bias가 누적되어 blind spot 증가
```

### 스키마 (22컬럼)

**[A] 기본 청구 특성 (11컬럼) — claims_small.csv와 동일**

**[B] 모델 버전별 컬럼 (각 버전 3컬럼 × 3버전 = 9컬럼)**

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `model_v{N}_score` | float | 모델 버전 N의 사기 위험 점수 (0–1) |
| `model_v{N}_investigated` | int | 버전 N 정책에 의해 조사됐으면 1, 아니면 0 |
| `model_v{N}_observed_fraud` | float | **조사된 경우** 사기 발견 1/0, **미조사는 NaN** |

**[C] 편의 컬럼 (2컬럼) — 현재 배포 모델(v3) 기준**

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `investigated` | int | `= model_v3_investigated` |
| `observed_fraud` | float | `= model_v3_observed_fraud` (미조사 = NaN) |

### 핵심 통계 — SFP loop 신호

| 지표 | ε=0 (loop 활성) | ε=0.10 (탐색 적용) |
|------|-----------------|-------------------|
| 진짜 사기율 (ground truth) | **17.2%** | **17.2%** |
| 관찰된 사기율 (편향) | **30.5%** | 26.5% |
| 조사율 | 27.6% | 35.3% |
| SFP 증폭 배율 | **1.81×** | 1.80× |
| Blind spot (탐지 불가 사기) | **50.0%** | 44.4% ✓ 개선 |

> **핵심:** 관찰된 사기율(30.5%)이 진짜 사기율(17.2%)의 **1.81배** — 이것이 SFP loop의 증거.

### 결측값 패턴 — 의도적 구조

```
model_v1_observed_fraud: 7,108개 NaN (71.1%) ← v1이 미조사한 청구
model_v2_observed_fraud: 7,242개 NaN (72.4%) ← v2가 미조사한 청구
model_v3_observed_fraud: 7,242개 NaN (72.4%) ← v3가 미조사한 청구
observed_fraud:          7,242개 NaN (72.4%) ← = v3 기준
```

> **이 NaN은 오류가 아님.** 현실을 반영 — 조사하지 않은 청구의 사기 여부는 알 수 없음. IPS 보정이 필요한 이유.

### 샘플 행

```
claim_id   true_fraud  model_v1_score  model_v1_investigated  model_v1_observed_fraud  model_v2_score  model_v2_investigated  ...  investigated  observed_fraud
CLM000000  0           0.3167          0                      NaN                      0.1340          0                      ...  0             NaN
CLM000003  1           0.7821          1                      1.0                      0.8234          1                      ...  1             1.0
CLM000007  0           0.8102          1                      0.0                      0.7956          1                      ...  1             0.0
```

---

## 3. Loop 지표 요약 — `loop_metrics.csv`

### 용도
세대별 loop 심화 효과를 한눈에 파악. 논문/발표 자료의 핵심 표.

### 스키마 (9컬럼)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `version` | int | 모델 버전 번호 (1, 2, 3) |
| `n_investigated` | int | 해당 세대에서 조사된 청구 수 |
| `investigation_rate` | float | 전체 대비 조사 비율 |
| `observed_fraud_rate` | float | 조사된 청구 내 관찰 사기율 **(편향됨)** |
| `auc_on_investigated` | float | 조사 청구 기준 AUC **(편향됨 → 높게 보임)** |
| `true_recall` | float | 전체 진짜 사기 중 발견 비율 **(실제 성능)** |
| `precision` | float | 조사 청구 내 정밀도 |
| `loop_amplification_factor` | float | 상위 25% 사기율 / 전체 진짜 사기율 (>1 = loop 활성) |
| `blind_spot_fraud_fraction` | float | 미조사 청구에 숨어있는 진짜 사기 비율 |

### 실제 데이터 값

**ε=0 (loop 활성, loop_metrics.csv)**

| version | n_investigated | observed_fraud_rate | auc_on_investigated | true_recall | loop_amplification | blind_spot |
|---------|---------------|--------------------|--------------------|-------------|-------------------|------------|
| 1 | 2,892 | 0.274 | 0.641 | **0.465** | 1.61× | 53.5% |
| 2 | 2,758 | 0.304 | 0.599 | **0.500** | 1.81× | 50.0% |
| 3 | 2,758 | 0.305 | 0.607 | **0.500** | 1.81× | 50.0% |

**ε=0.10 (탐색 적용, loop_metrics_eps.csv)**

| version | n_investigated | observed_fraud_rate | auc_on_investigated | true_recall | loop_amplification | blind_spot |
|---------|---------------|--------------------|--------------------|-------------|-------------------|------------|
| 1 | 3,607 | 0.247 | 0.656 | **0.524** | 1.61× | 47.6% |
| 2 | 3,442 | 0.269 | 0.650 | **0.551** | 1.81× | 44.9% |
| 3 | 3,530 | 0.265 | 0.654 | **0.556** | 1.80× | 44.4% ✓** |

> **해석:** ε=0.10 적용 시 v3 true_recall이 0.500 → 0.556으로 +5.6%p 개선, blind spot 50.0% → 44.4%로 감소.

---

## 4. COIL 2000 원본 — `coil2000.csv`

### 출처
- **원본:** Dutch insurance company benchmark data (CoIL Challenge 2000)
- **다운로드:** UCI ML Repository — https://archive.ics.uci.edu/dataset/125
- **라이선스:** 공개 학술용

### 용도
실제 보험 고객 데이터. SFP loop 추가 전 원본.

### 스키마 (86컬럼)

**[M 컬럼] 사회통계 특성 (43컬럼)**

| 컬럼 접두사 | 의미 | 예시 컬럼 |
|-----------|------|----------|
| `MOSTYPE` | 고객 주요 유형 (1–41) | `MOSTYPE=33` → 고소득 가족 |
| `MGEMOMV` | 평균 가구원 수 (1–6) | |
| `MOPLHOOG` | 고학력 비율 (0–9) | |
| `MINKM30`~`MINK123M` | 소득 구간별 비율 (0–9) | |
| `MKOOPKLA` | 구매력 등급 (1–8) | |
| `MAUT1/2/0` | 차량 보유 비율 | |

**[P 컬럼] 보험 기여도/정책 수 (18컬럼) — 가장 중요한 예측 변수**

| 컬럼 | 의미 |
|------|------|
| `PPERSAUT` | 자동차 보험 가입 비율 (0–9) |
| `PBESAUT` | 배달 밴 보험 |
| `PMOTSCO` | 오토바이/스쿠터 보험 |
| `PBRAND` | 화재 보험 |
| `PLEVEN` | 생명 보험 |
| `PPERSONG` | 상해 보험 |
| `PGEZONG` | 건강 보험 |
| `PFIETS` | 자전거 보험 |

**[A 컬럼] 실제 보험 가입 건수 (18컬럼)**

P 컬럼의 쌍. `PPERSAUT`(비율) vs `APERSAUT`(실제 건수).

**[TARGET]**

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `CARAVAN` | int | 캐러밴 보험 구매 여부 (1=구매, 0=미구매) |

### 핵심 통계

| 항목 | 값 |
|------|-----|
| 총 행 수 | 9,822 |
| 총 컬럼 수 | 86 |
| CARAVAN 구매율 (target) | **6.0%** (586명 / 9,822명) |
| 결측값 | **없음** |
| 모든 값 범위 | 정수 0–9 (P, M, A 컬럼) |

### 샘플 행

```
MOSTYPE  MAANTHUI  MGEMOMV  ...  PPERSAUT  PBRAND  PLEVEN  ...  CARAVAN
33       1         3        ...  6         5       0       ...  0
37       1         2        ...  0         2       0       ...  0
```

---

## 5. COIL 2000 + SFP Loop — `coil2000_with_sfp_loop.csv`

### 용도
실제 보험 고객 특성 위에 사기 조사 SFP loop를 시뮬레이션한 데이터.
**실제 보험 피처 + 현실적인 loop 신호** — 인터뷰 데모용 최적 데이터.

### 추가된 컬럼 (10컬럼)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `true_fraud` | int | 시뮬레이션된 사기 레이블 (Oracle) |
| `true_fraud_prob` | float | 사기 확률 (Oracle) |
| `model_v1_score` | float | v1 모델 예측 점수 (0–1) |
| `model_v2_score` | float | v2 모델 예측 점수 — v1 편향 레이블로 훈련됨 |
| `model_v1_investigated` | int | v1 정책 조사 여부 |
| `model_v2_investigated` | int | v2 정책 조사 여부 |
| `model_v1_observed_fraud` | float | v1 조사 결과 (미조사=NaN) |
| `model_v2_observed_fraud` | float | v2 조사 결과 (미조사=NaN) |
| `investigated` | int | 현재(v2) 조사 여부 — 편의 컬럼 |
| `observed_fraud` | float | 현재 관찰 사기 레이블 — 편의 컬럼 |

### 사기 생성 공식 (P 컬럼 기반)

```
logit(P(사기)) = -2.5
               + 0.6 × PPERSAUT (자동차 보험 건수 — 위험도↑)
               + 0.5 × PBRAND   (화재 보험 건수)
               + 0.4 × PLEVEN   (생명 보험 건수)
               + N(0, 0.8)
```

### 핵심 통계

| 지표 | 값 | 의미 |
|------|----|------|
| 진짜 사기율 | **12.0%** | Oracle 기준 |
| 관찰된 사기율 | **25.6%** | 조사된 청구 기준 (편향) |
| SFP 증폭 배율 | **2.15×** | 합성 데이터(1.81×)보다 강한 loop |
| 조사율 | 25.0% | v2 정책 기준 |
| Blind spot 사기 | **45.7%** | 전체 사기의 절반 가까이 탐지 불가 |
| 결측값 | 7,365개 (NaN) | observed_fraud — 의도적 |

### 샘플 행 (추가 컬럼만)

```
MOSTYPE  ...  CARAVAN  true_fraud  true_fraud_prob  model_v1_score  model_v2_score  investigated  observed_fraud
33       ...  0        0           0.2926           0.6453          0.5698          1             0.0
37       ...  0        0           0.0182           0.4683          0.1253          0             NaN
```

---

## 6. Porto Seguro 샘플 — `porto_seguro_sample.csv`

### 출처
- **원본:** Kaggle Competition "Porto Seguro's Safe Driver Prediction" (2017)
- **이 파일:** 동일 스키마로 생성한 10,000행 합성 샘플 (Kaggle 로그인 불필요)
- **전체 데이터:** `kaggle competitions download -c porto-seguro-safe-driver-prediction`

### 용도
실제 자동차 보험 특성 구조 연습. **가격 책정 SFP loop** (Pillar 1) 시뮬레이션에 적합.

### 스키마 (39컬럼)

**[ps_ind] 개인/계약자 특성 (18컬럼)**

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `ps_ind_01` | int | 개인 특성 1 (0–6) |
| `ps_ind_02_cat` | int | 범주형 특성 (1–4) |
| `ps_ind_06_bin`~`ps_ind_13_bin` | int | 이진 특성 (0 또는 1) |
| `ps_ind_15` | int | 청구 관련 특성 (0–13) |

**[ps_reg] 지역 특성 (3컬럼)**

| 컬럼 | 타입 | 설명 | 보험 유사 특성 |
|------|------|------|--------------|
| `ps_reg_01` | float | 지역 위험 지수 (0.0–0.9) | 우편번호 위험 |
| `ps_reg_02` | float | 지역 특성 2 (0.0–1.8) | |
| `ps_reg_03` | float | 지역 특성 3 (0.1–4.0) | |

**[ps_car] 차량 특성 (17컬럼)**

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `ps_car_01_cat`~`ps_car_10_cat` | int | 범주형 차량 특성 (-1=결측) |
| `ps_car_11` | int | 차량 특성 (0–3) |
| `ps_car_12` | float | 차량 특성 (0.3–0.7) |
| `ps_car_13` | float | **가장 예측력 높은 특성** (0.2–3.7) |
| `ps_car_14` | float | 차량 특성 (0.0–0.7) |
| `ps_car_15` | float | 차량 특성 (0.0–3.7) |

**[TARGET]**

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `target` | int | 보험 기간 중 청구 여부 (1=청구, 0=미청구) |

### 핵심 통계

| 항목 | 값 |
|------|-----|
| 총 행 수 | 10,000 |
| target 비율 | 19.4% (이 샘플) / 실제 데이터 3.6% |
| 결측값(-1) | ps_car_03_cat, ps_car_05_cat, ps_car_07_cat |
| 모든 컬럼 수 | 39 |

---

## 데이터셋 선택 가이드

```
어떤 데이터를 써야 할까?

SFP loop 메커니즘 설명/증명
  └── claims_with_versions.csv          ← ground truth 있음, 직관적

실제 보험 데이터에서 loop 탐지
  └── coil2000_with_sfp_loop.csv        ← 실제 네덜란드 보험사 피처

탐색 정책(ε-greedy) 효과 비교
  └── claims_with_versions.csv (ε=0)    ← loop 활성
  └── claims_with_versions_eps.csv      ← ε=0.10 적용됨

단위 테스트 (빠른 실행)
  └── claims_small.csv                  ← 3,000행, 11컬럼

IPS 보정 효과 검증
  └── claims_with_versions.csv          ← observed vs true fraud rate 비교 가능

인터뷰 데모 (실제 데이터 강조)
  └── coil2000_with_sfp_loop.csv        ← "실제 네덜란드 보험사 데이터로 검증"
```

---

## 빠른 로드 예시

```python
import pandas as pd

# 합성 데이터 (핵심 — loop 활성)
df = pd.read_csv("datasets/synthetic/claims_with_versions.csv")

# Loop 지표 요약
metrics = pd.read_csv("datasets/synthetic/loop_metrics.csv")
print(metrics[["version", "true_recall", "loop_amplification_factor", "blind_spot_fraud_fraction"]])

# 실제 보험 데이터 (COIL 2000 + SFP loop)
coil = pd.read_csv("datasets/real/coil2000_with_sfp_loop.csv")

# SFP 증폭 확인
invest = coil[coil["investigated"] == 1]
print(f"True fraud rate:     {coil['true_fraud'].mean():.3f}")
print(f"Observed fraud rate: {invest['observed_fraud'].mean():.3f}")
print(f"SFP amplification:   {invest['observed_fraud'].mean() / coil['true_fraud'].mean():.2f}x")
```

**출력:**
```
   version  true_recall  loop_amplification_factor  blind_spot_fraud_fraction
0        1       0.4651                      1.608                     0.5349
1        2       0.5000                      1.813                     0.5000
2        3       0.5000                      1.813                     0.5000

True fraud rate:     0.120
Observed fraud rate: 0.256
SFP amplification:   2.15x
```

---

## 데이터 재생성 방법

```bash
# 합성 데이터 전체 재생성
python3 datasets/generate_synthetic.py

# COIL 2000 다운로드 + SFP loop 추가
python3 datasets/download_real.py

# 실제 대용량 데이터 (Kaggle 계정 필요)
# pip install kaggle
# kaggle competitions download -c porto-seguro-safe-driver-prediction
# kaggle competitions download -c ieee-fraud-detection
```
