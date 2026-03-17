# Graph 기반 자금세탁(AML) 탐지 모델 연구

금융 거래 데이터를 **그래프(Network) 구조로 모델링**하여 자금세탁(AML, Anti-Money Laundering) 의심 계좌를 탐지하는 모델을 연구한 프로젝트입니다.

기존 **전통적인 머신러닝 기반 AML 탐지 방식의 한계**를 극복하기 위해  
거래 관계를 활용한 **Graph Feature Engineering**과 **Graph Neural Network(GNN)** 기반 접근을 적용했습니다.

특히 **대규모 금융 거래 네트워크 데이터(약 3200만 거래)**를 기반으로  
ML → Graph Feature → GNN → Hybrid 모델까지 단계적으로 실험을 진행했습니다.

---

# 팀 소개

## Team 은결원

**팀장**

조은별

**팀원**

- 임성현
- 최효근
- 한혁

---

# 프로젝트 배경

전 세계 금융 범죄 규모는 매년 증가하고 있으며, 기존 AML 시스템은 다음과 같은 한계를 가지고 있습니다.

- 규칙 기반 탐지 방식은 **오탐(False Positive)**이 많음
- 전통적인 ML 모델은 **개별 계좌의 통계적 특성만 분석**
- **조직적 관계 기반 범죄 패턴 탐지 어려움**

따라서 본 프로젝트는 **거래 관계 구조 자체를 학습하는 Graph 기반 접근**을 통해 AML 탐지 성능을 개선하는 것을 목표로 합니다. :contentReference[oaicite:0]{index=0}

---

# 프로젝트 목표

금융 거래 데이터를 **그래프 구조로 모델링**하여 계좌 간 관계 정보를 활용하고, 이를 통해 **자금세탁 의심 계좌 탐지 성능을 개선**하는 것을 목표로 합니다.

### Task

Node Classification

각 계좌(Node)가 **자금세탁 관련 계좌인지 분류**

### 실험 단계

1️⃣ Baseline ML 모델 구축  
2️⃣ Graph Feature 기반 ML 확장  
3️⃣ Graph Neural Network 적용  
4️⃣ GNN + ML Hybrid 모델 구축

---

# 데이터셋

본 프로젝트는 **Kaggle IBM AMLWorld (HI-Medium)** 데이터셋을 기반으로 진행되었습니다. :contentReference[oaicite:1]{index=1}

### 데이터 규모

| 항목 | 규모 |
|---|---|
| 거래 데이터 | 약 32M edges |
| 계좌 데이터 | 약 2.08M nodes |
| 관측 기간 | 28일 |

### 그래프 구조

Temporal Directed Multigraph

- **Directed** : 송금 → 수신 방향 존재
- **Multi-edge** : 동일 계좌 간 반복 거래
- **Temporal** : 거래 시간 순서 존재

---

# 데이터 스키마

### Edge (거래)

- Timestamp
- From Account
- To Account
- Amount
- Currency
- Payment Format
- Is Laundering (Label)

### Node (계좌)

- Account ID
- Bank 정보
- 국가 정보
- Entity 정보

또한 **15개 국가 통화 및 Bitcoin 거래를 USD 기준으로 환산하여 통일된 거래 가중치**를 구성했습니다. :contentReference[oaicite:2]{index=2}

---

# 대규모 데이터 처리 전략

본 데이터는 **수천만 거래 규모**이기 때문에 메모리 사용량을 줄이기 위한 최적화를 적용했습니다.

### 메모리 최적화

- **Polars Lazy API 사용**
- **Parquet 포맷 활용**
- 데이터 타입 최적화
  - float64 → float32
  - int64 → int32

이를 통해 대규모 거래 데이터를 **Out-of-Memory 없이 효율적으로 처리**했습니다. :contentReference[oaicite:3]{index=3}

---

# 기술 스택

### Data Processing

- Polars
- Parquet
- Pandas

### Machine Learning

- XGBoost
- Scikit-learn

### Graph Learning

- PyTorch
- PyTorch Geometric

### Experiment Tracking

- Weights & Biases

### Infrastructure

- GCP Instance (T4 GPU)

---

# 프로젝트 구조

```
AML_Project

ML_Baseline
│
├── ML_Baseline_EDA.ipynb
└── ML_Baseline_Final.ipynb

Advanced_ML
│
├── ML_Groupby_EDA.ipynb
├── ML-Groupby-Final.ipynb
└── ML_Advanced_Final_ipynb

GNN
│
├── GATv2
│   └── GATv2-Final-1stage.ipynb
│
├── LAS_GNN
│   └── las_gnn_1stage_Final.ipynb
│
└── GraphSAGE

    End-to-End
    ├── GraphSAGE_v3_4_1(민민).ipynb
    ├── GraphSAGE_v3_4_2(민풀).ipynb
    └── GraphSAGE_v3_4_3(풀풀).ipynb

    Two-Stage
    ├── AML_GraphSAGE_XGB_v6 민민.ipynb
    ├── AML_GraphSAGE_XGB_v6 민풀.ipynb
    └── AML_GraphSAGE_XGB_v6 풀풀.ipynb
```

---

# 실험 1 — Baseline ML

XGBoost 기반 **기본 AML 탐지 모델**을 구축했습니다.

### 특징

- 계좌 통계 Feature 기반
- 거래 금액 및 빈도 중심 분석

### 결과

| Metric | Score |
|---|---|
| Test AUPRC | 0.603 |
| Top-3000 F1 | 0.377 |

대규모 국제 거래는 잘 탐지했지만  
**소액 분산 거래(Smurfing)** 패턴 탐지는 어려웠습니다. :contentReference[oaicite:4]{index=4}

---

# 실험 2 — Graph Feature + ML

거래 네트워크 구조를 활용한 **Graph Feature Engineering**을 적용했습니다.

### 주요 Feature

Network Centrality

- Degree
- Betweenness
- Closeness

Neighbor Risk Propagation

자금세탁 계좌와 연결된 계좌의 **위험도 전파**

AML 탐지 모듈

- Structuring
- Cycle
- Cross-border pattern
- Pass-through account

총 **16개 탐지 모듈 기반 Feature** 생성

---

# 실험 3 — Graph Neural Network

그래프 구조를 직접 학습하기 위해 **GNN 모델**을 적용했습니다.

### 사용 모델

GraphSAGE

- 이웃 샘플링 기반 대규모 그래프 학습

GATv2

- Attention 기반 관계 중요도 학습

LAS-GNN

- 금융 거래 특화 구조
- Edge 속성 반영
- 송금/수신 방향 분리 학습

핵심 개념

**Message Passing**

> 노드는 이웃 노드 정보를 받아 자신의 표현을 업데이트한다. :contentReference[oaicite:5]{index=5}

---

# 실험 4 — GNN + ML Hybrid (2-Stage)

GNN의 **관계 표현력**과 XGBoost의 **분류 성능**을 결합한 하이브리드 모델을 구축했습니다.

### Pipeline

```
Raw Graph
   ↓
GNN Encoder
   ↓
Node Embedding
   ↓
Statistical Features
   ↓
Feature Concatenation
   ↓
XGBoost Classifier
```

### 장점

- GNN : 관계 구조 표현
- XGBoost : 정밀한 분류

---

# 최종 결과

| 모델 | Test AUPRC |
|---|---|
| XGBoost Baseline | 0.603 |
| Graph Feature + XGB | 0.585 |
| GraphSAGE (2-stage) | 0.198 |
| GATv2 (2-stage) | 0.332 |
| **LAS-GNN (2-stage)** | **0.666** |

Baseline 대비 **AUPRC +10.4% 성능 향상**을 달성했습니다. :contentReference[oaicite:6]{index=6}

---

# 프로젝트 핵심 인사이트

1️⃣ **금액 기반 분석만으로는 AML 탐지에 한계 존재**

2️⃣ **거래 관계 구조(Graph)가 중요한 단서**

3️⃣ **GNN + ML Hybrid 모델이 가장 효과적**

---

# 향후 연구 방향

- GNN 모델 앙상블
- Self-Supervised Graph Learning
- Contrastive Learning
- On-chain 금융 데이터 결합

---

# 참고

데이터 용량이 매우 크기 때문에 **GitHub Repository에는 포함되어 있지 않습니다.**

전체 실험 과정은 각 Notebook 파일에서 확인할 수 있습니다.
