## Llama-3.2-1B-Instruct 파인튜닝 실습

### 1. 모델 개요

| 항목 | 설명 |
|------|------|
| **모델 이름** | `Llama-3.2-1B-Instruct` |
| **파라미터 수** | 약 1.2B (12억 개) |
| **모델 타입** | Causal Language Model (GPT류) |
| **특징** | Instruct-tuned (질문/명령 잘 수행) |

### 2. 학습 데이터

| 항목 | 설명 |
|------|------|
| **데이터셋** | KorQuAD v1.0 (Korean QA Dataset) |
| **샘플 수** | 약 60,000개 |
| **구성** | 문맥(Context) + 질문(Question) → 답변(Answer) |
| **언어** | 한국어 |

### 3. 파인튜닝 방식: LoRA + 4bit 양자화
#### LoRA (Low-Rank Adaptation) 
- 원본 모델 전체를 학습하지 않고, 일부 핵심 레이어만 소량의 파라미터로 조정
  
| 기술 | 설명 |
|------|------|
| **LoRA** | 핵심 레이어만 저용량 파라미터로 학습 (메모리 절약) |
| **LoRA 적용 범위** | `["q_proj", "v_proj"]` 등 attention 계열 모듈 중심 |

#### 4bit 양자화
- 	모델 파라미터를 float16/32 대신 4bit 정밀도로 압축. GPU 메모리 부족 해결
  
| 기술 | 설명 |
|------|------|
| **4bit 양자화** | GPU 메모리 사용량 최소화 |
| **도구** | `bitsandbytes`, `prepare_model_for_kbit_training()` |

### 4. 학습 설정
| 항목 | 값 |
|------|----|
| **에폭 수** | 3~5 |
| **Batch Size** | 1 (accumulation으로 보완) |
| **학습률** | 2e-4 |
| **Optimizer** | `paged_adamw_32bit` |
| **시퀀스 길이** | 512 (적절히 조절) |
| **프롬프트 포맷** | ### Question / ### Context / ### Answer |

---------------------

- 오류 해결 참고:
https://stackoverflow.com/questions/79509805/typeerror-sfttrainer-init-got-an-unexpected-keyword-argument-dataset-tex
