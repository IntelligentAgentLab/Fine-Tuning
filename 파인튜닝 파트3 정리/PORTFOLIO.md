# 도메인 특화 사내 AI 구축기: 파인튜닝 → RAG → 결합 → 추론 강화

> 한국 공공 규정 문서(국가공무원 복무규정)를 도메인으로, 로컬 환경(Mac Studio)에서
> LLM 파인튜닝 → RAG → 두 방식의 결합 → 추론 강화 후학습(CoT 증류ㆍDPO)까지
> 전 과정을 직접 구축하고 실험한 기록.
> 모든 실험은 "하나만 바꾸고 재측정"하는 방식으로 진행했으며, 실패와 그 원인 분석을 포함한다.

## 요약 (TL;DR)

| 단계 | 무엇을 | 핵심 결과 |
|---|---|---|
| 1. 파인튜닝 | 규정 Q&A 합성 데이터로 LoRA SFT | 키워드 적중률 12.5% → **75%**, 중국어 유출 2건 → **0건**, 근거 없는 질문 거절 0/3 → **3/3** |
| 2. RAG | 조문 임베딩 검색 + 근거 주입 생성 | 검색 hit@3 90.5% → 청킹 재설계로 **93.5%** |
| 3. 결합 | "질문+검색 근거" 형식의 행동 SFT | 근거 인용ㆍ거절ㆍ중의성 인식 행동 안착. **SFT로 가르칠 수 있는 것의 경계(행동 vs 추론)를 실험으로 확인** |
| 4. 추론 강화 | 코드 검증 CoT 증류 → 벤치마크 → DPO | **학습에 없던 숫자 조합 홀드아웃 전 문항 정답** (추론 절차 이식), KMMLU 역행 없음, DPO로 풀이 형식 5/6 → **6/6** |

**기술 스택**: Apple Silicon(M3 Ultra 256GB) / mlx-lm(LoRA) / mlx-tune(DPO) / Qwen2.5-7B-Instruct-4bit / sentence-transformers(bge-m3) / Pydantic / numpy

**핵심 교훈 다섯 가지**
1. 파인튜닝의 성패는 모델이나 학습량이 아니라 **데이터 설계**가 결정한다 (동일 조건에서 데이터만 바꿔 0/4 → 4/4)
2. **사실은 RAG가, 행동은 파인튜닝이** - 실무 설계 원칙을 3자 비교 실험으로 재현
3. loss는 나침반이지 심판이 아니다 - **최종 판정은 항상 행동 평가로**
4. 시연 SFT는 행동을 심고, **추론은 풀이 궤적(CoT)을 시연해야 심긴다** - 문장 틀만 흉내내던 모델이 절차를 수행하게 되는 전환을 관측
5. **양자화 반올림은 파인튜닝의 성과를 지울 수 있다** - 4bit 재양자화 병합에서 학습된 행동이 소실되는 사건을 3중 비교로 격리ㆍ해결

---

## 0. 문제 정의

출발점은 실패 경험이다. Q/A 두 컬럼짜리 데이터를 Qwen Instruct 모델에 그대로 학습시켰더니
답변에 중국어가 섞여 나오고, 학습시킨 내용은 나오지 않았다. 에포크를 바꿔가며 재시도해도 원인을 알 수 없었다.

사후 진단으로 밝혀진 원인 세 가지:

| 문제 | 결과 |
|---|---|
| 챗 템플릿 없이 raw 텍스트로 학습 | 학습한 형식 ≠ 질문받는 형식 → 학습 내용이 발현되지 않음 |
| 검증셋 없이 에포크만 조정 | 과적합 시작 시점을 관측할 수단이 없음 |
| 잘못된 형식으로 반복 학습 | 정렬(대화 훈련) 층이 깎임 → 사전학습의 중국어 분포가 노출 (파국적 망각) |

이 프로젝트는 위 세 문제를 구조적으로 해결하는 파이프라인을 만들고,
"조직 문서에 근거해 답하는 사내 AI"를 로컬에서 끝까지 구축하는 것을 목표로 한다.

---

## 1단계. 파인튜닝: 데이터 파이프라인과 3번의 실험

### 1.1 파이프라인 구조

```
원문 문서(법령 웹 복사본)
  → parse_doc.py      조문 단위 파싱 (노이즈 제거, 부칙 절단)
  → (Q&A 생성)        조문별 질문/답변 작성 - 생성자는 LLM
  → verify_qa.py      근거 검수: 답변의 모든 숫자가 원문에 존재하는지 대조 (환각 차단)
  → build_dataset.py  혼합: 도메인 Q&A + 거절 예시 + 일반 대화(replay)
  → validate_data.py  Pydantic 검증: 길이ㆍ한글 비율ㆍ한자 오염ㆍ중복 필터
  → prepare_data.py   챗 템플릿 변환 + train/valid/holdout 3분할
  → train.sh          mlx-lm LoRA 학습 (계기판: 25스텝마다 val loss)
  → evaluate.py       원본 모델과 동일 문항 비교 평가
```

### 1.2 핵심 코드

**챗 템플릿 변환** - 지난 실패와의 차이가 사실상 이 함수 하나에 담겨 있다.
채팅 앱의 입력은 항상 이 구조로 모델에 들어가므로, 학습도 같은 구조여야 한다:

```python
def to_chat(row, system_prompt):
    return {"messages": [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": row["question"]},
        {"role": "assistant", "content": row["answer"]},
    ]}
# mlx-lm이 학습 시 이를 Qwen 고유 템플릿(<|im_start|>...<|im_end|>)으로 렌더링
```

**근거 검수** - 합성 데이터의 환각을 기계적으로 차단한다.
규정 데이터에서 가장 위험한 것은 숫자 환각이므로, 답변의 모든 숫자를 원문과 대조:

```python
missing_nums = set(re.findall(r"\d+", answer)) - set(re.findall(r"\d+", source_text))
if missing_nums:
    reject(f"근거에 없는 숫자 포함: {missing_nums}")   # 예: 원문 "최대 21일" ↔ 답변 "25일" → 탈락
# + 답변 내용어의 40% 이상이 원문에 존재해야 통과 (무관한 창작 차단)
```

**언어 오염 필터** (Pydantic validator) - 유니코드 범위로 한글/한자 비율을 계산해
한글 30% 미만 또는 한자ㆍ가나 5% 초과 답변을 학습 전에 제거:

```python
@field_validator("answer")
def language_check(cls, v):
    hangul, cjk_other = char_stats(v)          # 유니코드 블록 기반 비율
    if hangul < 0.3:      raise ValueError("한글 비율 미달")
    if cjk_other > 0.05:  raise ValueError("한자/가나 비율 초과 - 오염 의심")
```

**LoRA 설정** (mlx-lm) - 76억 파라미터는 얼리고 0.15%(11.5M)의 어댑터만 학습.
`출력 = W@x + scale*(B@(A@x))` 구조라, 어댑터를 빼면 원본이 그대로 복원된다:

```yaml
model: mlx-community/Qwen2.5-7B-Instruct-4bit
fine_tune_type: lora
num_layers: 16          # 뒤쪽 16개 레이어에만 부착
batch_size: 4
iters: 200              # 에포크는 iters × batch ÷ 데이터 수로 역산 (~3 에포크)
learning_rate: 1.0e-05
steps_per_eval: 25      # 25스텝마다 val loss = 과적합 계기판
save_every: 25          # 체크포인트 = 롤백 가능한 타임머신
lora_parameters: {rank: 8, scale: 20.0}
```

### 1.3 실험 서사: run1 → run3

**run1 - 과적합을 계기판으로 목격.** 합성 리허설 데이터 93건을 의도적으로 600스텝(26 에포크) 과다 학습:

| Iter | Train loss | Val loss |
|---|---|---|
| 100 | 0.038 | **0.044** ← 학습 완료 시점 |
| 600 | 0.026 | 0.049 ← train↓ val↑ = 암기 구간 |

행동 검증에서 질문 표현이 한 단어만 달라져도 답변이 무너지고(표면 암기),
일반 질문에서 중국어가 재출현(망각 시작). 처방 3종 도출: 거절 예시, 일반 대화 혼합(replay), 질문 표현 다양화.

**run2 - 스타일만 배우고 사실은 못 배움.** 복무규정 합성 Q&A 93건 + 거절 4% + replay 8%, 150스텝:
- "복무규정 제N조에 따르면..." 인용 말투는 완벽 습득했으나, 조문 번호와 내용은 전부 창작 (표준 문항 0/4)
- train loss 0.063(사실상 암기)인데도 표현이 바뀐 질문에서 외운 조각을 뒤섞음
- **진단: 문제는 학습량이 아니라 데이터의 사실 밀도.** 사실 하나당 1~2개 표현으로는 지식이 형성되지 않는다

**run3 - 사실 단위 데이터 증강.** 모델ㆍ학습량 동일, 데이터만 변경:
연가 표를 "재직 연차 12종 × 질문 표현 6종"으로 프로그램 전개, 거절 13%, replay 15%:

```python
for y in [1, 2, 3, ..., 20]:                 # 사실(표의 구간)을 다양한 상황으로 전개
    band, days = yeonga_band(y)
    for tpl in YEAR_QUESTIONS:               # "7년째인데...", "입사 N년차..." 등
        yield {"question": tpl.format(y=y),
               "answer": f"...재직기간 {band}인 공무원의 연가 일수는 {days}일입니다."}
```

결과: 표준 문항 **4/4**. 정식 평가(원본 vs run3, 11문항)에서 키워드 적중률 12.5→75%, 중국어 오염 2→0건, 거절 0/3→3/3.

### 1.4 1단계에서 확인한 것

- val loss 최저점 ≠ 품질 최적점 (체크포인트별 행동 비교로 확인) → 최종 판정은 행동 평가로
- 환각은 그럴듯한 얼굴로 온다 - 존재하지 않는 단어("라일리아스")를 자연스럽게 읽고 넘길 뻔한 경험 이후,
  평가를 키워드ㆍ숫자 대조 같은 기계적 절차에 맡기도록 설계
- 남은 결함(조문 번호 오인, 표 구간 혼동)은 암기 의존의 한계 → 2단계의 동기

---

## 2단계. RAG: 검색의 품질은 청킹이 결정한다

### 2.1 구조와 핵심 코드

46개 규모에서는 벡터 DB 없이 numpy로 검색의 본질을 드러냈다:

```python
# 인덱스 구축 (embed_chunks.py)
vectors = model.encode(chunk_texts, normalize_embeddings=True)   # bge-m3, 1024차원

# 검색 (rag_search.py) - 정규화된 벡터의 코사인 유사도 = 내적
scores = vectors @ model.encode([question], normalize_embeddings=True)[0]
top3 = np.argsort(scores)[::-1][:3]

# 생성 (rag_chat.py) - 근거를 프롬프트에 주입
prompt = f"[근거]\n{top3_조문들}\n\n[질문]\n{question}"
# 시스템 프롬프트: "근거의 내용만 사용. 없으면 '확인되지 않는다'고 답하라. 조문 번호를 표기하라."
```

### 2.2 검색 평가와 청킹 실험

합성 Q&A에 달아둔 근거 라벨(chunk_id)을 정답지로 재활용해 201개 질문으로 hit@k를 자동 채점.
실패 분석 결과, 실패의 절반이 여러 주제가 나열된 긴 조문(공가 14개 사유, 특별휴가 20개 항)에 집중
→ **임베딩은 조각 전체를 하나의 좌표로 평균내므로, 잡화점식 조각은 어떤 질문과도 어중간하게 멀어진다.**

처방: 700자 초과 조문을 항(①)ㆍ호(1.) 단위로 재분할하되, 머리 문장을 각 조각에 붙여 맥락 보존:

```python
# "8. 헌혈에 참가할 때" 만으로는 의미 불충분 →
# "행정기관의 장은 ... 공가로 승인해야 한다\n8. 헌혈에 참가할 때" 로 임베딩
sub_text = lead_sentence + "\n" + sub_part
```

| 지표 | 개선 전 | 개선 후 |
|---|---|---|
| hit@1 | 77.6% | 82.1% |
| hit@3 (실전 지표) | 90.5% | **93.5%** |
| hit@5 | 92.0% | 96.0% |

변인 하나(청킹), 같은 질문 201개 → 개선폭이 온전히 청킹의 효과.

### 2.3 3자 비교: 결합의 필요성 증명

같은 질문ㆍ같은 근거로 세 구성을 비교:

| | 사실 정확성 | 조문 인용 | 거절 판단 | 형식 일관성 |
|---|---|---|---|---|
| 파인튜닝 단독 | 표 구간 혼동 | 가끔 착각 | ✓ | ✓ |
| 원본 + RAG | 원문 기반 ↑ | ✓ | **자기모순** ("...17일입니다. 제공된 규정에서 확인되지 않습니다.") | ✗ |
| **파인튜닝 + RAG** | 원문 기반 ↑ | ✓ | ✓ | ✓ |

부수 발견: 평가 정답지 자체의 오류를 발견 ("4년차"는 만 3~4년/만 4년의 중의적 표현인데
정답지가 한쪽 해석을 임의 선택) → **평가의 품질은 라벨의 품질을 넘을 수 없다.**

---

## 3단계. 결합: 근거를 다루는 행동을 가르치다

### 3.1 설계 원칙: 학습 형식 = 추론 형식

추론 때 입력이 "[근거]+[질문]"이므로 학습 데이터도 같은 형식으로 구성.
근거는 **실제 검색기 top-3**로 만들어 방해 조각이 섞인 실전 소음 환경을 학습시킴:

```python
{"messages": [
  {"role": "system", "content": RAG_SYSTEM},                    # 추론 프롬프트와 동일
  {"role": "user", "content": "[근거]\n(검색된 조문들)\n\n[질문]\n..."},
  {"role": "assistant", "content": "(행동 시연)"}]}
# 행동 4종: 근거 인용 / 근거 밖 거절 / "N년차" 중의성 해석 명시 / 안정성 replay
```

### 3.2 run4 → run4b: 행동에도 밀도와 일관성이 필요하다

- **run4 (실패)**: 중의성 경우 구분이 발현되지 않음. 원인은 행동 충돌 -
  "N년차 → 즉답" 예시 수십 건 vs 경우구분 예시 15건의 다수결
- **run4b (수정)**: 년차류 표현을 즉답 예시에서 전면 제외(정책 일관화) + 경우구분 54건 증량
  → 경우 구분 **발현**. 단, 문장 틀은 정확하나 수치가 한 칸씩 밀림

### 3.3 핵심 발견: SFT로 가르칠 수 있는 것의 경계

| 가르치려던 것 | 성격 | 결과 |
|---|---|---|
| 조문 인용, 거절, 형식 일관성 | 행동 | ✓ 밀도ㆍ일관성을 갖추면 SFT로 안착 |
| "해석이 갈린다"는 인식 | 행동 | ✓ 발현 |
| N년차 → 표에서 두 구간을 찾아 일수 대응 | **추론** | ✗ 문장 틀만 배우고 산술이 어긋남 |

시연을 찍어내는 SFT는 행동을 심지만 추론은 심지 못한다.
→ 다음 단계(추론 강화: 풀이 과정 CoT 데이터 증류)의 필요성이 실험으로 도출됨.

---

## 4단계. 추론 강화: CoT 증류 → 벤치마크 → DPO

빅테크의 추론 강화 후학습 코스(풀이 데이터 증류 → 외부 시험 → 선호학습)를 소규모로 재현했다.

### 4.1 CoT 증류 (run5): 결론이 아니라 풀이 과정을 시연하다

3단계의 결론은 "산술이 어긋난다"였다. 처방은 답이 아니라 **푸는 절차**를 데이터로 시연하는 것:

```python
# 풀이 궤적을 사람이 쓰지 않고 코드로 생성 - 코드가 곧 외부 검증기 (환각 원천 차단)
band, days = yeonga_band(y + m / 12)          # 규정의 표를 함수로 구현
answer = (f"[풀이]\n1. 질문의 재직기간은 만 {y}년 {m}개월이다.\n"
          f"2. 근거 제15조의 표에서 이는 '{band}' 구간에 해당한다.\n"
          f"3. 해당 구간의 연가 일수는 {days}일이다.\n"
          f"[답변] ... {days}일입니다.")
```

절차 4종(구간 판정, 중의성 경우 구분, 문턱 비교, 누계 나눗셈)을 [풀이]→[답변] 형식으로 시연.
**실험 설계의 핵심은 숫자 홀드아웃**: 학습과 평가에 서로 다른 숫자 조합을 배정해,
정답을 외웠는지("4년 7개월"은 학습에 없음) 절차를 배웠는지를 가르게 했다.

결과: **홀드아웃 전 문항 정답** - 처음 보는 숫자 조합에서 구간 판정ㆍ경우 구분ㆍ문턱 비교가 모두 수행됨.
run4b에서 "문장 틀만 흉내내던" 모델이 절차를 수행하는 모델로 전환. → 증류된 것은 답이 아니라 **절차**다.

### 4.2 KMMLU 벤치마크: 역행 없음의 확인

우리 규정 문답은 우리가 만든 시험이다. 제3자 시험(KMMLU 3과목 150문항, zero-shot)으로 일반 능력을 검사:

| 모델 | 정확도 |
|---|---|
| 원본 Qwen2.5-7B-4bit | 31.3% |
| run5 (누적 5회 학습 후) | 34.7% |

읽는 법이 중요하다. +3.4pp는 150문항 중 5문항 차이 = **잡음 범위**. 결론은 "향상"이 아니라 **역행 없음** -
replay 혼합을 유지한 누적 학습이 일반 능력을 깎지 않았다는 파국적 망각 방어의 검증이다.

### 4.3 DPO (run6): 관측된 실패를 선호 쌍으로 되갚다

SFT는 좋은 예시만 보여주지만 DPO는 (선호, 비선호) 쌍으로 "이렇게 하지 마라"의 방향을 가르친다.
**rejected는 창작이 아니라 run1~run5에서 실제 관측한 실패 유형 4종의 재현**:
풀이 생략 / 인접 구간 한 칸 밀림 / 자기모순(답변+거절 문구) / 잡탕 인용. 총 23쌍, lr 5e-7, 60스텝.

| 홀드아웃 6문항 | run5 (기준선) | run6 (DPO) |
|---|---|---|
| 정답 | 6/6 | 6/6 |
| [풀이] 형식 | 5/6 (1건 생략) | **6/6** |

- 겨냥한 "풀이 생략"이 정확히 그 문항에서 교정되고 나머지는 무손상 보존 → **파괴 없는 미세 개선**
- 못 고친 것도 명확: 학습 쌍에 없는 숫자의 궤적 결함은 그대로 → **DPO는 보여준 선호의 방향으로만 민다**

### 4.4 부수 사건: 양자화가 학습을 지운다

DPO를 위해 run5 어댑터를 베이스에 병합(fuse)하자 행동이 무너졌다(풀이 형식 소실, 영어 답변, 거절 오답).
어댑터 원본 / 4bit 병합본 / 16bit 병합본을 같은 6문항으로 3중 비교해 원인을 격리:

- 어댑터 원본 정상, 병합본만 손상 → 병합 과정이 범인
- 원인: 4bit 재양자화 시 LoRA의 미세한 가중치 변화가 반올림에 소실. **행동처럼 섬세한 조정일수록 먼저 죽는다**
- 해결: `--dequantize`로 16bit 병합 → 행동 완전 복원. DPO 결과 저장도 같은 이유로 16bit 병합 사용

---

## 재현 가이드

```bash
# 환경 (Apple Silicon)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # mlx-lm, pydantic, pyyaml, datasets, pandas, sentence-transformers

# 1단계: 데이터 → 학습 → 평가
python scripts/parse_doc.py --input data/source_docs/복무규정.txt
#   (조문별 Q&A 생성: LLM으로 생성 후 아래 검수를 통과시킬 것)
python scripts/verify_qa.py --qa data/synth/qa_generated.jsonl
python scripts/make_augmented_qa.py && python scripts/verify_qa.py --qa data/synth/qa_augmented.jsonl --output data/synth/qa_augmented_verified.jsonl
python scripts/build_dataset.py && python scripts/validate_data.py && python scripts/prepare_data.py
bash scripts/train.sh                  # 로그의 val loss 추이를 반드시 관찰
python scripts/evaluate.py --base && python scripts/evaluate.py   # 전후 비교 리포트

# 2단계: RAG
python scripts/embed_chunks.py         # 임베딩 인덱스
python scripts/eval_retrieval.py       # hit@k (90% 미만이면 생성 전에 검색부터 수리)
python scripts/rag_chat.py -q "질문" --show-context

# 3단계: 결합
python scripts/build_rag_dataset.py    # 실제 검색 근거로 행동 데이터 빌드
bash scripts/train_rag.sh
python scripts/rag_chat.py --adapter adapters/run4b -q "질문"

# 4단계: 추론 강화
python scripts/build_cot_dataset.py    # 코드 검증 CoT 생성 (숫자 홀드아웃 분리)
bash scripts/train_rag.sh              # → adapters/run5
python scripts/eval_benchmark.py --base && python scripts/eval_benchmark.py   # KMMLU 전후 비교
mlx_lm fuse --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --adapter-path adapters/run5 --save-path fused_run5 --dequantize   # 반드시 --dequantize (16bit)
python scripts/build_dpo_pairs.py      # 관측 실패 유형 4종 → 선호 쌍 23건
python scripts/train_dpo.py            # mlx-tune DPO → fused_run6 (16bit 병합 저장)
python scripts/rag_chat.py --model fused_run6 --base -q "질문"
```

## 트러블슈팅에서 배운 것 (발췌)

- 프레임워크의 암묵적 파일명 관례 (mlx-lm이 test.jsonl을 학습 데이터로 읽음) → traceback은 맨 아래부터
- `unzip -o`는 잉여 파일을 지우지 않는다 → 생성 스크립트가 구버전 산출물을 자동 정리
- 같은 시드ㆍ같은 데이터의 완전 동일한 loss → 재현성이 곧 "지금 어느 버전을 돌리는가"의 판별 근거
- 실행 환경별 네트워크 차이 → 로드 실패 시 죽지 않고 건너뛰는 방어적 파이프라인
- 작업공간은 휘발될 수 있다 → 산출물은 생성 즉시 별도 저장소로 (git의 존재 이유)
- 라이브러리 경계를 넘는 산출물은 표준 형식으로 - mlx-tune 어댑터를 mlx-lm이 못 읽음 → HF 표준 병합 저장으로 우회
- 병합ㆍ변환ㆍ양자화 등 모델을 만지는 모든 단계 후에는 행동 평가를 다시 돌려라 - 파일이 생겼다고 성공이 아니다

## 다음 단계

- **자기개선 루프**: 모델이 생성한 답을 검증기로 걸러 다음 학습 재료로 (rejection sampling / STaR의 소규모 재현)
- 확장 카드: AI Hub 데이터 병합, 한국어 특화 모델(Kanana) 비교, 하이브리드 검색ㆍreranker, 질문 라우팅, 궤적 교정 쌍 추가 DPO
