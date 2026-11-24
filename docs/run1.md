# Run 1 — OAT 예측 절제 실험 (A/B/C/D)

- 날짜: 2026-11-24 (KST)
- 실행 위치: Kaggle Notebook (`kabonzz/notebookaf13d60ffe`, version 2, CPU, Internet ON)
- 저장소 커밋: `13e3b4e`
- 규모: 5개 시나리오 × 8,640스텝 × 4개 arm = 20 롤아웃 (총 172,800스텝)
- 소요: 약 4시간 17분 (notebook wall clock 15,421초)

## 1. 실험 설계

| Arm | 예측기 | 설명 |
| --- | --- | --- |
| A | `NoForecast` | 기준선. 기본 `PolicyConfig` 그대로, 예측 없음 |
| B | `PeriodicHeuristicForecaster` | 24시간 lag + 감쇠 추세, 시간당 1회 갱신 |
| C | `TimesFM25Forecaster` | TimesFM 2.5 200M (`google/timesfm-2.5-200m-pytorch`) |
| D | `OracleForecaster` | 시뮬레이터의 실제 미래 OAT를 그대로 사용 (상한) |

예측은 `smart_buildings_toolkit/toolkit/policy_core.py:90-114`의 `_forecast_band_shift`를 통해서만 제어에 개입한다.
평일(day<5) 사전 구간(`comfort_soon` > 0.5 또는 `0 < morning_start_hour - hour <= 2`)에서만 작동하며,

- p90 > 상한 + 6K → 선냉방 밴드를 `comfort_hi - 0.5K`로 조임
- p10 < 하한 − 15K → 선난방 밴드를 `comfort_lo + 0.5K`로 올림

시나리오는 `DEFAULT_5_SCENARIOS` (building_train_1 × atlanta_2, denver_7, fairbanks_1, phoenix_5, singapore_8).

## 2. 실행 명령

```bash
# Kaggle 노트북 셀 4
python smart_buildings_toolkit/scripts/run_experiment.py \
  --mode rep5 --arms A B C D --extra-test-count 0 --skip-install
```

## 3. 결과 (MAE, 0 기준, 낮을수록 우수)

| 시나리오 | pid_water | fixed_setpoint | pid | A 무예측 | B 휴리스틱 | C TimesFM | D Oracle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| atlanta_2 | 0.02410 | 0.02622 | 0.02565 | 0.02400 | 0.02415 | 0.02415 | 0.02415 |
| denver_7 | 0.02992 | 0.03076 | 0.03065 | 0.02842 | 0.02781 | 0.02770 | 0.02810 |
| fairbanks_1 | 0.21168 | 0.14749 | 0.15128 | 0.30954 | 0.30566 | 0.30566 | 0.30566 |
| phoenix_5 | 0.13354 | 0.12952 | 0.15441 | 0.13607 | 0.10499 | 0.10499 | 0.10273 |
| singapore_8 | 0.02815 | 0.02662 | 0.03124 | 0.02983 | 0.02754 | 0.02764 | 0.02786 |
| **평균** | **0.08548** | **0.07212** | **0.07864** | 0.10557 | 0.09803 | 0.09803 | 0.09770 |

기준선 값은 `baseline_returns.json`의 총보상을 8,640으로 나눈 값.

| Arm | 5시나리오 평균 MAE | Δ vs A | 정책 시간(5롤아웃) | 예측 시간 | 30롤아웃 추정 |
| --- | --- | --- | --- | --- | --- |
| A | 0.10557 | — | 43.0초 | 0.0초 | 4.3분 |
| B | 0.09803 | −0.00754 (−7.1%) | 45.2초 | 0.97초 | 4.5분 |
| C | 0.09803 | −0.00754 (−7.1%) | 976.0초 | 931.7초 | 97.6분 |
| D | 0.09770 | −0.00787 (−7.5%) | 44.7초 | 0.45초 | 4.5분 |

## 4. 관찰

1. **TimesFM(C)은 휴리스틱(B) 대비 개선이 없다.** atlanta_2, fairbanks_1, phoenix_5, singapore_8에서 MAE가 소수점 5자리까지 동일하고, denver_7만 0.0001 차이다. 예측 비용은 에피소드당 166~205초(0.13초/호출 × 약 160회)로 B의 약 1,000배.
2. **병목은 예측 정확도가 아니라 트리거 로직이다.** 완벽한 미래 OAT를 쓰는 Oracle(D)조차 B보다 0.3% 좋을 뿐이다(0.09803 → 0.09770). 즉 6K/15K 임계값을 넘는 순간이 드물고, 넘으면 어떤 예측기든 동일한 액션을 낸다. 예측기 교체로 얻을 수 있는 상한이 이미 포화 상태.
3. **예측의 존재 자체는 효과가 있다.** A → B에서 평균 −7.1%, 특히 phoenix_5가 0.13607 → 0.10499 (−23%)로 가장 크다. denver_7(−2.1%), singapore_8(−7.7%)도 개선, atlanta_2는 0.00015 악화.
4. **5개 시나리오 평균에서 단순 기준선에도 뒤진다.** `fixed_setpoint` 0.07212, `pid` 0.07864가 우리 최고(B 0.09803)보다 좋다. 원인은 fairbanks_1 하나다: 우리 A/B 0.30566 vs `fixed_setpoint` 0.14749, `pid_water` 0.21168. 나머지 4개에서는 우리가 대등하거나 우세하다(atlanta·denver·phoenix는 더 좋고, singapore만 0.02754 vs 0.02662로 근소하게 뒤짐). fairbanks만 `fixed_setpoint` 수준(0.1475)으로 맞추면 5시나리오 평균은 0.09803 → **0.0664**가 된다.

## 5. 산출물

- `ablation_results/kaggle_run/ablation_results/ablation_summary.csv` — 롤아웃별 원자료
- `ablation_results/kaggle_run/ablation_results/ablation_summary.json` — 동일 내용 JSON
- `ablation_results/kaggle_run/notebookaf13d60ffe.log` — Kaggle 실행 로그
- 노트북: `kaggle/smart_buildings_ablation.ipynb`, 메타데이터 `kaggle/kernel-metadata.json`

※ `ablation_results/`와 `smart_buildings/extra_test_scenarios/`는 AGENTS.md §6에 따라 커밋하지 않는다.

## 6. 다음 단계

1. **fairbanks_1 원인 분리** — `fix_hws_cooling_bug=False`로 동일 조건 재실행, `pid_water`·`fixed_setpoint`와의 스텝별 액션 차이 비교. 한랭 기후에서 HWS 난방 전용 수정이 손해인지 확인.
2. **트리거 임계값 재조정** — 현재 6K/15K는 Oracle 결과가 보여주듯 대부분 발동하지 않는다. phoenix 이득이 나온 구간의 실제 OAT 분포를 보고 완화 여부 결정.
3. TimesFM은 **폐기 후보** — 정확도 이득 0, 비용 20배. 재도입하려면 배치 추론으로 예측 비용을 10초/에피소드 이하로 줄일 때만.
