# Run 3 — 일반화 가능한 한랭·전 기후 물리 제어 절제 실험

- 날짜: 2026-11-25 ~ 2026-11-26 (KST)
- 실행 위치: Kaggle Notebook (`kabonzz/notebookaf13d60ffe`, version 5, CPU 4 workers, Internet ON)
- 저장소 커밋: `758b8d5`
- 규모: 총 30롤아웃 × 8,640스텝 = 259,200스텝 (`shipped` 5×4 + `unseen_extra` 5×2)
- 소요: 약 4시간 56분 (notebook wall clock 17,759초)

## 1. 실험 설계

시뮬레이터(`vav.py`, `air_handler.py`)의 열·유체 역학에 근거한 4가지 일반화 물리 제어 메커니즘을 단계적으로 추가해 검증했다. 예측기는 모두 순수 NumPy `PeriodicHeuristicForecaster`(`B`)를 사용했다.

| Arm | 구성 | 추가 메커니즘 |
| --- | --- | --- |
| `R3_A_Run2Ref` | Run 2 `R2_C_Ramp_Heuristic` | 기준선(공식 30시나리오 MAE 0.041093 달성 설정) |
| `R3_B_SoftColdGuard` | `R3_A` + `use_soft_heat_guard=True`, `use_heat_pressure_scaling=True` | 외곽 저온 구역 기아 방지 연속 가드(`min_w=0.35`, `2.0K→4.5K`) + 난방 정압 스케일링(`min_scale=0.35`) |
| `R3_C_ContSetback` | `R3_B` + `use_continuous_eco_cap=True`, `adaptive_eco_cap=False`, `precool_ramp_start_delta_k=1.0` | `|T_comfort - OAT|` 비례 연속 야간 셋백 + 주말 주간 완화 |
| `R3_D_FullColdPhys` | `R3_C` + `cold_temper_margin_k=0.35` | $\text{OAT} < 0^\circ\text{C}$ 시 하한 직상단 구역 재열 밸브 소량 템퍼링(상한 0.35) |

- `shipped`: `DEFAULT_5_SCENARIOS` (`building_train_1` × `atlanta_2`, `denver_7`, `fairbanks_1`, `phoenix_5`, `singapore_8`)
- `unseen_extra`: 미공개 절차적 건물 5개(`building_0500`, `0501`, `0502`, `0504`, `0505`) × `minneapolis_1~5`에서 `R3_A_Run2Ref` vs `R3_D_FullColdPhys` 비교

## 2. 실행 명령

```bash
# Kaggle 노트북 셀 4
python smart_buildings_toolkit/scripts/run_experiment.py \
  --mode run3 --extra-test-count 5 --workers 4 --skip-install
```

## 3. 결과 (MAE, 0 기준, 낮을수록 우수)

### 3.1 `shipped` 5개 시나리오 (`building_train_1`)

| 시나리오 | `pid_water` | `R3_A_Run2Ref` | `R3_B_SoftColdGuard` | `R3_C_ContSetback` | `R3_D_FullColdPhys` | Δ (`R3_D` vs `R3_A`) |
| --- | --- | --- | --- | --- | --- | --- |
| `atlanta_2` | 0.02410 | 0.02708 | 0.01968 | 0.01945 | **0.01926** | **−0.00782 (−28.9%)** |
| `denver_7` | 0.02992 | 0.02687 | 0.01708 | **0.01702** | **0.01702** | **−0.00985 (−36.7%)** |
| `fairbanks_1` | 0.21168 | **0.16578** | 0.19059 | 0.19418 | 0.17488 | +0.00910 (+5.5%) |
| `phoenix_5` | 0.13354 | **0.06022** | 0.12106 | 0.12187 | 0.12187 | +0.06165 (+102.4%) |
| `singapore_8` | 0.02815 | **0.02682** | **0.02682** | 0.02694 | 0.02694 | +0.00012 (+0.4%) |
| **평균** | 0.08548 | **0.06135** | 0.07504 | 0.07589 | 0.07199 | +0.01064 (+17.3%) |

| Arm | 5시나리오 평균 MAE | Δ vs `R3_A` | 정책 시간(5롤아웃) | 30롤아웃 추정 |
| --- | --- | --- | --- | --- |
| `R3_A_Run2Ref` | **0.06135** | — | 82.26초 | 8.2분 |
| `R3_B_SoftColdGuard` | 0.07504 | +0.01369 | 90.47초 | 9.0분 |
| `R3_C_ContSetback` | 0.07589 | +0.01454 | 91.79초 | 9.2분 |
| `R3_D_FullColdPhys` | 0.07199 | +0.01064 | 95.10초 | 9.5분 |

### 3.2 `unseen_extra` 5개 미공개 절차적 건물 × `minneapolis` (8,640스텝)

| 건물 / 기상 | `R3_A_Run2Ref` | `R3_D_FullColdPhys` | Δ (`R3_D` vs `R3_A`) |
| --- | --- | --- | --- |
| `building_0500` / `minneapolis_1` | 0.14393 | **0.10793** | −0.03600 (−25.0%) |
| `building_0501` / `minneapolis_2` | 0.14783 | **0.06151** | −0.08632 (−58.4%) |
| `building_0502` / `minneapolis_3` | 0.05794 | **0.03553** | −0.02241 (−38.7%) |
| `building_0504` / `minneapolis_4` | 0.04023 | **0.03986** | −0.00037 (−0.9%) |
| `building_0505` / `minneapolis_5` | **0.02857** | 0.02872 | +0.00015 (+0.5%) |
| **평균** | 0.08370 | **0.05471** | **−0.02900 (−34.6%)** |

## 4. 관찰

1. **미공개 다구역 건물 및 온화·간절기에서의 큰 폭 개선**: `unseen_extra` 5개 건물 평균이 **0.08370 → 0.05471 (−34.6%)**로 하락했고(`building_0501` −58.4%, `building_0502` −38.7%), `atlanta_2`(0.02708 → **0.01926**, −28.9%)와 `denver_7`(0.02687 → **0.01702**, −36.7%)도 대폭 개선되었다.
2. **`phoenix_5` 악화 원인 (`R3_A` 0.06022 → `R3_B` 0.12106)**: `use_soft_heat_guard=True`가 데드밴드 없이 `worst_heat > 0.0`이면 즉시 개입하고, `mean_err < 0`인 냉방 상황에서도 `worst_heat >= 2.0`이면 AHU를 난방 모드로 전환해 HWS 보일러·펌프를 켜기 때문이다.
3. **`fairbanks_1`에서의 정압 스케일링 vs 템퍼링 효과**: `R3_B`의 `use_heat_pressure_scaling`이 극한 한랭에서 유량을 과도하게 줄여 `fairbanks_1`이 0.16578 → 0.19059로 악화되었으나, `R3_D`에서 `cold_temper_margin_k=0.35` 추가 시 0.19418 → 0.17488로 상당 부분 회복되었다.

## 5. 산출물

- `ablation_results/kaggle_run3/ablation_results/ablation_summary.csv` — 30롤아웃 원자료
- `ablation_results/kaggle_run3/ablation_results/ablation_summary.json` — 동일 내용 JSON
- `ablation_results/kaggle_run3/notebookaf13d60ffe.log` — Kaggle 실행 로그

※ `ablation_results/`와 `smart_buildings/extra_test_scenarios/`는 `AGENTS.md §6`에 따라 커밋하지 않는다.

## 6. 다음 단계 (Run 4)

1. `use_soft_heat_guard`에 **발동 데드밴드(`soft_heat_guard_deadband_k=1.5`) 및 냉방 우선 조건**을 적용해 `phoenix_5` 회귀를 제거하고, `use_heat_pressure_scaling` 유무를 분리 검증.
2. 미사용 기상 스트립 6개 도시(`minneapolis`, `atlanta`, `denver`, `fairbanks`, `phoenix`, `singapore`)를 1개씩 라운드 로빈으로 뽑는 **다양한 다기후 합성 테스트(`unseen_diverse`)**로 일반화 검증 확대.
