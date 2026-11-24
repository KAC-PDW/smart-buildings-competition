# Run 2 — Fairbanks 원인 수정, 연속 사전 냉난방 및 미공개 일반화 검증

- 날짜: 2026-11-24 (KST)
- 실행 위치: Kaggle Notebook (`kabonzz/notebookaf13d60ffe`, version 3, CPU 4 workers, Internet ON)
- 저장소 커밋: `beb6831`
- 규모: 총 35롤아웃 × 8,640스텝 = 302,400스텝 (`shipped` 5×5 + `unseen_extra` 5×2)
- 소요: 약 5시간 6분 (notebook wall clock 18,341초; `run2` 본 실행 17,426초)

## 1. 실험 설계

Run 1([docs/run1.md](docs/run1.md))에서 `fairbanks_1` 부진과 6K/15K 이진 트리거 포화의 원인을 코드 수준에서 분리한 뒤, 아래 5개 arm을 비교했다.

| Arm | 예측기 | 핵심 제어 구성 | 목적 |
| --- | --- | --- | --- |
| `R2_A_NoHWSFix` | `NoForecast` | Run 1 기본 설정 + `fix_hws_cooling_bug=False`, `fix_sensor_band_precond=False`, `fix_hws_pump_demand_check=False` | 한랭 기후에서 HWS 냉방 버그 수정이 손해인지 분리 확인 |
| `R2_B_BugfixOnly` | `NoForecast` | `fix_hws_cooling_bug=True`, `fix_sensor_band_precond=True`, `fix_hws_pump_demand_check=True`, `worst_zone_guard_heat_k=999.0`, `enable_cold_hws_boost=True`, `reheat_full_k=1.5`, `reset_full_k=3.0`, `reset_effort_quantile=0.50` | 예측 없이 버그 수정·한랭 보정 단독 효과 측정 |
| `R2_C_Ramp_Heuristic` | `PeriodicHeuristicForecaster` | `R2_B` + `use_continuous_precond=True`, `adaptive_eco_cap=True`, `pre_window_hours=3`, `precool_margin_k=1.0`, `preheat_margin_k=1.0` | 연속 사전 냉난방 램프 + 야간 드리프트 캡 + 순수 NumPy 휴리스틱 |
| `R2_D_Ramp_AdaptiveFcst` | `DiurnalAdaptiveForecaster` | `R2_C`와 동일 제어 + 다일간 EWMA 일중 프로파일·편향 보정·경험적 분위수 예측기 | 경량 적응형 예측기 대비 휴리스틱 비교 |
| `R2_E_Ramp_Oracle` | `OracleForecaster` | `R2_C`와 동일 제어 + 실제 미래 OAT(상한) | 연속 램프 구조에서의 예측 상한 측정 |

- `shipped`: `DEFAULT_5_SCENARIOS` (`building_train_1` × `atlanta_2`, `denver_7`, `fairbanks_1`, `phoenix_5`, `singapore_8`)
- `unseen_extra`: 미공개 절차적 생성 건물 5개(`building_0500`, `0501`, `0502`, `0504`, `0505`) × 미사용 기상 스트립(`minneapolis_1~5`)에서 `R2_B_BugfixOnly` vs `R2_C_Ramp_Heuristic` 비교

## 2. 실행 명령

```bash
# Kaggle 노트북 셀 4
python smart_buildings_toolkit/scripts/run_experiment.py \
  --mode run2 --extra-test-count 5 --workers 4 --skip-install
```

## 3. 결과 (MAE, 0 기준, 낮을수록 우수)

### 3.1 `shipped` 5개 시나리오 (`building_train_1`)

| 시나리오 | `pid_water` | `fixed_setpoint` | `pid` | Run 1 `Arm_B` | `R2_A_NoHWSFix` | `R2_B_BugfixOnly` | `R2_C_Ramp_Heur` | `R2_D_Ramp_Adapt` | `R2_E_Ramp_Oracle` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `atlanta_2` | **0.02410** | 0.02622 | 0.02565 | 0.02415 | 0.02402 | 0.02462 | 0.02708 | 0.02701 | 0.02678 |
| `denver_7` | 0.02992 | 0.03076 | 0.03065 | 0.02781 | 0.02750 | 0.02846 | **0.02687** | 0.02690 | 0.02689 |
| `fairbanks_1` | 0.21168 | **0.14749** | 0.15128 | 0.30566 | 0.30954 | 0.16505 | 0.16578 | 0.16598 | 0.16619 |
| `phoenix_5` | 0.13354 | 0.12952 | 0.15441 | 0.10499 | 0.13468 | 0.12346 | 0.06022 | 0.06028 | **0.05811** |
| `singapore_8` | 0.02815 | **0.02662** | 0.03124 | 0.02754 | 0.02924 | 0.02992 | 0.02682 | 0.02683 | 0.02698 |
| **평균** | 0.08548 | 0.07212 | 0.07864 | 0.09803 | 0.10500 | 0.07430 | **0.06135** | 0.06140 | **0.06099** |

기준선 값은 `baseline_returns.json`의 총보상을 8,640으로 나눈 값.

| Arm | 5시나리오 평균 MAE | Δ vs `R2_A` | 정책 시간(5롤아웃) | 예측 시간 | 30롤아웃 추정 |
| --- | --- | --- | --- | --- | --- |
| `R2_A_NoHWSFix` | 0.10500 | — | 89.64초 | 0.01초 | 9.0분 |
| `R2_B_BugfixOnly` | 0.07430 | −0.03069 (−29.2%) | 90.16초 | 0.01초 | 9.0분 |
| `R2_C_Ramp_Heuristic` | 0.06135 | −0.04364 (−41.6%) | 94.05초 | 1.68초 | 9.4분 |
| `R2_D_Ramp_AdaptiveFcst` | 0.06140 | −0.04360 (−41.5%) | 94.98초 | 2.57초 | 9.5분 |
| `R2_E_Ramp_Oracle` | 0.06099 | −0.04401 (−41.9%) | 84.53초 | 0.67초 | 8.5분 |

### 3.2 `unseen_extra` 5개 미공개 절차적 건물 × `minneapolis` (8,640스텝)

| 건물 / 기상 | `R2_B_BugfixOnly` | `R2_C_Ramp_Heuristic` | Δ (C vs B) |
| --- | --- | --- | --- |
| `building_0500` / `minneapolis_1` | **0.14375** | 0.14393 | +0.00018 |
| `building_0501` / `minneapolis_2` | **0.14463** | 0.14783 | +0.00320 |
| `building_0502` / `minneapolis_3` | 0.06221 | **0.05794** | −0.00427 |
| `building_0504` / `minneapolis_4` | 0.04134 | **0.04023** | −0.00111 |
| `building_0505` / `minneapolis_5` | 0.02943 | **0.02857** | −0.00086 |
| **평균** | 0.08427 | **0.08370** | **−0.00057** |

## 4. 관찰

1. **5개 시나리오 평균에서 모든 공식 기준선 돌파**: `R2_C_Ramp_Heuristic`의 평균 MAE **0.06135**는 Run 1 최고(`Arm_B` 0.09803) 대비 **−37.4%** 개선이며, `fixed_setpoint`(0.07212), `pid`(0.07864), `pid_water`(0.08548)를 모두 큰 폭으로 앞섰다. 30롤아웃 정책 추론 추정 시간도 9.4분으로 9시간 한도 대비 여유가 크다.
2. **`fairbanks_1` 원인 확정 및 해소**: `R2_A_NoHWSFix`가 0.30954로 Run 1 `Arm_A`와 동일해 `fix_hws_cooling_bug`가 원인이 아님이 확인되었다. 반면 `R2_B_BugfixOnly`만으로 `fairbanks_1`이 **0.30954 → 0.16505 (−46.7%)**로 급락해 `pid_water`(0.21168)보다 크게 우수해졌다.
3. **연속 사전 냉난방 램프와 야간 드리프트 캡의 큰 폭 개선 (`phoenix_5`, `singapore_8`, `denver_7`)**: `R2_B` → `R2_C`에서 `phoenix_5`가 **0.12346 → 0.06022 (−51.2%)**로 절반 이하로 줄었고, `singapore_8`(0.02992 → 0.02682)과 `denver_7`(0.02846 → 0.02687)도 개선되었다. 반면 온화한 `atlanta_2`에서는 0.02462 → 0.02708로 소폭 상승해 `adaptive_eco_cap`/`precool_ramp_start_delta_k`가 과도하게 개입하는 구간이 있다.
4. **휴리스틱 ≈ 적응형 예측기 ≈ Oracle**: `R2_C_Ramp_Heuristic`(0.06135), `R2_D_Ramp_AdaptiveFcst`(0.06140), `R2_E_Ramp_Oracle`(0.06099)의 차이가 0.6% 이내로 수렴해, 순수 NumPy `PeriodicHeuristicForecaster`만으로 예측 상한에 거의 도달했다.

## 5. 산출물

- `ablation_results/kaggle_run2/ablation_results/ablation_summary.csv` — 35롤아웃 원자료
- `ablation_results/kaggle_run2/ablation_results/ablation_summary.json` — 동일 내용 JSON
- `ablation_results/kaggle_run2/notebookaf13d60ffe.log` — Kaggle 실행 로그
- 노트북: `kaggle/smart_buildings_ablation.ipynb`, 메타데이터 `kaggle/kernel-metadata.json`

※ `ablation_results/`와 `smart_buildings/extra_test_scenarios/`는 `AGENTS.md §6`에 따라 커밋하지 않는다.

## 6. 다음 단계

1. **`R2_C_Ramp_Heuristic`을 공식 `TransferPolicy`에 반영하고 30개 공식 시나리오 전체 평가 실행** — `score_leaderboard.py`로 전체 기준선 `pid_water`(**0.054141**) 돌파 여부 검증.
2. **온화한/극한 기후 미세 조정** — `atlanta_2` 소폭 악화 완화(`precool_ramp_start_delta_k` / `adaptive_eco_cap`) 및 `fairbanks_1` 추가 개선 여지 탐색.
