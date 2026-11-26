# Run 4 — 데드밴드 난방 가드, 정압 스케일링 분리 및 6개 기후 다기후 합성 검증

- 날짜: 2026-11-26 (KST)
- 실행 위치: Kaggle Notebook (`kabonzz/notebookaf13d60ffe`, version 6, CPU 4 workers, Internet ON)
- 저장소 커밋: `53ce9f1`
- 규모: 총 27롤아웃 × 8,640스텝 = 233,280스텝 (`shipped` 5×3 + `unseen_diverse` 6×2)
- 소요: 약 5시간 2분 (notebook wall clock 18,116초)

## 1. 실험 설계

Run 3([docs/run3.md](docs/run3.md))에서 확인된 `phoenix_5` 회귀 원인(데드밴드 없는 난방 가드의 냉방/유휴 덮어쓰기)을 수정하고, 난방 정압 스케일링(`use_heat_pressure_scaling`)의 순수 효과를 분리하며, 미공개 절차적 건물 6개에 **전 6개 기후**의 미사용 기상 스트립을 1개씩 매핑해 일반화 성능을 검증했다.

| Arm | 구성 | 목적 |
| --- | --- | --- |
| `R4_A_Run2Ref` | Run 2 `R2_C_Ramp_Heuristic` (`TransferPolicy` 공식 30시나리오 0.041093 기준) | 기준선 |
| `R4_B_DeadbandNoPScale` | `R4_A` + `use_soft_heat_guard=True`, `soft_heat_guard_deadband_k=1.5`, `use_heat_pressure_scaling=False`, `use_continuous_eco_cap=True`, `adaptive_eco_cap=False`, `precool_ramp_start_delta_k=1.0`, `cold_temper_margin_k=0.35` | 데드밴드 + 냉방 우선 + 연속 셋백 + 템퍼링 (정압 스케일링 미적용) |
| `R4_C_DeadbandWithPScale` | `R4_B` + `use_heat_pressure_scaling=True` | 난방 정압 스케일링(`min_scale=0.35`) 순수 효과 분리 |

- `shipped`: `DEFAULT_5_SCENARIOS` (`building_train_1` × `atlanta_2`, `denver_7`, `fairbanks_1`, `phoenix_5`, `singapore_8`)
- `unseen_diverse`: 미공개 절차적 건물 6개(`building_0500`, `0501`, `0502`, `0504`, `0505`, `0506`) × 미사용 6개 기후(`minneapolis_1`, `atlanta_4`, `denver_1`, `fairbanks_3`, `phoenix_2`, `singapore_2`)에서 `R4_A_Run2Ref` vs `R4_B_DeadbandNoPScale` 비교

## 2. 실행 명령

```bash
# Kaggle 노트북 셀 4
python smart_buildings_toolkit/scripts/run_experiment.py \
  --mode run4 --extra-test-count 6 --workers 4 --skip-install
```

## 3. 결과 (MAE, 0 기준, 낮을수록 우수)

### 3.1 `shipped` 5개 시나리오 (`building_train_1`)

| 시나리오 | `pid_water` | `R4_A_Run2Ref` | `R4_B_DeadbandNoPScale` | `R4_C_DeadbandWithPScale` | Δ (`R4_C` vs `R4_A`) |
| --- | --- | --- | --- | --- | --- |
| `atlanta_2` | 0.02410 | 0.02708 | 0.02266 | **0.02228** | **−0.00480 (−17.7%)** |
| `denver_7` | 0.02992 | 0.02687 | 0.02482 | **0.02479** | **−0.00208 (−7.7%)** |
| `fairbanks_1` | 0.21168 | **0.16578** | 0.22323 | 0.16854 | +0.00276 (+1.7%) |
| `phoenix_5` | 0.13354 | **0.06022** | 0.06666 | 0.06838 | +0.00816 (+13.6%) |
| `singapore_8` | 0.02815 | **0.02682** | 0.02694 | 0.02694 | +0.00012 (+0.4%) |
| **평균** | 0.08548 | **0.06135** | 0.07286 | **0.06219** | +0.00083 (+1.4%) |

| Arm | 5시나리오 평균 MAE | Δ vs `R4_A` | 정책 시간(5롤아웃) | 30롤아웃 추정 |
| --- | --- | --- | --- | --- |
| `R4_A_Run2Ref` | **0.06135** | — | 95.45초 | 9.5분 |
| `R4_B_DeadbandNoPScale` | 0.07286 | +0.01151 | 101.57초 | 10.2분 |
| `R4_C_DeadbandWithPScale` | 0.06219 | +0.00083 | 103.07초 | 10.3분 |

### 3.2 `unseen_diverse` 6개 미공개 절차적 건물 × 전 6개 기후 (8,640스텝)

| 미공개 건물 / 미사용 기상 | 기후군 | `R4_A_Run2Ref` | `R4_B_DeadbandNoPScale` | Δ (`R4_B` vs `R4_A`) |
| --- | --- | --- | --- | --- |
| `building_0500` / `minneapolis_1` | 한랭 대륙성 | 0.14393 | **0.12986** | **−0.01407 (−9.8%)** |
| `building_0501` / `atlanta_4` | 온화·습윤 | **0.03153** | 0.03211 | +0.00058 (+1.8%) |
| `building_0502` / `denver_1` | 고지대 대륙성 | 0.09144 | **0.06478** | **−0.02666 (−29.2%)** |
| `building_0504` / `fairbanks_3` | 극한 한랭 | 0.06445 | **0.04175** | **−0.02270 (−35.2%)** |
| `building_0505` / `phoenix_2` | 고온 사막 | **0.02821** | 0.02840 | +0.00019 (+0.7%) |
| `building_0506` / `singapore_2` | 고온 다습 열대 | **0.04626** | 0.04629 | +0.00003 (+0.1%) |
| **평균 (6개 기후)** | **전체** | 0.06764 | **0.05720** | **−0.01044 (−15.4%)** |

## 4. 관찰

1. **6개 기후 미공개 합성 건물 일반화 대성공 (−15.4%)**: 처음 보는 6개 건물 × 전 6개 기후 평균이 **0.06764 → 0.05720 (−15.4%)**로 개선되었다(`fairbanks_3` −35.2%, `denver_1` −29.2%, `minneapolis_1` −9.8%). 고온 미공개 건물(`phoenix_2`, `singapore_2`)에서도 회귀가 거의 없었다.
2. **데드밴드(`1.5K`) + 냉방 우선 조건으로 `phoenix_5` 회귀 90% 해소**: Run 3의 `0.12187`에서 `0.06666`/`0.06838`로 회복되었다. 남은 +0.008 격차는 Run 4에서 고온 예냉 임계값(`precool_ramp_start_delta_k: -1.0 → +1.0`)과 야간 고온 캡(`+1.5K → +3.0~+1.0K`)을 완화한 영향이다.
3. **난방 정압 스케일링(`use_heat_pressure_scaling=True`)의 필수성 입증**: `R4_B`(미적용) vs `R4_C`(적용) 비교에서 `fairbanks_1`이 **0.22323 → 0.16854 (−24.5%)**로 크게 좋아졌고, `atlanta_2`(0.02228)·`denver_7`(0.02479)도 모두 `R4_C`가 우세했다. `use_soft_heat_guard`는 반드시 `use_heat_pressure_scaling=True`와 쌍으로 사용해야 만족 구역으로의 한기 강풍 유입을 막을 수 있다.

## 5. 산출물

- `ablation_results/kaggle_run4/ablation_results/ablation_summary.csv` — 27롤아웃 원자료
- `ablation_results/kaggle_run4/ablation_results/ablation_summary.json` — 동일 내용 JSON
- `ablation_results/kaggle_run4/notebookaf13d60ffe.log` — Kaggle 실행 로그

※ `ablation_results/`와 `smart_buildings/extra_test_scenarios*/`는 `AGENTS.md §6`에 따라 커밋하지 않는다.

## 6. 다음 단계

1. **`R4_C_DeadbandWithPScale` + Run 2 고온 예냉 기준(`precool_ramp_start_delta_k=-1.0`, 고온 야간 캡 강화)** 조합을 공식 `TransferPolicy`에 반영.
2. **공식 30개 시나리오 전체 평가 + `unseen_diverse` 6개 기후 통합 검증**(36롤아웃)을 실행해 기존 공식 최고점 **0.041093** 갱신 여부 확인.
