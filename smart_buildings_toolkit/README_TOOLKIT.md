# Smart Buildings Competition — Generalized Toolkit

[KAC-PDW/starter-kit](https://github.com/KAC-PDW/starter-kit) 루트에 그대로 압축을 풀어 사용하는 모듈형 실험·최적화·검증·3D 시각화 툴킷입니다.

## 디렉터리 구조

| 경로 | 역할 |
| :--- | :--- |
| `AGENTS.md` | 코딩 에이전트 및 팀원용 저장소 개발·검증·규칙 가이드 |
| `toolkit/forecasters.py` | 플러그형 외기 온도 예측기 (`NoForecast`, `PeriodicHeuristic`, `TimesFM25`, `Oracle`) |
| `toolkit/policy_core.py` | 파라미터화된 일반화 제어 정책 (`PolicyConfig` + `GeneralizedHVACPolicy`) |
| `toolkit/scenarios.py` | 기본 30개 / 대표 5개 / 미공개 절차적 10개 시나리오 및 Leave-Buildings-Out 분할 관리 |
| `toolkit/evaluator.py` | 통합 롤아웃 평가, 30시나리오 9시간 런타임 환산, CSV/JSON 리포트 생성 |
| `toolkit/exporter_3d.py` | 임의의 건물·정책 롤아웃을 Three.js 3D 뷰어용 JSON으로 내보내기 |
| `toolkit/optimizer_cma.py` | Leave-Buildings-Out 일반화 파라미터 최적화 (`optuna` / `cma` / 순수 NumPy 폴백) |
| `scripts/run_experiment.py` | 로컬·Kaggle Notebook 원커맨드 실험 실행기 (`smoke`, `rep5`, `all40`, `export3d`) |
| `scripts/check_submission.py` | Kaggle 제출 전 사전 감사 (가변 차원, NaN/범위, 0-액션 함정, 9시간 런타임 체크) |
| `scripts/sync_to_transfer_policy.py` | 최적화된 `PolicyConfig` JSON을 오프라인 제출용 `transfer_policy.py`에 반영 |
| `../smart_buildings_3d_viewer.html` | Three.js 3D 구역 열지도·AHU/HWS 상태·타임라인·JSON 드래그앤드롭 뷰어 (저장소 루트) |

## 빠른 실행 (로컬 및 Kaggle Notebook 공통)

```bash
# 1. 빠른 1일 스모크 테스트 + 3D 뷰어 JSON 생성
python scripts/run_experiment.py --mode smoke --arms A B D

# 2. 대표 5기후 + 미공개 절차적 테스트 10개 (30일 = 8,640스텝)
python scripts/run_experiment.py --mode rep5 --arms A B D

# 3. 기본 30개 + 미공개 10개 전체 평가 (40개 시나리오, 8,640스텝)
python scripts/run_experiment.py --mode all40 --arms A B D

# 4. Leave-Buildings-Out 파라미터 최적화
python -m toolkit.optimizer_cma --trials 20 --steps 288

# 5. Kaggle 제출 전 transfer_policy.py 자동 검증
python scripts/check_submission.py --steps 288
```
