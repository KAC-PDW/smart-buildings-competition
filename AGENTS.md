# AGENTS.md — Smart Buildings Competition (`KAC-PDW/starter-kit`)

이 문서는 [NeurIPS 2026 Smart Buildings Competition](https://www.google.com/search?q=https%3A%2F%2Fwww.kaggle.com%2Fcompetitions%2Fneurips-2026-smart-buildings)([공식 사이트](https://www.google.com/search?q=https%3A%2F%2Fsmart-buildings-2026.github.io%2Fcompetition%2F)) 개발을 위해 포크한 [KAC-PDW/starter-kit](https://github.com/KAC-PDW/starter-kit) 저장소에서 코딩 에이전트 및 팀원이 따라야 할 아키텍처, 규칙, 검증 프로토콜을 정의합니다.

## 1. 프로젝트 목표 및 평가 기준

* **핵심 과제**: 시뮬레이션 건물에서 학습·설계한 **단일 HVAC 제어 정책**을 크기, 평면도, 단열 자재, 기후, AHU/VAV 구성이 다른 **처음 보는 상업용 건물**에 그대로 적용해 일반화 성능을 극대화합니다.

* **최종 마감**: 2027년 1월 15일 (Kaggle Private Leaderboard 코드 제출 및 OpenReview 최종 논문상 동시 마감).

* **리더보드 평가 지표**: 30개 시나리오(비공개 건물 6개 × 비공개 기상 관측소 5개, 각 30일 = 8,640스텝, 5분 간격)의 스텝당 평균 보상에 대한 **0 기준 MAE (낮을수록 우수)**.

* **보상 함수**: 매 스텝 `0.6 × 쾌적도 + 0.2 × 에너지 비용 + 0.2 × 탄소 배출`의 가중 페널티(항상 0 이하). 재실 구역 온도가 쾌적 밴드를 5K 초과해 벗어나면 2차(무한대) 페널티가 적용됩니다.

* **공식 기준 점수 (`baseline_returns.json`, 30개 개발 시나리오, 8,640스텝)**:

| 정책 | 30개 시나리오 평균 MAE | 평균 총보상 | 비고 | 
| ----- | ----- | ----- | ----- | 
| `pid_water` | **0.054141** | \-467.78 | 1차 돌파 목표 (Singapore·Fairbanks·Phoenix 개선 여지 큼) | 
| `fixed_setpoint` | 0.065915 | \-569.51 | Singapore(0.02522)에서 `pid_water`(0.04665)보다 우수 | 
| `pid` | 0.067356 | \-581.95 | Fairbanks(0.07546)에서 `pid_water`(0.07847)보다 우수 | 
| `bang_bang` | 0.225203 | \-1,945.75 | 최하위 기준선 | 

## 2. 저장소 구조 및 주요 파일

* `smart_buildings/baselines/transfer_policy.py`: **제출용 정책 구현 진입점** (`TransferPolicy`, `POLICY_MAP["transfer"]`).

* `smart_buildings/baselines/pid.py`, `pid_plus.py`: 참조 제어기(`pid`, `pid_water`). `pid_plus.py`는 `USE_SENSOR_BAND`, `USE_WORST_ZONE`, 이코노마이저, 비례 재열, 온수 리셋 로직을 포함합니다.

* `smart_buildings/evaluate.py`: 시나리오 로드(`load_hybrid_environment`), 메타데이터 생성(`create_building_meta_data`), 정책 평가 및 `POLICY_MAP` 레지스트리.

* `smart_buildings/score_leaderboard.py`: 8,640스텝 결과를 단일 리더보드 MAE로 집계하고 기준선과 나란히 출력.

* `smart_buildings/training_set.yaml`: 공식 개발용 30개 시나리오(6개 건물 × 5개 관측소) 매핑.

* `smart_buildings/run_floor_generator.py` / `run_scenario_generator.py`: 절차적 건물 및 시나리오 생성기 (`minneapolis` 포함 미사용 기상 스트립 활용 가능).

* `smart_buildings_toolkit/`: 실험 도구 모음. 진입점은 `scripts/run_experiment.py`(`--mode smoke|rep5|all40|export3d`), 내부에 `toolkit/policy_core.py`(`PolicyConfig`, `GeneralizedHVACPolicy`), `forecasters.py`(Arm A/B/C/D), `evaluator.py`, `scenarios.py`, `exporter_3d.py`, `optimizer_cma.py`, `paths.py`가 있습니다. 루트의 `run_kaggle_experiment.py`·`oat_forecast_ablation.py`는 이 스크립트로 위임하는 deprecated 래퍼입니다.

* `kaggle/`: Kaggle 노트북(`smart_buildings_ablation.ipynb`)과 푸시 메타데이터(`kernel-metadata.json`). 장시간 실험은 이 노트북으로 업로드해 실행합니다(§5).

* `docs/`: 실험 리포트(`run<N>.md`). 실험 종료 시 결과 표와 관찰을 커밋합니다.

## 3. 정책 인터페이스 계약 (필수 준수)

모든 정책은 상속 없이 아래 두 메서드만 구현합니다.

```
class TransferPolicy:
    def __init__(self, building_meta_data: Mapping[str, Any], action_spec: Any) -> None:
        # 인덱스 테이블과 AHU-구역 매핑은 반드시 __init__에서 1회만 구축
        ...

    def get_action(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        # discrete_action: int32, shape (K + 1,), values in {0, 1}
        # continuous_action: float32, shape (3K + 2 + 2V,), values in [-1, 1]
        ...

```

## 4. 절대 어기지 말아야 할 도메인·구현 함정

1. **연속 행동 인덱스 하드코딩 금지**: `continuous_action`은 `[temp_0, pressure_0, recirc_0, temp_1, ...]` 순으로 교차 배치(interleaved)되어 있습니다. 반드시 `building_meta_data["temp_indices"]`, `["pressure_indices"]`, `["recirculation_indices"]`, `["water_indices"]`, `["zone_action_indices"]`를 읽어 기록하세요.

2. **관측 벡터 오프셋 하드코딩 금지**: 건물마다 관측 길이가 다릅니다(예: `building_train_1` 352차원, `building_train_8` 271차원). `__init__`에서 `building_meta_data["observation_field_names"]`로 인덱스 딕셔너리를 구축해 슬라이싱하세요.

3. **0 액션은 꺼짐(Off)이 아님**: `0.0`은 정규화 구간 `[-1, 1]`의 중간값(예: 댐퍼 50% 개방)입니다. 쉬게 하려는 모든 슬롯에도 명시적으로 `-1.0` 등을 기록하세요.

4. **온수 펌프 버그 재발 금지**: `pid_plus.py`의 `PIDWorstZonePolicy`는 `any_active`로 온수 펌프(`water_discrete_idx`)와 차압(`water_pressure_idx`)을 켜서 냉방 전용 기후에서도 펌프를 돌립니다. 신규 정책에서는 반드시 **실제 난방 수요(`any_heating`)가 있을 때만** 온수 펌프를 가동하세요.

5. **물리적 제약 인지**: AHU는 **냉방 전용**이며 모든 난방은 HWS + VAV 재열 밸브에서 공급됩니다. 외기 온도가 영하 6.8°C 미만이면 AHU 급기 온도 명령은 무효화됩니다. 또한 시뮬레이터가 환기 하한(재순환 최대 70%, 댐퍼 최소 10% 개방, 가동 AHU 최소 정압 200 Pa)을 강제합니다.

6. **데이터 누수 및 규칙 위반 금지**: 관측에 없는 기상 관측소 이름, 월, 건물 ID, 비공개 시드를 추론하거나 하드코딩하는 행위는 실격 사유입니다.

7. **`smart_buildings/smart_control/` 수정 금지**: 시뮬레이터·보상·환경·프로토콜 파일(`*_pb2.py` 포함)은 평가 서버와 동일해야 하므로 절대 수정하지 마세요.

## 5. 환경 설정, 실행 및 검증 명령어

* **Python 및 의존성**: Python 3.12 이상 권장. `pyproject.toml`의 패키지 버전은 Kaggle 공식 이미지(`gcr.io/kaggle-images/python:latest`)에 고정되어 있으므로 버전을 임의로 올리지 마세요.

* **설치**:

  ```
  pip install -e .
  
  ```

* **단일 시나리오 스모크 테스트 (1일 = 288스텝, 점수 비교 불가)**:

  ```
  python -m smart_buildings.evaluate --policies transfer --buildings building_train_1 --weathers fairbanks_1 --steps 288
  
  ```

* **공식 30개 시나리오 전체 평가 및 리더보드 점수 계산 (반드시 8,640스텝)**:

  ```
  python -m smart_buildings.evaluate --policies transfer
  python score_leaderboard.py
  
  ```

* **외기 온도 예측 절제 실험 + 미공개 절차적 테스트 10개 검증**:

  ```
  python smart_buildings_toolkit/scripts/run_experiment.py --mode rep5 --arms A B D --extra-test-count 10

  ```

* **Kaggle 노트북에서 실행 (장시간 실험 기본 경로)**:

  * 인증: Kaggle Settings → API → **Generate New Token**으로 발급한 `KGAT_…` 토큰을 시스템 환경변수 `KAGGLE_API_TOKEN`으로 정의합니다(`kaggle` CLI ≥ 1.8.0 필요). legacy `~/.kaggle/kaggle.json`은 선택이며, 토큰 값은 문서·로그·커밋에 남기지 않습니다.

  * 노트북 정의: `kaggle/smart_buildings_ablation.ipynb`(셀: 클론 → `pip install -e . --no-deps` + `timesfm` → smoke → rep5 A/B/C/D → 결과 피벗)와 `kaggle/kernel-metadata.json`(id `kabonzz/notebookaf13d60ffe`, Internet ON, GPU 불필요 — 시뮬레이터가 CPU 바운드).

  * 업로드·실행(버전이 올라가며 즉시 재실행됨):

    ```
    python -m kaggle kernels push -p kaggle/

    ```

  * 상태 확인 / 결과 회수:

    ```
    python -m kaggle kernels status kabonzz/notebookaf13d60ffe
    python -m kaggle kernels output kabonzz/notebookaf13d60ffe -p ablation_results/kaggle_run

    ```

  * `kernels output`은 `/kaggle/working` 전체를 받으므로 클론본(`kaggle_run/smart-buildings-competition/`, 약 100MB)이 함께 내려옵니다 — 받은 뒤 삭제하세요. 결과 CSV는 `ablation_results/kaggle_run/ablation_results/`에 있습니다.

  * 노트북 셀은 `--skip-install`을 붙여 러너의 자동 `pip install`을 막습니다(설치는 노트북 셀에서 직접 수행).

* **실험 리포트**: 실험을 마치면 `docs/run<N>.md`에 조건·결과 표·관찰·다음 단계를 정리해 커밋합니다(예: `docs/run1.md`).

* **정적 검사 / 린트**:

  ```
  ruff check smart_buildings
  
  ```

## 6. 에이전트 작업 및 실험 워크플로 규칙

* **검증 없이 개선 주장 금지**: 모든 정책 변경은 최소 5개 대표 기후 시나리오(8,640스텝) 및 미공개 절차적 테스트 10개(`unseen_extra10`)에서 MAE 변화를 측정해 기록합니다. 스텝 수가 다른 에피소드(예: 288 vs 8,640)끼리는 절대 점수를 비교하지 마세요.

* **과적합 방지 (Leave-Buildings-Out)**: 개발 건물 6개에 파라미터를 맞출 때는 반드시 건물 단위 교차 검증과 `extra_test_scenarios`(시드 500+, `minneapolis` 포함)에서 일반화 성능을 함께 확인합니다.

* **Kaggle 제출 제약 준수**:

  * 최종 추론 코드는 **인터넷 차단** 환경과 **CPU/GPU 9시간 이하** 런타임(30개 롤아웃 = 259,200스텝)을 여유 있게 통과해야 합니다(목표: 30개 롤아웃 추론 시간 6시간 이하).

  * 학습에 외부 라이브러리(`torch_geometric`, `d3rlpy`, `optuna`, `cma`, `casadi`, `timesfm` 등)를 사용하더라도, **최종 제출 추론 코드는 가능하면 순수 NumPy/PyTorch만으로 동작**하도록 분리합니다.

  * 수상 시 소스코드는 **CC-BY 4.0 또는 상업적 이용을 제한하지 않는 OSI 승인 오픈소스 라이선스**로 공개해야 하므로, 비상업적(NC) 라이선스 코드·가중치는 사용하지 않습니다.

* **생성 파일 정리**: 실험 결과(`ablation_results/`, `smart_buildings/results/`, `smart_buildings/extra_test_scenarios/`)와 대용량 모델 체크포인트는 Git에 커밋하지 않도록 `.gitignore`에 유지합니다.