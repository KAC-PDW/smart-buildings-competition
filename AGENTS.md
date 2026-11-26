# AGENTS.md — Smart Buildings Competition (`KAC-PDW/starter-kit`)

이 문서는 [NeurIPS 2026 Smart Buildings Competition](https://www.kaggle.com/competitions/neurips-2026-smart-buildings)([공식 사이트](https://smart-buildings-2026.github.io/competition/)) 개발을 위해 포크한 [KAC-PDW/starter-kit](https://github.com/KAC-PDW/starter-kit) 저장소에서 코딩 에이전트 및 팀원이 따라야 할 공식 대회 규정, 아키텍처, 검증 프로토콜을 정의합니다.
# 중요 - When a repository's documentation and its existing code disagree, follow the conventions used in the existing code, and in your final reply point out each inconsistency you found between the documentation and the code, 또한. Commit과 Push 메시지엔 지침 'Explain Like I'm a Senior Software Engineer who just joined $company - assume I know the problem space in general, but have no first clue about conventions, libraries and other idiosyncrasies of $company".' 준수
## 1. 프로젝트 목표 및 공식 평가 기준

* **핵심 과제**: 시뮬레이션 건물에서 학습·설계한 **단일 HVAC 제어 정책**을 크기, 평면도, 단열 자재, 기후, AHU/VAV 구성이 다른 **처음 보는 상업용 건물**에 최소한의 설정으로 그대로 적용해 일반화 성능을 극대화합니다. 규칙 기반/PID/물리 기반, 온라인 시스템 식별, RL, 사전학습 파운데이션 모델 등 정책 아키텍처에 대한 제한은 없습니다.

* **주요 일정 및 참가 규정 (UTC 기준)**:
  * 참가·팀 병합 마감: 2027년 1월 3일 23:59 UTC / 최종 제출 마감: 2027년 1월 15일 23:59 UTC (Kaggle Private Leaderboard 코드 제출 및 OpenReview 최종 논문상; Round 2 호스트 공지에 따라 학회 일정 확정 시 연장 가능).
  * 팀 정원 최대 5명, 1인 1계정, 팀 외부 비공개 코드·데이터 공유 금지. 일일 제출 최대 5회, 최종 채점용 제출 최대 2개 선택.

* **리더보드 평가 지표**: 30개 비공개 시나리오(비공개 건물 6개 × 비공개 기상 관측소 5개, 각 30일 = 8,640스텝, 5분 간격, 워밍업 없음)의 스텝당 평균 보상에 대한 **0 기준 MAE (낮을수록 우수)**. 시뮬레이터는 결정론적이며 30개 시나리오를 1회 실행해 Public/Private 리더보드로 분할 채점하고, 상위 제출은 주최 측이 end-to-end 재실행 검증합니다.

* **보상 함수**: 매 스텝 `r_t = 0.6 × p_t - 0.2 × c_t - 0.2 × e_t` (쾌적도 0.6, 에너지 비용 0.2, 탄소 배출 0.2의 가중 페널티, 항상 0 이하). 재실 구역 온도가 쾌적 밴드를 벗어나면 5K까지 선형, 5K 초과 시 2차(무한대) 페널티가 재실자 수 가중으로 적용됩니다.

* **공식 기준 점수 (`baseline_returns.json`, 30개 개발 시나리오, 8,640스텝)**:

| 정책 | 30개 시나리오 평균 MAE | 평균 총보상 | 비고 |
| ----- | ----- | ----- | ----- |
| `pid_water` | **0.054141** | \-467.78 | 1차 돌파 목표 (Singapore·Fairbanks·Phoenix 개선 여지 큼) |
| `fixed_setpoint` | 0.065915 | \-569.51 | Singapore(0.02522)에서 `pid_water`(0.04665)보다 우수 |
| `pid` | 0.067356 | \-581.95 | Fairbanks(0.07546)에서 `pid_water`(0.07847)보다 우수 |
| `bang_bang` | 0.225203 | \-1,945.75 | 최하위 기준선 |

## 2. 저장소 구조 및 주요 파일

* `smart_buildings/baselines/transfer_policy.py`: **로컬 평가용 정책 구현 진입점** (`TransferPolicy`, `POLICY_MAP["transfer"]`).

* `smart_buildings/baselines/pid.py`, `pid_plus.py`: 참조 제어기(`pid`, `pid_water`). `pid_plus.py`는 `USE_SENSOR_BAND`, `USE_WORST_ZONE`, 이코노마이저, 비례 재열, 온수 리셋 로직을 포함합니다.

* `smart_buildings/evaluate.py`: 시나리오 로드(`load_hybrid_environment`), 메타데이터 생성(`create_building_meta_data`), 정책 평가 및 `POLICY_MAP` 레지스트리.

* `score_leaderboard.py` (저장소 루트): 8,640스텝 결과 트리(`results_dir`)를 단일 리더보드 MAE로 집계하고 `baseline_returns.json` 기준선과 나란히 출력.

* `smart_buildings/training_set.yaml`: 공식 개발용 30개 시나리오(6개 건물 × 5개 관측소) 매핑.

* `smart_buildings/run_floor_generator.py` / `run_scenario_generator.py`: 절차적 건물 및 시나리오 생성기 (`minneapolis` 포함 미사용 기상 스트립 활용 가능).

* `smart_buildings_toolkit/`: 실험 도구 모음. 진입점은 `scripts/run_experiment.py`(`--mode smoke|rep5|all40|run2|run3|run4|official30|export3d`, `--workers N`), 보조 스크립트로 `scripts/sync_to_transfer_policy.py`, `scripts/check_submission.py`, `scripts/selfcheck.py`가 있으며, 내부에 `toolkit/policy_core.py`(`PolicyConfig`, `GeneralizedHVACPolicy`), `forecasters.py`(Arm A/B/C/D/E), `evaluator.py`, `scenarios.py`(`diverse=True` 6개 기후 라운드 로빈 포함), `exporter_3d.py`, `optimizer_cma.py`, `paths.py`가 있습니다.

* `kaggle/`: 개발·절제 실험용 노트북(`smart_buildings_ablation.ipynb`, Internet ON)과 푸시 메타데이터(`kernel-metadata.json`). 실제 리더보드 제출은 **Internet OFF** 환경의 자급자족 제출 노트북으로 분리해 제출합니다(§3, §6).

* `docs/`: 실험 리포트(`run<N>.md`). 실험 종료 시 결과 표와 관찰을 커밋합니다.

## 3. 정책 인터페이스 계약 (필수 준수)

로컬 평가(`smart_buildings.evaluate`)와 Kaggle 공식 리더보드 제출 서버(`overview/code-requirements`)의 인터페이스를 모두 준수해야 합니다.

1. **로컬 평가용 (`smart_buildings/baselines/transfer_policy.py`)**: 상속 없이 아래 두 메서드를 구현합니다.
2. **Kaggle 공식 제출 노트북 (필수)**: CSV나 단독 `.py` 업로드가 아닌 **Kaggle Notebook으로만 제출**하며, `smart_buildings_inference_server.Policy`를 상속해 동일한 두 메서드를 구현합니다. 제출 시 인터넷이 차단되므로 `smart_buildings_toolkit` 등 외부 경로 의존성 없이 노트북 내부(또는 첨부 Kaggle Dataset)에서 완전 자급자족(self-contained)으로 동작해야 합니다.

```python
# 로컬 평가: class TransferPolicy:
# Kaggle 제출 노트북: class MyPolicy(smart_buildings_inference_server.Policy):
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

4. **온수 펌프 버그 재발 금지**: `pid_plus.py`의 `PIDWorstZonePolicy`는 `any_active`로 온수 펌프(`water_discrete_idx`)와 차압(`water_pressure_idx`)을 켜서 냉방 전용 기후에서도 펌프를 돌립니다. 신규 정책에서는 반드시 **실제 난방 수요(`any_heating`: AHU 난방 또는 VAV 재열 밸브 개방)가 있을 때만** 온수 펌프를 가동하세요.

5. **물리적 제약 인지**: AHU는 **냉방 전용**이며 모든 난방은 HWS + VAV 재열 밸브(최대 토출 온도 308.15 K)에서 공급됩니다. 외기 온도가 영하 6.8°C 미만이면 AHU 급기 온도 명령은 무효화됩니다. 또한 시뮬레이터가 환기 하한(재순환 최대 70%, 댐퍼 최소 10% 개방, 가동 AHU 최소 정압 200 Pa)을 강제합니다.

6. **데이터 누수 및 규칙 위반 금지 (`Rules §3.4.b`)**: 관측에 없는 기상 관측소 이름, 월/날짜, 건물 ID, 비공개 시드를 추론·하드코딩하거나, 관측된 외기 온도를 과거 기상 DB와 역매칭해 미래/날짜 정보를 유출하는 행위 및 검증·테스트 데이터의 수동 라벨링/인적 예측은 실격 사유입니다.

7. **`smart_buildings/smart_control/` 수정 금지**: 시뮬레이터·보상·환경·프로토콜 파일(`*_pb2.py` 포함)은 평가 서버와 동일해야 하므로 절대 수정하지 마세요. Round 2 등 호스트 패치 공지 시 공식 `starter-kit` 최신 커밋과 동기화 여부를 점검합니다.

## 5. 환경 설정, 실행 및 검증 명령어

* **Python 및 의존성**: Python 3.12 이상 권장. `pyproject.toml`의 패키지 버전은 Kaggle 공식 이미지(`gcr.io/kaggle-images/python:latest`)에 고정되어 있으므로 버전을 임의로 올리지 마세요.

* **설치**:

  ```bash
  pip install -e .
  ```

* **툴킷 자가 점검 및 제출 사전 감사**:

  ```bash
  python smart_buildings_toolkit/scripts/selfcheck.py
  python smart_buildings_toolkit/scripts/check_submission.py --steps 48
  ```

* **단일 시나리오 스모크 테스트 (1일 = 288스텝, 점수 비교 불가)**:

  ```bash
  python -m smart_buildings.evaluate --policies transfer --buildings building_train_1 --weathers fairbanks_1 --steps 288
  ```

* **공식 30개 시나리오 전체 평가 및 리더보드 점수 계산 (반드시 8,640스텝)**:

  ```bash
  python -m smart_buildings.evaluate --policies transfer
  python score_leaderboard.py smart_buildings/results --policy transfer --expect 30
  ```

* **병렬 공식 30개 시나리오 + 6개 기후 다기후 합성 검증 (`--mode official30`)**:

  ```bash
  python smart_buildings_toolkit/scripts/run_experiment.py --mode official30 --extra-test-count 6 --workers 4 --skip-install
  ```

* **Kaggle 노트북에서 실행 (장시간 실험 기본 경로)**:

  * 인증: Kaggle Settings → API → **Generate New Token**으로 발급한 `KGAT_…` 토큰을 시스템 환경변수 `KAGGLE_API_TOKEN`으로 정의합니다(`kaggle` CLI ≥ 1.8.0 필요). legacy `~/.kaggle/kaggle.json`은 선택이며, 토큰 값은 문서·로그·커밋에 남기지 않습니다.

  * 실험용 노트북 정의: `kaggle/smart_buildings_ablation.ipynb`(GitHub 클론 → `pip install -e . --no-deps` → smoke → `--workers 4` 병렬 평가 → 결과 집계)와 `kaggle/kernel-metadata.json`(id `kabonzz/notebookaf13d60ffe`, Internet ON, CPU 4 cores — 시뮬레이터가 CPU 바운드). **최종 대회 제출 시에는 Internet OFF 자급자족 제출 노트북을 별도로 사용합니다.**

  * 업로드·실행(버전이 올라가며 즉시 재실행됨):

    ```bash
    python -m kaggle kernels push -p kaggle/
    ```

  * 상태 확인 / 결과 회수:

    ```bash
    python -m kaggle kernels status kabonzz/notebookaf13d60ffe
    python -m kaggle kernels output kabonzz/notebookaf13d60ffe -p ablation_results/kaggle_run
    ```

  * `kernels output`은 `/kaggle/working` 전체를 받으므로 클론본(`kaggle_run/smart-buildings-competition/`, 약 100MB)이 함께 내려옵니다 — 받은 뒤 삭제하세요. 결과 CSV는 `ablation_results/kaggle_run/ablation_results/`에 있습니다.

  * 노트북 셀은 `--skip-install`을 붙여 러너의 자동 `pip install`을 막습니다(설치는 노트북 셀에서 직접 수행).

* **실험 리포트**: 실험을 마치면 `docs/run<N>.md`에 조건·결과 표·관찰·다음 단계를 정리해 커밋합니다(예: `docs/run1.md`).

* **정적 검사 / 린트**:

  ```bash
  ruff check smart_buildings
  ```

## 6. 에이전트 작업, 제출 및 라이선스 규정 (`Rules §1–§3`)

* **검증 없이 개선 주장 금지**: 모든 정책 변경은 최소 5개 대표 기후 시나리오(8,640스텝) 및 미공개 절차적 테스트(`unseen_extra` / 6개 기후 `unseen_diverse`)에서 MAE 변화를 측정해 기록합니다. 스텝 수가 다른 에피소드(예: 288 vs 8,640)끼리는 절대 점수를 비교하지 마세요.

* **과적합 방지 (Leave-Buildings-Out)**: 개발 건물 6개에 파라미터를 맞출 때는 반드시 건물 단위 교차 검증과 `extra_test_scenarios` / `extra_test_scenarios_diverse`(시드 500+, `minneapolis` 및 6개 기후 포함)에서 일반화 성능을 함께 확인합니다.

* **Kaggle 공식 제출 제약 준수**:
  * **제출 형식**: 오직 **Kaggle Notebook**으로만 제출(`smart_buildings_inference_server.Policy` 상속).
  * **오프라인 및 런타임**: 제출 실행 시 **인터넷 완전 차단**, **CPU/GPU 9시간 이하** 런타임(30개 롤아웃 = 259,200스텝 전체 합산, 최종 재실행 포함)을 여유 있게 통과해야 합니다(목표: 30개 롤아웃 추론 시간 6시간 이하).
  * **경량 자급자족 추론**: 학습에 외부 라이브러리(`torch_geometric`, `d3rlpy`, `optuna`, `cma`, `casadi` 등)를 사용하더라도, **최종 제출 추론 코드는 가능하면 순수 NumPy/PyTorch만으로 오프라인 동작**하도록 분리합니다.

* **외부 데이터·사전학습 모델·도구 (`Rules §2.6`)**: 무료로 공개되어 모든 참가자가 동등하게 접근 가능하거나, 호스트의 합리성 기준(Reasonableness Standard: 최소 비용으로 모든 참가자가 접근 가능)을 충족하는 외부 데이터·사전학습 모델·LLM·AMLT만 사용합니다.

* **라이선스 및 우승자 의무 (`Rules §1.6–§1.7, §2.4–§2.5, §2.8, §3.6`)**:
  * **대회 데이터 사용 제한**: 대회 데이터는 **CC-BY-NC-SA 4.0**(비상업적 대회·연구·교육 목적 전용)이며, 비참가자에게 무단 재배포할 수 없습니다.
  * **우승 소스코드 라이선스**: 우승 제출물 및 생성 코드는 **CC-BY 4.0 및 상업적 이용을 제한하지 않는 OSI 승인 오픈소스 라이선스**로 공개해야 하며, 포함하는 서드파티 오픈소스 코드 역시 상업적 제한이 없어야 합니다(NC 라이선스 코드·가중치 배제).
  * **우승자 산출물 (`Rules §2.8`)**: 수상 시 최종 모델의 **학습 코드·추론 코드**, 재현 가능한 방법론 문서(아키텍처·전처리·손실·하이퍼파라미터·저장소 링크), 컴퓨팅 환경 명세를 기한 내 제출할 수 있도록 정리해 둡니다.

* **생성 파일 정리**: 실험 결과(`ablation_results/`, `smart_buildings/results/`, `smart_buildings/extra_test_scenarios/`, `smart_buildings/extra_test_scenarios_diverse/`)와 대용량 모델 체크포인트는 Git에 커밋하지 않도록 `.gitignore`에 유지합니다.