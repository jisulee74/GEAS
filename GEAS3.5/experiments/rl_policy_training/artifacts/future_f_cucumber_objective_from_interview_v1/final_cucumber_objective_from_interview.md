# Cucumber Future-F Objective from Completed Interview

## A. Executive Decision

**CUCUMBER FUTURE-F OBJECTIVE — READY FROM COMPLETED INTERVIEW**

READY는 인터뷰가 뒷받침하는 최소 **목적·의미 계약**의 준비 완료다. 실행 reward나 PPO 승인, 상주 현장 설정값 확인이라는 뜻은 아니다. 직전 `HUMAN CONFIRMATION REQUIRED`와 추가 상주 질문 필수 조건은 이번 프로젝트 결정으로 대체한다. 기존 문서는 보존하되 질문지를 필수 gate로 사용하지 않는다.

## B. Completed Interview Sources Used

사용자가 제공한 `GEAS_농가_인터뷰_Evidence_Summary__오이토마토.pdf` 8쪽 전체, 기존 integration의 CU01–06 registry와 final re-audit, reward_contract_v2를 대조했다. 실제 확보된 원자료는 완료 인터뷰의 구조화 요약본이며, 별도 Clova 전사/음성 파일을 읽었다고 주장하지 않는다. PDF 자체에 발언 locator가 있는 항목만 이를 사용한다. 식별자는 Cucumber-G1로 제한한다.
Cucumber-G1은 순천 외부 상업농가1명, 백다다기계 굿모닝·남부 가을작형·토양관비 맥락이다. 상주 운영자나 농가 consensus가 아니다. 이번 연구에서 이 인터뷰가 오이 목적 범위를 정의하도록 사용자가 결정했다. 토마토 절대습도/조조가온/4°C 설명은 오이 목표로 옮기지 않는다.

## C. What the Cucumber Expert Actually Treats as Important

온도는 생육상태×시간대×계절/시설 한계에 의존한다. RH는 시간대와 날씨에 따라 관리하며 과습·결로 피해를 줄이기 위해 팬/환기/온풍을 조합한다. 정상 난방 ON/OFF보다 환경 유지가 우선이다. 광 확보는 중요하고 차광은 과열 대응이다. 생육상태 중첩, 방제·예방, 비용/설비 현실도 중요하되 모두 scalar climate reward가 되는 것은 아니다.

## D. Objective Components Retained

| 항목 | 인터뷰 분류 | GEAS 역할 / 필요한 관측 |
|---|---|---|
| 조건부 온도관리 | INTERVIEW_SUPPORTED | 핵심 목적; 실내T 및 실제 알려진 context. 당장 자동 scalar reward 아님 |
| 시간대별 RH 관리 | INTERVIEW_SUPPORTED | 핵심 목적; 실내T/RH 함께 해석, 기존 실외T/RH는 맥락 |
| 과습·결로 피해 완화 | INTERVIEW_NOT_NUMERIC | 질적 위험관리·진단; leaf wetness 없으면 실제 결로 발생 확정 금지 |
| 의도적 heat+vent | INTERVIEW_SUPPORTED | 동시 사용만으로 벌점화하지 않는 해석 제약; 명령/상황 기록 |
| 정상 난방 cycling | INTERVIEW_SUPPORTED | generic switching 벌점 제외; sampled command≠장비 내부 cycling |
| 광/차광·생육중첩 | INTERVIEW_SUPPORTED | context/진단. lux tracking이나 stage reward는 제외 |
| 병해충·비용 현실 | INTERVIEW_NOT_NUMERIC | 질적 context/decision support. 정량 disease/energy reward 제외 |

분류는 각 후보에 정확히 하나씩 부여했다(JSON). 병해충과 비용은 ‘중요하지 않다’고 삭제한 것이 아니라 중요성은 보존하고 현재 scalar reward에서 제외한다. 독립적으로 승인된 support/물리/causal 안전 제약은 별도 유지한다.

## E. Numeric vs Qualitative Evidence

| PDF 근거 | 원래 수치/의미 | 보존할 제한 |
|---|---|---|
| p1–2 CU01 초기/활착 | 약23°C | approximate baseline 사례; ±1 같은 band 생성 금지 |
| p2 CU01 비대 | 오전 약30°C까지, 오후 약23°C | 조건부 운영 사례. 현재 오이72가 비대라고 DAT로 단정하지 않음 |
| p2 CU01 겨울 | 최소13–15°C, 일부 새벽13/저녁15 | 최소 설정 사례의 범위; 야간 허용 band로 변환하지 않음 |
| p2 CU03 RH | 오전 약90%, 낮 약60% | 시간대 practice reference; 90%를 severe threshold로 만들지 않음 |
| p2 CU03 과습 | 겨울/강우 시100% 수준 | 불가피한 노출 설명이지 원하는 목표 아님 |
| p3 CU05 광 | 약4만–5.5만lux, 맑은날 약7만lux 관찰 | W/m²인 P0 센서에 수치 대입/단위변환하지 않음 |
| p3 방제 | 약3일 주기 | 작업 관행; 기후 목표/자동 방제 action 아님 |

오전/낮/오후의 정확한 시각, fixed stage 기간, 노출 duration, tolerances는 만들지 않는다. 수치 없는 중요 항목은 질적·조건부·진단으로 충분히 표현할 수 있으므로 새로운 인터뷰 blocker가 아니다.

## F. Components Explicitly Excluded

`NOT_IDENTIFIED_AS_IMPORTANT`: 독립 VPD band, 오이의 독립 절대습도 목표, 모든 항목에 preferred/tolerated/intervention 3종 범위·노출시간·센서선택 규칙을 의무화하는 schema. 현재 목표 scope 밖이며 검색/질문/필수 reward로 되살리지 않는다. growers에게 exact effective time을 요구하는 것도 제거한다; 실제 prospective binding timestamp는 프로젝트가 실행 시 기록할 기술 메타데이터다.
기존 제안의 generic heat+vent conflict 및 switching penalty는 오히려 인터뷰의 조건부 운전 의미에 맞지 않아 제외한다. 실제 kWh/cost, lux 기반 scalar, disease score는 중요 context와 구별하여 정량화를 하지 않는다. 토마토 근권온도·절대습도·조조가온 수치를 오이에 이식하지 않는다.

## G. Resulting Future-F Cucumber Objective Contract

최소 목적은 **조건에 맞는 온도관리 + 시간/외기/설비 맥락을 고려한 RH 관리 및 과습 피해 완화**다. 열과 습도는 함께 해석하며 필요한 난방/환기를 기계적으로 벌점화하지 않는다. 광/생육상태/시설/방제는 위 범위의 context 또는 진단이다.
현재 P0 실내 두 T/RH 채널, 실외T/RH, 일사W/m², 명령 이력을 사용 가능한 관측으로 명시한다. 센서 대표값·rain·실제 결로·운전목적·stage를 만들어 채우지 않는다. 알 수 없는 조건에서는 조건부 숫자 사례를 활성화하지 않고 관측/질적 해석을 유지한다. 이는 추가 질문 gate가 아니라 과잉 정밀도를 피하는 계약이다.
출처 수준은 EXTERNAL DOMAIN-EXPERT PRACTICE EVIDENCE. 상주 실설정/보편 optimum이 아니다. 이번에는 numeric endpoint를 P0에 적용하거나 scalar formula/weight를 정하지 않았다.

## H. P0 Collection Health

collector PID 2616900 생존과 sleeping 상태를 확인했다. 중지/재시작/코드변경 없이 계속 수집한다. 기존 P0 row와 reward-ineligible 표시는 변경하지 않았다.

## I. Remaining True Blockers

**이번 인터뷰 수준의 목적 계약에는 blocker 없음.** 별도 전사 파일 미확보는 근거 fidelity 제한이며 요약에서 회복 가능한 위 의미를 무효화하지 않는다. 정확한 방제 용어·주차 전사 불확실성은 정량 규칙으로 사용하지 않아 본 계약을 막지 않는다.
실행 가능한 평가/reward와 PPO가 완료됐다는 주장은 하지 않는다. 다음 구현에서 이 최소 계약의 계산 가능한 표현과 prospective binding을 정해야 하지만, 인터뷰 밖 농업 숫자나 추가 운영자 답변을 필수화하지 않는다. 기존 독립 안전/배포 상태는 유지한다.

## J. Exact Next Step

**이 최소 오이 계약만 대상으로 prospective 평가 표현·적용 binding을 구체화한다.** 수치가 없는 항목은 질적/진단 또는 scalar 제외를 유지하고, 새 목적·VPD 검색·추가 질문지를 만들지 않는다. 이 작업에서 PPO/reward 실행은 하지 않았다.
