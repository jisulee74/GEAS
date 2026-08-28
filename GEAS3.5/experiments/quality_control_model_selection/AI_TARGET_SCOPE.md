# AI quality-model target scope

새 HPO 및 임계값 보정은 다음 8개 변수만 사용한다.

- `in_medium_temp1`
- `in_temp`, `in_temp2`
- `in_hum`, `in_hum2`
- `in_medium_hum1`
- `in_co2`, `in_co2_2`

HPO는 작물별로 다시 수행한다. 임계값은 작물 × 모델 × 변수별로 validation reconstruction error의 최소~최대 구간을 100등분해 각 변수 validation F1가 최대인 값을 선택한다.
규칙 기반 탐지는 이 AI 범위와 독립적으로 도메인 규칙 CSV의 16개 센서 컬럼에 적용된다.
