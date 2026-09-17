# 데이터 품질 관리 패널 전달 묶음

Antigravity는 GEAS3.5 없이 이 폴더와 기존 GEAS_RL 사이트로 패널을 구현할 수 있습니다. 먼저 IMPLEMENTATION_PROMPT.md와 EXPERIMENT_GUIDE.md를 읽으세요.

- data/quality/: 최종 artifacts의 CSV·JSON·PNG 원본. 백업 실험 제외.
- data/validation_results.json, test_results.json: 통합 CSV를 브라우저에서 읽기 쉬운 행 배열로 변환. 값은 원본 문자열을 보존하므로 숫자 필드는 명시적으로 변환하세요.
- data/preprocessing/: 실제 리샘플링·분할 단계 요약. QC 적용 후 결과로 표현하지 마세요.
- reference/: 실험 설명·설정·현재 구현 소스. 소스는 설명 근거이며 실행 패키지가 아닙니다.
- source_manifest.json: 원본 경로·SHA256·파일 크기·자료 가용성.
- file_inventory.json: 문서 및 변환 JSON을 포함한 묶음 파일 무결성 목록.

## 전달
기존 GEAS_RL에 quality_panel_bundle 폴더를 그대로 넣으세요. ZIP은 이 폴더를 포함하므로 GEAS_RL 루트에서 풀면 됩니다. Antigravity에 IMPLEMENTATION_PROMPT.md 내용으로 작업을 요청하세요. 로컬 Python/GPU/모델 가중치 없이 정적 자료를 읽어 화면을 구성합니다.

## 주의
원본 JSON 안의 경로는 학습 환경의 출처 정보이며 웹 URL이 아닙니다. fetch는 이 묶음 내부 상대경로를 사용하세요. 품질 모델 적용 JSON은 모델별 후보 산출물이며 최종 선택 증거가 아닙니다. 원본 artifact_integrity.json은 과거 실행 검사 결과로, 제외된 가중치가 이 묶음에도 있다는 뜻은 아닙니다.
