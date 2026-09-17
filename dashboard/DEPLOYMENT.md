# GEAS 대시보드 실행 및 배포

기존 `GEAS_RL_1`(이전 이름 `GEAS_RL (1)`)의 통합 대시보드입니다.
데이터 품질 관리, 전이모델 선정, 강화학습 및 제어 실험 준비 화면을 포함합니다.

## 로컬 실행

GEAS 저장소 루트에서 실행합니다.

```bash
python -m http.server 8080 --directory dashboard
```

같은 컴퓨터의 브라우저에서 http://localhost:8080/ 을 엽니다.
원격 서버에서는 포트 전달 또는 접근 가능한 서버 주소가 필요합니다.

## 공개 웹사이트

https://jisulee74.github.io/GEAS/

GitHub Pages에는 HTML, CSS, JavaScript, CSV 및 품질 평가 데이터/그림을 배포합니다.
실험 참고 코드, 디버깅 스크립트, 문서는 웹 배포 대상에서 제외합니다.

## 업데이트

이 저장소의 `dashboard/`를 수정한 뒤 `main`에 push하면 GitHub Actions가 자동 재배포합니다.
로컬 파일 저장만으로 GitHub에 업로드되지는 않습니다.

```bash
git add dashboard
git commit -m "Update GEAS dashboard"
git push origin HEAD:main
```

원격 main에 다른 변경이 있으면 먼저 해당 변경을 통합한 뒤 push하세요.

실행 결과: https://github.com/jisulee74/GEAS/actions/workflows/deploy-dashboard.yml
Actions에서 `Run workflow`로 수동 재배포할 수도 있습니다.
