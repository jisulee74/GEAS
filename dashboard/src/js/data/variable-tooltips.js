/**
 * GEAS Dashboard - Detailed Variable Tooltip Definitions (GEAS3.5 Code Grounded)
 * 
 * Code sources:
 * - geas35/features/derived.py
 * - geas35/features/context.py
 * - geas35/core/solar_time.py
 * - geas35/core/growth_stage.py
 * - geas35/rl/mdp_v1.py
 */

export const VARIABLE_TOOLTIPS = {
  // ================= 1행: 시간 조건 =================
  'time_sin_cos': {
    title: '시간 주기 sin/cos 인코딩 (2개)',
    source: 'geas35/features/context.py',
    formula: 'h = 시 + 분/60 + 초/3600\nsin(2πh / 24),  cos(2πh / 24)',
    symbols: 'h: 24시간 실수 시각 | 단위: 무차원 [-1, 1]',
    exceptions: '기본값: 일주기(24시간) 삼각함수 연속 인코딩으로 23:59와 00:00의 시간적 인접성 보존'
  },
  'is_daytime': {
    title: '주간 여부 (1개)',
    source: 'geas35/features/context.py',
    formula: '1[I ≥ 10 W/m²]',
    symbols: 'I: 외부 일사량 (W/m²) | 단위: 이진 0/1 (0: 야간, 1: 주간)',
    exceptions: '기본값: 일사량 임계값 10 W/m². 일사량이 결측인 경우 06:00 이상~18:00 미만을 주간(1)으로 대체 처리.'
  },

  // ================= 2행: 일사·일조 조건 =================
  'solar_cycle': {
    title: '일사 주기: 주간, 초저녁, 심야 (3개)',
    source: 'geas35/core/solar_time.py',
    formula: '1[현재 주기 범주 = 해당 범주] (원핫 인코딩)\n주간: 일출~일몰 | 초저녁: 일몰 후 6시간 | 심야: 나머지 야간',
    symbols: '일출·일몰: 위도/경도/날짜 기반 태양 위치 계산 | 단위: 3개 범주 원핫 이진 0/1',
    exceptions: '기존 유효 분류값 우선. 위치 정보 없거나 실패 시 07:00~18:00 주간, 18:00~24:00 초저녁, 00:00~07:00 심야 대체. ※ 주간 여부(일사 10 W/m² 기준)와 판정 기준이 다름.'
  },
  'sunshine_state': {
    title: '일조 상태: 맑음, 부분 흐림, 흐림, 미확인 (4개)',
    source: 'geas35/core/solar_time.py',
    formula: '일조율(%) = (실제 일조시간 / 가능한 일조시간) × 100\n≥ 70%: 맑음 | ≤ 30%: 흐림 | 30~70%: 부분 흐림 | 부재: 미확인',
    symbols: '가능한 일조시간 = 일몰 − 일출 | 단위: 4개 범주 원핫 이진 0/1',
    exceptions: '기존 날씨 분류값 우선 사용. 관련 입력 부재 시 미확인. ※ 누적 일사 에너지 비율인 SunRatio와 완전히 다른 범주형 변수임.'
  },

  // ================= 2행: 생육 정보 =================
  'growth_stage': {
    title: '생육단계 1–6 one-hot encoding (6개)',
    source: 'geas35/core/growth_stage.py',
    formula: '1[생육단계 = k] (k = 1, 2, ..., 6)',
    symbols: 'k: 생육단계 인덱스 (1~6단계) | 단위: 6개 범주 원핫 이진 0/1',
    exceptions: '작물별 생육단계 규칙표에서 DAT에 해당하는 구간을 자동 조회하여 원핫 변환. MDP 입력 구성에서는 전달받은 생육단계와 DAT 직접 매핑.'
  },
  'dat': {
    title: '정식 후 경과일수 (DAT, 1개)',
    source: 'geas35/core/growth_stage.py',
    formula: 'DAT = 관측 날짜 − 정식 날짜 + 1일',
    symbols: 'DAT: Days After Transplanting | 단위: 정수 (일, days)',
    exceptions: '기본값: 정식 당일은 1일. 정식 이전 날짜는 유효하지 않은 데이터로 오류 처리.'
  },

  // ================= 2행: 목표 온도 정보 =================
  'day_target_temp': {
    title: '주간 목표온도 (°C)',
    source: 'geas35/features/context.py',
    formula: 'T_target,day = 입력 설정값',
    symbols: 'T_target,day: 주간 시간대 관리 목표온도 | 단위: °C',
    exceptions: '기본값: 입력 설정값 우선, 미제공 시 기본 24°C 적용. (센서 측정값이 아닌 온실 관리 설정값)'
  },
  'night_target_temp': {
    title: '야간 목표온도 (°C)',
    source: 'geas35/features/context.py',
    formula: 'T_target,night = 입력 설정값',
    symbols: 'T_target,night: 야간 시간대 관리 목표온도 | 단위: °C',
    exceptions: '기본값: 입력 설정값 우선, 미제공 시 기본 12°C 적용. (센서 측정값이 아닌 온실 관리 설정값)'
  },
  'target_temp_low': {
    title: '현재 목표온도 하한 (°C)',
    source: 'geas35/features/context.py',
    formula: 'T_target,low = 현재 하한 설정값 → 주/야간 하한 설정값 → (T_target − 1°C)',
    symbols: 'T_target,low: 온도 제어 허용 하한값 | 단위: °C',
    exceptions: '기본값: 우선순위 순서대로 조회하며, 별도 설정 없을 시 현재 대표 목표 대비 −1°C (기본 온도 허용폭 ±1°C, 설정 가능)'
  },
  'target_temp_high': {
    title: '현재 목표온도 상한 (°C)',
    source: 'geas35/features/context.py',
    formula: 'T_target,high = 현재 상한 설정값 → 주/야간 상한 설정값 → (T_target + 1°C)',
    symbols: 'T_target,high: 온도 제어 허용 상한값 | 단위: °C',
    exceptions: '기본값: 우선순위 순서대로 조회하며, 별도 설정 없을 시 현재 대표 목표 대비 +1°C (기본 온도 허용폭 ±1°C, 설정 가능)'
  },
  'current_target_temp': {
    title: '현재 대표 목표온도 (°C)',
    source: 'geas35/features/context.py',
    formula: 'T_target = 현재 목표 설정값 → (주간이면 T_target,day, 야간이면 T_target,night)',
    symbols: 'T_target: 실시간 대표 제어 목표온도 | 단위: °C',
    exceptions: '기본값: 현재 목표 설정값 우선. 없으면 주간 여부에 따라 주간(24°C) 또는 야간(12°C) 자동 선택. 난방/냉방도분의 기준점.'
  },

  // ================= 3행: 생리 및 결로 안전 파생변수 =================
  'current_vpd': {
    title: '현재 VPD (Vapor Pressure Deficit, kPa)',
    source: 'geas35/features/derived.py',
    formula: 'es(T) = 0.6108 × exp(17.27T / (T + 237.3))\nVPD = max(es(T) × (1 − clip(RH, 0, 100) / 100), 0)',
    symbols: 'T: 실내온도(°C), RH: 실내상대습도(%), es(T): 포화수증기압(kPa) | 단위: kPa',
    exceptions: '기본값: clip(RH, 0, 100) 및 max(·, 0) 적용으로 음수 VPD 발생 원천 차단. 작물 증산 활동 최적 범위는 통상 0.8~1.2 kPa.'
  },
  'current_dew_point': {
    title: '현재 이슬점 (Dew Point Temperature, °C)',
    source: 'geas35/features/derived.py',
    formula: 'γ = ln(clip(RH, 10⁻⁶, 100) / 100) + 17.62T / (243.12 + T)\nTdew = 243.12γ / (17.62 − γ)',
    symbols: 'T: 실내온도(°C), RH: 실내상대습도(%) | 단위: °C (Magnus 공식)',
    exceptions: '기본값: ln(0) 발산 방지를 위해 RH 하한을 10⁻⁶으로 클리핑. 온실 내 공기가 포화되어 응결이 시작되는 한계 온도.'
  },
  'current_cond_margin': {
    title: '현재 결로 여유 (Condensation Margin, °C)',
    source: 'geas35/features/derived.py',
    formula: 'M = T − Tdew',
    symbols: 'T: 실내온도(°C), Tdew: 이슬점온도(°C) | 단위: °C',
    exceptions: '기본값: 현재 실내온도와 이슬점 간의 간격. 0°C 이하로 떨어지면 엽면 및 피복재에 결로가 발생하여 곰팡이병 발생 위험 급증.'
  },
  'high_humidity_duration': {
    title: '최근 1시간 고습 지속시간',
    source: 'geas35/features/derived.py',
    formula: '5 × Σ 1[RH ≥ 90%]',
    symbols: 'RH: 실내상대습도(%), 최근 60분(12스텝) 대상 | 단위: 분 (min)',
    exceptions: '기본값: 연속 지속시간이 아닌 60분 내 고습 누적시간이며, 관측당 5분 스텝 가산. 최대 60분.'
  },
  'low_vpd_duration': {
    title: '최근 1시간 저VPD 지속시간',
    source: 'geas35/features/derived.py',
    formula: '5 × Σ 1[VPD < 0.5 kPa]',
    symbols: 'VPD: 수증기압포화차(kPa), 최근 60분(12스텝) 대상 | 단위: 분 (min)',
    exceptions: '기본값: 연속 지속시간이 아닌 60분 내 저VPD 누적시간. 관측당 5분 스텝 가산. 엽면 증산 정체 및 병해 위험 대리지표.'
  },
  'pred_cond_margin_10m': {
    title: '10분 후 예측 결로 여유',
    source: 'geas35/features/derived.py',
    formula: '최근 30분 결로 여유 M(τ) = a + bτ (최소제곱 직선 적합) → M(τ=+10분) 예측',
    symbols: 'M: 결로 여유(°C), τ: 시간(분) | 단위: °C',
    exceptions: '기본값: 날짜·시계열 경계 또는 5분 간격 단절을 넘지 않음. 유효 관측 2개 미만 시 현재 결로 여유(M) 그대로 사용.'
  },
  'cond_risk_10m': {
    title: '10분 후 결로 위험 여부',
    source: 'geas35/features/derived.py',
    formula: '1[10분 후 예측 결로 여유 < 0.8°C]',
    symbols: '단위: 이진 0/1 (0: 안전, 1: 위험)',
    exceptions: '기본값: 임계값 기본 0.8°C (설정 가능). 10분 내 온실 표면 결로 위험 사전 경보 플래그.'
  },

  // ================= 3행: 광·에너지 균형 파생변수 =================
  'dli': {
    title: '누적 일사량 (DLI, Daily Light Integral)',
    source: 'geas35/features/derived.py',
    formula: 'DLI = Σ(max(I, 0) × Δt × 2.02) / 10⁶ (당일 누적)',
    symbols: 'I: 외부 일사량(W/m²), Δt: 관측 간격(초) | 단위: mol/m²',
    exceptions: '기본값: 날짜별 0으로 초기화. 외부 일사량을 PPFD(광합성 유효광)로 환산한 추정값이며 2.02 μmol/J는 기본 환산계수. 일사 음수/결측은 0 처리.'
  },
  'solar_acc': {
    title: '측정 일사 누적값',
    source: 'geas35/features/derived.py',
    formula: '제공된 누적값 우선 → 없으면 S = Σ(max(I, 0) × Δt) / 10⁴ (당일 누적)',
    symbols: 'I: 외부 일사량(W/m²), Δt: 관측 간격(초) | 단위: J/cm²',
    exceptions: '기본값: 날짜별 0으로 초기화. 당일 첫 관측 Δt = 0초. 일사 음수·결측은 적분 시 0으로 처리.'
  },
  'clear_sky_solar_acc': {
    title: '청천 일사 누적값',
    source: 'geas35/core/solar_time.py',
    formula: 'μ = max(cos θz, 0),  Iclear = 1098μ × exp(−0.059/μ) (μ>0)\nSc = Σ(Iclear × Δt) / 10⁴ (당일 누적)',
    symbols: 'θz: NOAA 태양 천정각 근사, Iclear: Haurwitz 청천 일사 모델 | 단위: J/cm²',
    exceptions: '기본값: 제공된 청천 누적값 우선. 청천 입력과 위치 정보가 모두 없으면 최종 입력값은 0.'
  },
  'solar_ratio': {
    title: '측정/청천 일사 비율 (SunRatio)',
    source: 'geas35/features/derived.py',
    formula: 'SunRatio = S / Sc',
    symbols: 'S: 측정 일사 누적값, Sc: 청천 일사 누적값 | 단위: 무차원 비율 [0, ∞)',
    exceptions: '기본값: 청천 누적 Sc ≤ 0 또는 계산 불가인 경우 최종 입력값 0. ※ 일조 상태(맑음/흐림)와 구분되는 에너지 누적 비율 지표.'
  },
  'solar_target_eta': {
    title: '목표 일사 도달 예상시간',
    source: 'geas35/features/derived.py',
    formula: '청천 누적곡선에서 목표 누적광에 처음 도달하는 시각 t* 검색\nETA = max((t* − 현재 시각) / 60초, 0)',
    symbols: 't*: 예상 도달 시각 | 단위: 분 (min)',
    exceptions: '기본값: 목표 누적광 기본 1200 J/cm² (입력 설정값 우선). 도달 시각 미발견 또는 계산 불가 시 0 입력되므로 0이 반드시 목표 달성을 의미하지는 않음.'
  },
  'vent_heat_loss_proxy': {
    title: '환기 열손실 proxy',
    source: 'geas35/features/derived.py',
    formula: 'u × max(T − Tout, 0) × [1 − clip(max(I, 0) / 1000, 0, 1)]',
    symbols: 'u: 0~1 정규화 천창 개도율, T, Tout: 실내외 온도(°C), I: 일사량 | 단위: 무차원 대리지표',
    exceptions: '기본값: 1000 W/m²는 기본 일사 기준값. 실제 열손실 전력(W)이 아니라 개도율·온도차·일사 효과를 결합한 환기 열손실 대리지표.'
  },

  // ================= 3행: 물리 안전 및 기구 한계 파생변수 =================
  'temp_diff_in_out': {
    title: '실내외 온도차 (Tin − Tout)',
    source: 'geas35/features/derived.py',
    formula: 'ΔT = T − Tout',
    symbols: 'T: 실내온도(°C), Tout: 외부온도(°C) | 단위: °C',
    exceptions: '기본값: 자연환기 구동력(굴뚝 효과) 및 구조체 열교환 부하의 기본 물리 지표.'
  },
  'day_min_vent': {
    title: '주간 최소 환기율',
    source: 'geas35/features/derived.py',
    formula: 'H = max(고습 누적분, 저VPD 누적분)\nH < 10분: 0.10 | 10 ≤ H < 20분: 0.10×1.2 | H ≥ 20분: 0.10×1.5',
    symbols: 'H: 최근 1시간 습도 스트레스 누적시간(분) | 단위: h⁻¹ (시간당 환기 횟수)',
    exceptions: '기본값: 센서 측정 환기량이 아닌 온실 제어 최소 환기율 정책 설정값. 습도 스트레스가 가중될수록 동적으로 상향 조정됨.'
  },
  'night_min_vent': {
    title: '야간 최소 환기율',
    source: 'geas35/features/derived.py',
    formula: 'H = max(고습 누적분, 저VPD 누적분)\nH < 10분: 0.05 | 10 ≤ H < 20분: 0.05×1.1 | H ≥ 20분: 0.05×1.3',
    symbols: 'H: 최근 1시간 습도 스트레스 누적시간(분) | 단위: h⁻¹ (시간당 환기 횟수)',
    exceptions: '기본값: 센서 측정 환기량이 아닌 야간 최소 환기율 정책값. 야간 보온과 제습 요구를 절충하는 정책 지표.'
  },
  'control_slew_rate_limit': {
    title: '제어 변화 제한값',
    source: 'geas35/features/derived.py',
    formula: 'ramp_limit_pct (설정값 직접 사용)',
    symbols: '단위: 백분율 (%)',
    exceptions: '기본값: 설정값 기본 15. 기구 마모 및 모터 과부하 방지를 위한 제어 변화 제한 설정값. 함수 자체는 시간당 변화율을 계산하지 않으므로 분당/10분당 단위를 붙이지 않음.'
  },
  'hdm': {
    title: '난방도분 (Heating Degree Minutes)',
    source: 'geas35/features/derived.py',
    formula: 'Σ max(Ttarget − T, 0) × Δt분 (당일 누적)',
    symbols: 'Ttarget: 현재 대표 목표온도, T: 실내온도, Δt: 시간차(분) | 단위: °C·min',
    exceptions: '기본값: 날짜별 0 초기화, 첫 행 Δt = 0. 실제 연료 사용량이 아닌 목표온도 대비 저온 편차 적산 지표.'
  },
  'cdm': {
    title: '냉방도분 (Cooling Degree Minutes)',
    source: 'geas35/features/derived.py',
    formula: 'Σ max(T − Ttarget, 0) × Δt분 (당일 누적)',
    symbols: 'Ttarget: 현재 대표 목표온도, T: 실내온도, Δt: 시간차(분) | 단위: °C·min',
    exceptions: '기본값: 날짜별 0 초기화, 첫 행 Δt = 0. 실제 냉방 전력량이 아닌 목표온도 대비 고온 편차 적산 지표.'
  },

  // ================= 4행: 직전 실행 Action (at-1) =================
  'prev_vent_opening': {
    title: '직전 천창 개도율 (%)',
    source: 'geas35/rl/mdp_v1.py',
    formula: '좌·우 천창 유효 개도율 평균 제어 로그를 한 행 이전(t-1)으로 이동',
    symbols: '단위: Continuous 0~100%',
    exceptions: '기본값: 직전 행 부재 시 현재 행 값 유지. 상태 이력(State History)으로 관측 st에 포함.'
  },
  'prev_shade_screen': {
    title: '직전 차광스크린 개도율 (%)',
    source: 'geas35/rl/mdp_v1.py',
    formula: '해당 차광스크린 제어 로그를 한 행 이전(t-1)으로 이동',
    symbols: '단위: Continuous 0~100%',
    exceptions: '기본값: 직전 행 부재 시 현재 행 값 유지. 상태 이력(State History)으로 관측 st에 포함.'
  },
  'prev_thermal_screen': {
    title: '직전 보온스크린 개도율 (%)',
    source: 'geas35/rl/mdp_v1.py',
    formula: '해당 보온스크린 제어 로그를 한 행 이전(t-1)으로 이동',
    symbols: '단위: Continuous 0~100%',
    exceptions: '기본값: 직전 행 부재 시 현재 행 값 유지. 상태 이력(State History)으로 관측 st에 포함.'
  },
  'prev_heater': {
    title: '직전 난방 작동 여부 (0/1)',
    source: 'geas35/rl/mdp_v1.py',
    formula: '직전 행 제어 로그값 > 0.5 이면 1, 아니면 0',
    symbols: '단위: Binary 0/1',
    exceptions: '기본값: 결측값은 0 처리. 상태 이력(State History)으로 관측 st에 포함.'
  },
  'prev_cooler': {
    title: '직전 냉방 작동 여부 (0/1)',
    source: 'geas35/rl/mdp_v1.py',
    formula: '직전 행 제어 로그값 > 0.5 이면 1, 아니면 0',
    symbols: '단위: Binary 0/1',
    exceptions: '기본값: 결측값은 0 처리. 상태 이력(State History)으로 관측 st에 포함.'
  },
  'prev_circ_fan': {
    title: '직전 순환팬 작동 여부 (0/1)',
    source: 'geas35/rl/mdp_v1.py',
    formula: '직전 행 제어 로그값 > 0.5 이면 1, 아니면 0',
    symbols: '단위: Binary 0/1',
    exceptions: '기본값: 결측값은 0 처리. 상태 이력(State History)으로 관측 st에 포함.'
  },

  // ================= 4행: 현재 적용할 Action (at) =================
  'curr_vent_opening': {
    title: '천창 개도율 (Continuous 0~100%)',
    source: 'geas35/rl/mdp_v1.py',
    formula: '좌·우 천창 목표 개도율 명령 평균 (제어 입력 at)',
    symbols: '단위: Continuous 0~100%',
    exceptions: '기본값: 5분 뒤 환경변화를 유도하는 핵심 제어 조건. 센서값에서 계산한 파생변수가 아닌 모델에 입력되는 현재 Action at.'
  },
  'curr_shade_screen': {
    title: '차광스크린 개도율 (Continuous 0~100%)',
    source: 'geas35/rl/mdp_v1.py',
    formula: '차광스크린 목표 개도율 명령값 (제어 입력 at)',
    symbols: '단위: Continuous 0~100%',
    exceptions: '기본값: 센서 파생변수가 아닌 현재 적용 제어 조건 at.'
  },
  'curr_thermal_screen': {
    title: '보온스크린 개도율 (Continuous 0~100%)',
    source: 'geas35/rl/mdp_v1.py',
    formula: '보온스크린 목표 개도율 명령값 (제어 입력 at)',
    symbols: '단위: Continuous 0~100%',
    exceptions: '기본값: 센서 파생변수가 아닌 현재 적용 제어 조건 at.'
  },
  'curr_heater': {
    title: '난방 작동 여부 (Binary 0/1)',
    source: 'geas35/rl/mdp_v1.py',
    formula: '난방 제어 명령값 > 0.5 이면 1, 아니면 0 (제어 입력 at)',
    symbols: '단위: Binary 0/1',
    exceptions: '기본값: 결측 시 0. 센서 파생변수가 아닌 현재 적용 제어 조건 at.'
  },
  'curr_cooler': {
    title: '냉방 작동 여부 (Binary 0/1)',
    source: 'geas35/rl/mdp_v1.py',
    formula: '냉방 제어 명령값 > 0.5 이면 1, 아니면 0 (제어 입력 at)',
    symbols: '단위: Binary 0/1',
    exceptions: '기본값: 결측 시 0. 센서 파생변수가 아닌 현재 적용 제어 조건 at.'
  },
  'curr_circ_fan': {
    title: '순환팬 작동 여부 (Binary 0/1)',
    source: 'geas35/rl/mdp_v1.py',
    formula: '순환팬 제어 명령값 > 0.5 이면 1, 아니면 0 (제어 입력 at)',
    symbols: '단위: Binary 0/1',
    exceptions: '기본값: 결측 시 0. 센서 파생변수가 아닌 현재 적용 제어 조건 at.'
  },

  // ================= 4행: Quality Flag (선택적) =================
  'imputation_flag': {
    title: '결측값 대체 여부 (Imputation Flag, 0/1)',
    source: 'geas35/rl/mdp_v1.py',
    formula: '전처리된 missing_imputed_flag 사용 → 없으면 imputed_flag → outlier_imputed_flag 순서 탐색',
    symbols: '단위: Binary 0/1 (0: 정상 관측, 1: 대체값 적용)',
    exceptions: '기본값: 입력 생성 단계에서 새로 판정하거나 OR로 합산하지 않으며, 해당 열 부재 또는 결측 시 0 처리. With-flags variant에만 상태 관측 st에 추가.'
  },
  'outlier_flag': {
    title: '이상치 탐지 여부 (Outlier Flag, 0/1)',
    source: 'geas35/rl/mdp_v1.py',
    formula: 'outlier_flag 사용 → 없으면 invalid_flag → rule_outlier_flag → ai_outlier_flag 순서 탐색',
    symbols: '단위: Binary 0/1 (0: 정상값, 1: 이상치 탐지)',
    exceptions: '기본값: 입력 생성 단계에서 새로 판정하거나 OR로 합산하지 않으며, 해당 열 부재 또는 결측 시 0 처리. With-flags variant에만 상태 관측 st에 추가.'
  }
};

/**
 * Helper to render an item with tooltip popover
 */
export function renderTooltipLi(key, displayText) {
  const meta = VARIABLE_TOOLTIPS[key];
  if (!meta) {
    return `<li>${displayText}</li>`;
  }

  // Convert linebreaks to <br> for formula
  const formulaHtml = meta.formula.replace(/\n/g, '<br>');
  const symbolsHtml = meta.symbols.replace(/\n/g, '<br>');
  const exceptionsHtml = meta.exceptions.replace(/\n/g, '<br>');

  return `
    <li class="setup-item-with-tooltip" tabindex="0" role="button" aria-haspopup="dialog" aria-expanded="false" data-tooltip-key="${key}">
      <div class="setup-item-row">
        <span class="setup-item-name">${displayText}</span>
        <span class="setup-tooltip-fx" title="계산식 및 생성규칙 툴팁 보기">fx</span>
      </div>
      <div class="setup-tooltip-popover" role="tooltip">
        <div class="setup-tooltip-header">
          <span class="setup-tooltip-title">${meta.title}</span>
          <span class="setup-tooltip-source">${meta.source}</span>
        </div>
        <div class="setup-tooltip-section">
          <div class="setup-tooltip-label">📐 계산식 또는 생성 규칙</div>
          <div class="setup-tooltip-formula">${formulaHtml}</div>
        </div>
        <div class="setup-tooltip-section">
          <div class="setup-tooltip-label">🏷️ 기호 · 단위</div>
          <div class="setup-tooltip-desc">${symbolsHtml}</div>
        </div>
        <div class="setup-tooltip-section">
          <div class="setup-tooltip-label">⚠️ 기본값 및 예외 처리</div>
          <div class="setup-tooltip-desc">${exceptionsHtml}</div>
        </div>
      </div>
    </li>
  `;
}
