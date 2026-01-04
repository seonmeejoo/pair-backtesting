import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import coint
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. 한국 시장 특화 페어 그룹 정의
# ---------------------------------------------------------
TICKER_NAMES = {
    '005930.KS': '삼성전자', '005935.KS': '삼성전자우',
    '000660.KS': 'SK하이닉스', '034730.KS': 'SK',
    '005380.KS': '현대차', '005387.KS': '현대차2우B', '000270.KS': '기아', '012330.KS': '현대모비스',
    '051910.KS': 'LG화학', '051911.KS': 'LG화학우', '003550.KS': 'LG', '373220.KS': 'LG에너지솔루션',
    '066570.KS': 'LG전자', '066575.KS': 'LG전자우', '011070.KS': 'LG이노텍',
    '035420.KS': 'NAVER', '035720.KS': '카카오',
    '105560.KS': 'KB금융', '055550.KS': '신한지주', '086790.KS': '하나금융지주', '316140.KS': '우리금융지주',
    '005490.KS': 'POSCO홀딩스', '004020.KS': '현대제철', '003670.KS': '포스코퓨처엠',
    '009540.KS': 'HD한국조선해양', '042660.KS': '한화오션', '010620.KS': 'HD현대중공업',
    '011200.KS': 'HMM', '028670.KS': '팬오션',
    '012450.KS': '한화에어로스페이스', '079550.KS': 'LIG넥스원', '047810.KS': '한국항공우주',
    '042700.KS': '한미반도체', '403870.KQ': 'HPSP', '009150.KS': '삼성전기',
    '247540.KQ': '에코프로비엠', '086520.KQ': '에코프로', '064000.KS': '삼성SDI',
    '000150.KS': '두산', '034020.KS': '두산에너빌리티', '241560.KS': '두산밥캣',
    '^KS11': 'KOSPI'
}

PAIR_GROUPS = {
    "1. 지주사-자회사 (Parent-Child)": [
        ('003550.KS', '051910.KS'), ('003550.KS', '066570.KS'), ('034730.KS', '000660.KS'),
        ('012330.KS', '005380.KS'), ('000150.KS', '034020.KS'), ('000150.KS', '241560.KS')
    ],
    "2. 우선주-보통주 (Arbitrage)": [
        ('005930.KS', '005935.KS'), ('005380.KS', '005387.KS'), ('051910.KS', '051911.KS'), ('066570.KS', '066575.KS')
    ],
    "3. 업종별 라이벌 (Industry Rivals)": [
        ('005930.KS', '000660.KS'), ('035420.KS', '035720.KS'), ('005380.KS', '000270.KS'),
        ('105560.KS', '055550.KS'), ('055550.KS', '316140.KS'), ('005490.KS', '004020.KS'),
        ('009540.KS', '042660.KS'), ('012450.KS', '079550.KS'), ('247540.KQ', '086520.KQ'),
        ('373220.KS', '064000.KS')
    ],
    "4. 밸류체인/소부장 (Supply Chain)": [
        ('005930.KS', '009150.KS'), ('000660.KS', '042700.KS'), ('000660.KS', '403870.KQ'),
        ('005380.KS', '011070.KS')
    ]
}

# ---------------------------------------------------------
# 2. UI 및 설정
# ---------------------------------------------------------
st.set_page_config(page_title="KRX Pair Trading Scanner", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    h1, h2, h3 { color: #3B82F6 !important; }
    div[data-testid="stExpander"] { background-color: #1A1C24; border: 1px solid #2B2D35; }
    .stDataFrame { background-color: #1A1C24; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("🔎 분석 설정")
    selected_group = st.multiselect("분석 그룹 선택", list(PAIR_GROUPS.keys()), default=list(PAIR_GROUPS.keys()))
    window_size = st.slider("Lookback Window (평균 회귀 기간)", 20, 120, 60)
    entry_z = st.slider("진입 Z-Score", 1.5, 3.5, 2.0)
    p_cutoff = st.slider("P-value 임계치 (낮을수록 엄격)", 0.01, 0.10, 0.05)
    
    app_mode = st.radio("모드", ["실시간 감시", "백테스트 (최근 1년)"])
    end_date = datetime.now()
    start_date = end_date - timedelta(days=500) # 충분한 데이터를 위해 기간 확장
    
    run_btn = st.button("🚀 분석 시작", type="primary", use_container_width=True)

# ---------------------------------------------------------
# 3. 분석 함수 (에러 방지 로직 포함)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_market_data(start, end):
    all_tickers = list(TICKER_NAMES.keys())
    # 멀티 인덱스 문제 방지를 위해 ['Close']를 확실히 지정
    data = yf.download(all_tickers, start=start, end=end, progress=False)
    if 'Close' in data:
        return data['Close'].ffill()
    return pd.DataFrame()

def calculate_pair_stats(df, t1, t2, window, entry):
    # MissingDataError 방지: 두 종목 모두 데이터가 있는 날짜만 추출
    pair_df = df[[t1, t2]].dropna()
    
    # 충분한 데이터가 있는지 확인
    if len(pair_df) < window + 10:
        return None

    s1, s2 = pair_df[t1], pair_df[t2]
    
    try:
        # 공적분 테스트
        _, pval, _ = coint(s1, s2)
        
        # 로그 스프레드 및 Z-Score 계산
        log_spread = np.log(s1) - np.log(s2)
        mean = log_spread.rolling(window).mean()
        std = log_spread.rolling(window).std()
        z_series = (log_spread - mean) / std
        
        # NaN 제거 (rolling 초기값)
        z_series = z_series.dropna()
        
        return {
            'pval': pval,
            'z_series': z_series,
            'spread': log_spread,
            'corr': s1.corr(s2),
            'last_z': z_series.iloc[-1],
            'clean_df': pair_df
        }
    except Exception:
        return None

# ---------------------------------------------------------
# 4. 메인 실행부
# ---------------------------------------------------------
if run_btn:
    with st.spinner("데이터를 불러오고 분석 중입니다..."):
        df_prices = load_market_data(start_date, end_date)
        
        if df_prices.empty:
            st.error("데이터를 가져오지 못했습니다. Yahoo Finance 연결을 확인하세요.")
        else:
            results = []
            for g_name in selected_group:
                for t1, t2 in PAIR_GROUPS[g_name]:
                    if t1 in df_prices.columns and t2 in df_prices.columns:
                        stats = calculate_pair_stats(df_prices, t1, t2, window_size, entry_z)
                        
                        if stats and stats['pval'] < p_cutoff:
                            results.append({
                                'Group': g_name,
                                'Pair': f"{TICKER_NAMES[t1]} / {TICKER_NAMES[t2]}",
                                'Z-Score': stats['last_z'],
                                'P-value': stats['pval'],
                                'Correlation': stats['corr'],
                                'stats': stats,
                                't1': t1, 't2': t2
                            })

            if not results:
                st.warning("선택한 그룹 내에 통계적으로 유효한 페어가 없습니다. P-value 임계치를 높여보세요.")
            else:
                # 결과 테이블
                res_df = pd.DataFrame(results).drop(columns=['stats', 't1', 't2'])
                st.subheader("📋 공적분 분석 결과")
                st.dataframe(res_df.sort_values('Z-Score', key=abs, ascending=False).style.format({
                    'Z-Score': '{:.2f}', 'P-value': '{:.4f}', 'Correlation': '{:.2f}'
                }), use_container_width=True)

                st.divider()
                
                # 시그널 시각화
                st.subheader("💡 실시간 트레이딩 시그널")
                signals = [r for r in results if abs(r['Z-Score']) >= entry_z]
                
                if not signals:
                    st.info(f"현재 Z-Score {entry_z}를 초과한 종목이 없습니다. 관망을 권장합니다.")
                else:
                    for sig in signals:
                        with st.expander(f"🎯 [{sig['Group']}] {sig['Pair']} (Z: {sig['Z-Score']:.2f})", expanded=True):
                            c1, c2 = st.columns([3, 1])
                            
                            with c1:
                                # 차트 시각화
                                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)
                                
                                p_df = sig['stats']['clean_df']
                                fig.add_trace(go.Scatter(x=p_df.index, y=p_df[sig['t1']]/p_df[sig['t1']].iloc[0], name=TICKER_NAMES[sig['t1']]), row=1, col=1)
                                fig.add_trace(go.Scatter(x=p_df.index, y=p_df[sig['t2']]/p_df[sig['t2']].iloc[0], name=TICKER_NAMES[sig['t2']]), row=1, col=1)
                                
                                z_vals = sig['stats']['z_series']
                                fig.add_trace(go.Scatter(x=z_vals.index, y=z_vals, name='Z-Score', line=dict(color='#00FFCC')), row=2, col=1)
                                fig.add_hline(y=entry_z, line_dash="dash", line_color="#FF4B4B", row=2, col=1)
                                fig.add_hline(y=-entry_z, line_dash="dash", line_color="#3B82F6", row=2, col=1)
                                
                                fig.update_layout(height=500, template="plotly_dark", margin=dict(t=10, b=10, l=10, r=10))
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with c2:
                                st.metric("현재 Z-Score", f"{sig['Z-Score']:.2f}")
                                if sig['Z-Score'] > 0:
                                    st.error(f"SELL {TICKER_NAMES[sig['t1']]}")
                                    st.success(f"BUY {TICKER_NAMES[sig['t2']]}")
                                    st.write("스프레드가 고평가되었습니다. 수렴을 기대하며 매도/매수 포지션을 잡으세요.")
                                else:
                                    st.success(f"BUY {TICKER_NAMES[sig['t1']]}")
                                    st.error(f"SELL {TICKER_NAMES[sig['t2']]}")
                                    st.write("스프레드가 저평가되었습니다. 반등을 기대하며 매수/매도 포지션을 잡으세요.")
else:
    st.info("👈 왼쪽 사이드바에서 분석 조건을 설정하고 '분석 시작' 버튼을 눌러주세요.")
