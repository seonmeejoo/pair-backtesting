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
# 1. 한국 시장 특화 페어 그룹 정의 (Watchlist)
# ---------------------------------------------------------
# (종목코드: 종목명) 매핑
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
    .stApp { background-color: #111317; color: #E0E0E0; }
    h1, h2, h3 { color: #3B82F6 !important; }
    div[data-testid="stExpander"] { background-color: #1A1C24; border: 1px solid #2B2D35; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("🔎 분석 설정")
    selected_group = st.multiselect("분석할 그룹 선택", list(PAIR_GROUPS.keys()), default=list(PAIR_GROUPS.keys()))
    window_size = st.slider("Lookback Window", 20, 120, 60)
    entry_z = st.slider("진입 Z-Score", 1.5, 3.5, 2.0)
    p_cutoff = st.slider("P-value 임계치", 0.01, 0.10, 0.05)
    
    app_mode = st.radio("모드 선택", ["실시간 감시", "백테스트"])
    if app_mode == "백테스트":
        start_date = st.date_input("시작일", datetime.now() - timedelta(days=365))
        end_date = st.date_input("종료일", datetime.now())
    else:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
    
    run_btn = st.button("🚀 분석 시작", type="primary", use_container_width=True)

# ---------------------------------------------------------
# 3. 데이터 엔진
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_market_data(start, end):
    all_tickers = list(TICKER_NAMES.keys())
    data = yf.download(all_tickers, start=start, end=end, progress=False)['Close']
    return data.ffill().bfill()

def calculate_pair_stats(df, t1, t2, window, entry):
    s1, s2 = df[t1], df[t2]
    # 공적분 테스트
    score, pval, _ = coint(s1, s2)
    
    # 스프레드 및 Z-Score 계산
    log_spread = np.log(s1) - np.log(s2)
    mean = log_spread.rolling(window).mean()
    std = log_spread.rolling(window).std()
    z_score = (log_spread - mean) / std
    
    # 수익률 계산 (단순 롱/숏)
    ret1 = s1.pct_change()
    ret2 = s2.pct_change()
    
    return {
        'pval': pval,
        'z_score': z_score,
        'spread': log_spread,
        'corr': s1.corr(s2),
        'last_z': z_score.iloc[-1]
    }

# ---------------------------------------------------------
# 4. 분석 실행 및 시각화
# ---------------------------------------------------------
if run_btn:
    df_prices = load_market_data(start_date - timedelta(days=200), end_date)
    
    results = []
    for g_name in selected_group:
        for t1, t2 in PAIR_GROUPS[g_name]:
            if t1 in df_prices.columns and t2 in df_prices.columns:
                stats = calculate_pair_stats(df_prices, t1, t2, window_size, entry_z)
                
                if stats['pval'] < p_cutoff:
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
        st.warning("선택한 조건 내에 유효한(공적분 관계인) 페어가 없습니다.")
    else:
        res_df = pd.DataFrame(results).drop(columns=['stats', 't1', 't2'])
        st.subheader("📊 유효 페어 리스트 (P-value 기준 통과)")
        st.dataframe(res_df.sort_values('Z-Score', key=abs, ascending=False), use_container_width=True)

        st.divider()
        st.subheader("🔥 실시간 매매 신호 (Z-Score 임계치 초과)")
        
        signals = [r for r in results if abs(r['Z-Score']) >= entry_z]
        
        if not signals:
            st.info("현재 진입 범위 내에 있는 페어가 없습니다.")
        else:
            for sig in signals:
                with st.expander(f"🎯 [{sig['Group']}] {sig['Pair']} (Z: {sig['Z-Score']:.2d})", expanded=True):
                    # 차트 생성
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.6, 0.4])
                    
                    # 1. 가격 차트 (정규화)
                    p1 = df_prices[sig['t1']].loc[start_date:end_date]
                    p2 = df_prices[sig['t2']].loc[start_date:end_date]
                    fig.add_trace(go.Scatter(x=p1.index, y=p1/p1.iloc[0], name=TICKER_NAMES[sig['t1']]), row=1, col=1)
                    fig.add_trace(go.Scatter(x=p2.index, y=p2/p2.iloc[0], name=TICKER_NAMES[sig['t2']]), row=1, col=1)
                    
                    # 2. Z-Score 차트
                    z_plot = sig['stats']['z_score'].loc[start_date:end_date]
                    fig.add_trace(go.Scatter(x=z_plot.index, y=z_plot, name='Z-Score', line=dict(color='white')), row=2, col=1)
                    fig.add_hline(y=entry_z, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=-entry_z, line_dash="dash", line_color="blue", row=2, col=1)
                    
                    fig.update_layout(height=500, template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    action = "Short A / Long B (스프레드 하락 베팅)" if sig['Z-Score'] > 0 else "Long A / Short B (스프레드 상승 베팅)"
                    st.success(f"**권장 액션:** {action}")

else:
    st.info("왼쪽 사이드바에서 그룹을 선택하고 분석 시작 버튼을 눌러주세요.")
