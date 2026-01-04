import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import coint
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import warnings
import time
import random

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. UI Settings
# ---------------------------------------------------------
st.set_page_config(
    page_title="Pair Trading Scanner",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #1A1C24; color: #E0E0E0; font-family: 'Pretendard', sans-serif; }
    section[data-testid="stSidebar"] { background-color: #111317; border-right: 1px solid #2B2D35; }
    div[data-testid="metric-container"] { background-color: #252830; border: 1px solid #363945; border-radius: 8px; padding: 15px; }
    div.stButton > button { background-color: #3B82F6; color: white; border: none; border-radius: 6px; height: 3em; font-weight: 600; }
    h1, h2, h3 { color: #F3F4F6 !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

st.title("Pair Trading Scanner")
st.markdown("지수(KOSPI) 대비 성과 분석 및 전체 종목 스캐닝")

# ---------------------------------------------------------
# 2. Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("설정 (Settings)")
    universe_mode = st.selectbox("분석 대상 그룹", ["KOSPI 200 (선물/헷지)", "시가총액 상위 100 (Long Only)"])
    st.divider()
    app_mode = st.radio("실행 모드", ["실시간 분석 (Live)", "백테스트 (Backtest)"])
    st.divider()
    total_capital = st.number_input("투자 원금 (KRW)", value=10000000, step=1000000, format="%d")
    
    with st.expander("⚙️ 전략 파라미터"):
        window_size = st.slider("분석 기간 (Window)", 20, 120, 60)
        entry_z = st.slider("진입 기준 (Z-Score)", 1.5, 3.0, 2.0)
        exit_z = st.slider("익절 기준 (Z-Score)", 0.0, 1.0, 0.0)
        stop_loss_z = st.slider("손절 기준 (Z-Score)", 3.0, 6.0, 4.0)
        default_p = 0.05 if "상위 100" in universe_mode else 0.10
        p_cutoff = st.slider("연관성 기준 (P-value)", 0.01, 0.20, default_p)

    st.divider()
    if app_mode == "백테스트 (Backtest)":
        st.subheader("검증 기간")
        c1, c2 = st.columns(2)
        start_input = c1.date_input("시작일", datetime(2025, 1, 1))
        end_input = c2.date_input("종료일", datetime(2025, 12, 31))
        run_label = "백테스트 실행"
    else:
        end_input = datetime.now()
        start_input = end_input - timedelta(days=365)
        run_label = "분석 시작"

    run_btn = st.button(run_label, type="primary", use_container_width=True)

# ---------------------------------------------------------
# 3. Data Loading (Full Ticker List Restored)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_stock_data(universe_type, start_date, end_date):
    # [복구] KOSPI 200 선물 가능 종목
    tickers_futures = {
        '005930.KS': '삼성전자', '000660.KS': 'SK하이닉스', '005380.KS': '현대차', 
        '000270.KS': '기아', '005490.KS': 'POSCO홀딩스', '006400.KS': '삼성SDI', 
        '051910.KS': 'LG화학', '035420.KS': 'NAVER', '035720.KS': '카카오', 
        '105560.KS': 'KB금융', '055550.KS': '신한지주', '086790.KS': '하나금융지주',
        '000810.KS': '삼성화재', '032830.KS': '삼성생명', '015760.KS': '한국전력', 
        '012330.KS': '현대모비스', '009540.KS': 'HD한국조선해양', '042660.KS': '한화오션', 
        '011200.KS': 'HMM', '003490.KS': '대한항공', '030200.KS': 'KT', '017670.KS': 'SK텔레콤',
        '009150.KS': '삼성전기', '011070.KS': 'LG이노텍', '018260.KS': '삼성SDS', 
        '259960.KS': '크래프톤', '036570.KS': '엔씨소프트', '251270.KS': '넷마블', 
        '090430.KS': '아모레퍼시픽', '097950.KS': 'CJ제일제당', '010130.KS': '고려아연', 
        '010950.KS': 'S-Oil', '096770.KS': 'SK이노베이션', '323410.KS': '카카오뱅크', 
        '377300.KS': '카카오페이', '034730.KS': 'SK', '003550.KS': 'LG',
        '247540.KQ': '에코프로비엠', '086520.KQ': '에코프로', '028300.KQ': 'HLB'
    }

    # [복구] 시가총액 상위 추가 종목
    additional = {
        '373220.KS': 'LG에너지솔루션', '207940.KS': '삼성바이오로직스', '068270.KS': '셀트리온', 
        '000100.KS': '유한양행', '128940.KS': '한미약품', '196170.KQ': '알테오젠', 
        '214150.KQ': '클래시스', '145020.KQ': '휴젤', '042700.KS': '한미반도체', 
        '403870.KQ': 'HPSP', '071050.KS': '한국금융지주', '024110.KS': '기업은행', 
        '316140.KS': '우리금융지주', '000120.KS': 'CJ대한통운', '028670.KS': '팬오션',
        '010120.KS': 'LS ELECTRIC', '267250.KS': 'HD현대일렉트릭', '012450.KS': '한화에어로스페이스',
        '047810.KS': '한국항공우주', '079550.KS': 'LIG넥스원', '021240.KS': '코웨이', 
        '033780.KS': 'KT&G', '004370.KS': '농심', '007310.KS': '오뚜기', 
        '271560.KS': '오리온', '280360.KS': '롯데웰푸드', '005940.KS': 'NH투자증권', 
        '016360.KS': '삼성증권', '039490.KS': '키움증권', '001450.KS': '현대해상',
        '000150.KS': '두산', '278280.KQ': '천보', '365550.KS': '성일하이텍'
    }
    
    manual_tickers = {**tickers_futures, **additional} if "상위 100" in universe_type else tickers_futures
    fetch_start = (pd.to_datetime(start_date) - timedelta(days=365)).strftime('%Y-%m-%d')
    fetch_end = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    
    tickers_list = list(manual_tickers.keys())
    
    try:
        # 지수(^KS11)와 종목 데이터 함께 로드
        df_all = yf.download(tickers_list + ['^KS11'], start=fetch_start, end=fetch_end, progress=False)['Close']
        kospi = df_all['^KS11'].rename('KOSPI')
        stocks = df_all.drop(columns=['^KS11']).rename(columns=manual_tickers)
        
        stocks = stocks.ffill().bfill().dropna(axis=1, how='any')
        return stocks, kospi, manual_tickers
    except:
        return pd.DataFrame(), pd.Series(), {}

# ---------------------------------------------------------
# 4. 분석 로직
# ---------------------------------------------------------
def run_analysis(df_prices, window, entry_thresh, exit_thresh, stop_loss, p_cutoff, start, end):
    pairs = []
    cols = df_prices.columns
    target_mask = (df_prices.index >= pd.to_datetime(start)) & (df_prices.index <= pd.to_datetime(end))
    
    prog_bar = st.progress(0, text="종목 간의 통계적 관계를 계산하고 있습니다...")
    checked = 0
    total_checks = len(cols) * (len(cols) - 1) // 2
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            sa, sb = cols[i], cols[j]
            if df_prices[sa].corr(df_prices[sb]) < 0.6: 
                checked += 1
                continue
            try:
                score, pval, _ = coint(df_prices[sa], df_prices[sb])
                if pval < p_cutoff:
                    log_a, log_b = np.log(df_prices[sa]), np.log(df_prices[sb])
                    spread = log_a - log_b
                    mean, std = spread.rolling(window).mean(), spread.rolling(window).std()
                    z_all = (spread - mean) / std
                    z_target = z_all.loc[target_mask]
                    if z_target.empty: continue
                    
                    # Strategy Simulation
                    positions = np.zeros(len(z_target))
                    current_pos = 0 
                    for k in range(len(z_target)):
                        z_val = z_target.iloc[k]
                        if current_pos == 0:
                            if z_val < -entry_thresh: current_pos = 1 
                            elif z_val > entry_thresh: current_pos = -1
                        elif current_pos == 1:
                            if z_val >= -exit_thresh or z_val < -stop_loss: current_pos = 0 
                        elif current_pos == -1:
                            if z_val <= exit_thresh or z_val > stop_loss: current_pos = 0   
                        positions[k] = current_pos

                    ret_a, ret_b = df_prices[sa].loc[target_mask].pct_change().fillna(0), df_prices[sb].loc[target_mask].pct_change().fillna(0)
                    spr_ret = (ret_a - ret_b) * pd.Series(positions, index=z_target.index).shift(1).fillna(0).values
                    
                    pairs.append({
                        'Stock A': sa, 'Stock B': sb, 'Z-Score': z_all.iloc[-1], 'Corr': df_prices[sa].corr(df_prices[sb]),
                        'Final_Ret': (1 + spr_ret).prod() - 1, 'Daily_Ret_Series': pd.Series(spr_ret, index=z_target.index),
                        'Spread': spread, 'Mean': mean, 'Std': std, 'Analysis_Dates': z_target.index,
                        'Price A': df_prices[sa].iloc[-1], 'Price B': df_prices[sb].iloc[-1], 'Status': "Signal"
                    })
            except: pass
            checked += 1
            if checked % 50 == 0: prog_bar.progress(min(checked/total_checks, 1.0))
            
    prog_bar.empty()
    return pd.DataFrame(pairs)

# ---------------------------------------------------------
# 5. 시각화 함수
# ---------------------------------------------------------
def plot_chart(row, df_prices, entry, exit, stop, mode):
    sa, sb = row['Stock A'], row['Stock B']
    dates = row['Analysis_Dates']
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25])
    pa, pb = df_prices[sa].loc[dates], df_prices[sb].loc[dates]
    fig.add_trace(go.Scatter(x=dates, y=(pa/pa.iloc[0])*100, name=sa, line=dict(color='#3B82F6', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=(pb/pb.iloc[0])*100, name=sb, line=dict(color='#F59E0B', width=1.5)), row=1, col=1)
    z_vals = ((row['Spread'] - row['Mean']) / row['Std']).loc[dates]
    fig.add_trace(go.Scatter(x=dates, y=z_vals, name='Z-Score', line=dict(color='#9CA3AF', width=1)), row=2, col=1)
    fig.add_hline(y=entry, line_dash="dot", line_color="#10B981", row=2, col=1); fig.add_hline(y=-entry, line_dash="dot", line_color="#10B981", row=2, col=1)
    if mode == "Backtest":
        cum = row['Cum_Ret_Series'] * 100 if 'Cum_Ret_Series' in row else (1 + row['Daily_Ret_Series']).cumprod() * 100 - 100
        fig.add_trace(go.Scatter(x=dates, y=cum, name='수익률 %', line=dict(color='#10B981', width=1.5), fill='tozeroy'), row=3, col=1)
    fig.update_layout(title=f"{sa} vs {sb} 상세 분석", height=600, template="plotly_dark", margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor='#1A1C24', paper_bgcolor='#1A1C24')
    return fig

# ---------------------------------------------------------
# 6. 메인 실행
# ---------------------------------------------------------
if run_btn:
    with st.spinner("데이터 로딩 및 지수 비교 분석 중..."):
        df_prices, df_kospi, ticker_map = load_stock_data(universe_mode, start_input, end_input)

    if df_prices.empty:
        st.error("데이터 로드 실패")
    else:
        results = run_analysis(df_prices, window_size, entry_z, exit_z, stop_loss_z, p_cutoff, start_input, end_input)
        
        if results.empty:
            st.warning("조건에 맞는 페어가 없습니다. P-value를 높여보세요.")
        elif app_mode == "백테스트 (Backtest)":
            kospi_period = df_kospi.loc[start_input:end_input]
            kospi_ret = (kospi_period / kospi_period.iloc[0]) - 1
            all_ret = pd.DataFrame(index=kospi_period.index)
            for _, row in results.iterrows():
                all_ret[f"{row['Stock A']}-{row['Stock B']}"] = row['Daily_Ret_Series'].reindex(all_ret.index).fillna(0)
            port_daily = all_ret.mean(axis=1)
            port_cum = (1 + port_daily).cumprod() - 1

            st.subheader("📊 전략 vs 시장(KOSPI) 성과 비교")
            c1, c2, c3 = st.columns(3)
            strategy_final = port_cum.iloc[-1] * 100
            kospi_final = kospi_ret.iloc[-1] * 100
            c1.metric("내 전략 수익률", f"{strategy_final:.2f}%", f"{strategy_final - kospi_final:.2f}% vs 지수")
            c2.metric("KOSPI 지수 수익률", f"{kospi_final:.2f}%")
            c3.metric("Alpha (초과수익)", f"{strategy_final - kospi_final:.2f}%p")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum*100, name='내 전략 (Pair Trading)', line=dict(color='#10B981', width=3)))
            fig.add_trace(go.Scatter(x=kospi_ret.index, y=kospi_ret*100, name='시장 지수 (KOSPI Buy & Hold)', line=dict(color='#9CA3AF', width=2, dash='dot')))
            fig.update_layout(title="누적 수익률 비교 차트", yaxis_title="수익률 (%)", hovermode="x unified", template="plotly_dark", height=450, plot_bgcolor='#1A1C24', paper_bgcolor='#1A1C24')
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("🏆 개별 페어 성과 (Top 5)")
            for idx, row in results.sort_values('Final_Ret', ascending=False).head(5).iterrows():
                with st.expander(f"🟢 {row['Stock A']} / {row['Stock B']} (수익률: {row['Final_Ret']*100:.1f}%)"):
                    st.plotly_chart(plot_chart(row, df_prices, entry_z, exit_z, stop_loss_z, "Backtest"), use_container_width=True)
        else:
            # Live 모드 신호 표시 로직 (생략된 경우를 위해 간략히 추가)
            st.subheader("🔥 실시간 매매 신호")
            # ... (이후 Live 모드 결과 표시 코드 추가 가능)
