import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import coint
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import warnings
import time
import random

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. 페이지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="Pro Quant Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Pro Quant Pair Trading System")

# ---------------------------------------------------------
# 2. 사이드바 (설정 + 자본금 입력)
# ---------------------------------------------------------
with st.sidebar:
    st.header("🎛️ System Control")
    app_mode = st.radio("Select Mode", ["📡 Live Analysis (실전)", "🔙 Backtest (과거 검증)"])
    st.divider()
    
    st.header("💰 Position Sizing")
    # [NEW] 투자금 입력 창 추가
    total_capital = st.number_input("총 투자금 (KRW)", min_value=1000000, value=10000000, step=1000000, format="%d")
    st.caption("롱/숏 각각 50%씩 배분하여 수량을 계산합니다.")
    st.divider()
    
    st.header("⚙️ Strategy Settings")
    window_size = st.slider("Rolling Window (Days)", 20, 120, 60)
    z_threshold = st.slider("Z-Score Threshold", 1.5, 3.0, 2.0, step=0.1)
    p_cutoff = st.slider("Max P-value", 0.01, 0.20, 0.10)
    
    st.divider()
    
    if app_mode == "🔙 Backtest (과거 검증)":
        st.header("📅 Backtest Period")
        col1, col2 = st.columns(2)
        with col1:
            start_date_input = st.date_input("Start Date", datetime(2023, 1, 1))
        with col2:
            end_date_input = st.date_input("End Date", datetime(2023, 12, 31))
        run_label = "RUN BACKTEST 🔙"
    else:
        run_label = "RUN LIVE ANALYSIS 🚀"
        end_date_input = datetime.now()
        start_date_input = end_date_input - timedelta(days=365)

    run_btn = st.button(run_label, type="primary", use_container_width=True)

# ---------------------------------------------------------
# 3. 데이터 로딩
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_stock_data(start_date, end_date):
    manual_tickers = {
        '005930.KS': '삼성전자', '000660.KS': 'SK하이닉스', '035420.KS': 'NAVER', '035720.KS': '카카오',
        '373220.KS': 'LG에너지솔루션', '006400.KS': '삼성SDI', '051910.KS': 'LG화학', '005490.KS': 'POSCO홀딩스',
        '005380.KS': '현대차', '000270.KS': '기아', '003490.KS': '대한항공', '011200.KS': 'HMM',
        '105560.KS': 'KB금융', '055550.KS': '신한지주', '086790.KS': '하나금융지주', '323410.KS': '카카오뱅크',
        '207940.KS': '삼성바이오로직스', '068270.KS': '셀트리온', '000100.KS': '유한양행', '128940.KS': '한미약품',
        '015760.KS': '한국전력', '033780.KS': 'KT&G', '097950.KS': 'CJ제일제당', '032640.KS': 'LG유플러스',
        '259960.KS': '크래프톤', '009150.KS': '삼성전기', '018260.KS': '삼성SDS', '010130.KS': '고려아연',
        '012330.KS': '현대모비스', '096770.KS': 'SK이노베이션', '011070.KS': 'LG이노텍', '003550.KS': 'LG',
        '032830.KS': '삼성생명', '000810.KS': '삼성화재', '017670.KS': 'SK텔레콤', '030200.KS': 'KT',
        '247540.KQ': '에코프로비엠', '086520.KQ': '에코프로', '196170.KQ': '알테오젠', '028300.KQ': 'HLB'
    }
    
    fetch_start = (pd.to_datetime(start_date) - timedelta(days=150)).strftime('%Y-%m-%d')
    fetch_end = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    
    tickers_list = list(manual_tickers.keys())
    all_data_list = []
    
    status_text = st.status(f"Fetching data ({fetch_start} ~ {fetch_end})...", expanded=True)
    
    chunk_size = 5
    for i in range(0, len(tickers_list), chunk_size):
        chunk = tickers_list[i:i + chunk_size]
        try:
            status_text.write(f"📥 Batch {i//chunk_size + 1} Downloading...")
            df_chunk = yf.download(chunk, start=fetch_start, end=fetch_end, progress=False)['Close']
            
            if isinstance(df_chunk, pd.Series): df_chunk = df_chunk.to_frame(name=chunk[0])
            all_data_list.append(df_chunk)
            time.sleep(random.uniform(0.1, 0.5))
        except: continue
        
    status_text.update(label="Download Complete!", state="complete", expanded=False)
    
    if all_data_list:
        df_final = pd.concat(all_data_list, axis=1)
        df_final = df_final.rename(columns=manual_tickers)
        return df_final.ffill().dropna(axis=1)
    return pd.DataFrame()

# ---------------------------------------------------------
# 4. 분석 엔진
# ---------------------------------------------------------
def analyze_and_backtest(df_prices, window, threshold, p_cutoff, mode, start_date, end_date):
    pairs = []
    cols = df_prices.columns
    
    if mode == "🔙 Backtest (과거 검증)":
        mask = (df_prices.index >= pd.to_datetime(start_date)) & (df_prices.index <= pd.to_datetime(end_date))
        df_analysis = df_prices.loc[mask]
    else:
        df_analysis = df_prices 
        
    if len(df_analysis) < window:
        st.error("데이터 기간이 너무 짧습니다.")
        return pd.DataFrame()

    progress_bar = st.progress(0)
    total_checks = len(cols) * (len(cols) - 1) // 2
    checked = 0
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            stock_a = cols[i]
            stock_b = cols[j]
            
            try:
                score, pvalue, _ = coint(df_analysis[stock_a], df_analysis[stock_b])
                
                if pvalue < p_cutoff:
                    log_a = np.log(df_prices[stock_a])
                    log_b = np.log(df_prices[stock_b])
                    spread = log_a - log_b
                    
                    rolling_mean = spread.rolling(window=window).mean()
                    rolling_std = spread.rolling(window=window).std()
                    rolling_z = (spread - rolling_mean) / rolling_std
                    
                    z_score_period = rolling_z.loc[df_analysis.index]
                    
                    # Backtest Logic
                    positions = np.where(z_score_period < -threshold, 1, 
                                       np.where(z_score_period > threshold, -1, 0))
                    
                    ret_a = df_analysis[stock_a].pct_change().fillna(0)
                    ret_b = df_analysis[stock_b].pct_change().fillna(0)
                    spread_ret = (ret_a - ret_b) * pd.Series(positions).shift(1).fillna(0).values
                    cum_ret = (1 + spread_ret).cumprod() - 1
                    
                    # Live Logic
                    current_z = rolling_z.iloc[-1]
                    corr = df_analysis[stock_a].corr(df_analysis[stock_b])
                    
                    status = "Watch"
                    if current_z < -threshold: status = "Buy A / Sell B"
                    elif current_z > threshold: status = "Sell A / Buy B"
                    
                    # [NEW] 현재 가격 정보 저장 (수량 계산용)
                    price_a_last = df_analysis[stock_a].iloc[-1]
                    price_b_last = df_analysis[stock_b].iloc[-1]

                    pairs.append({
                        'Stock A': stock_a, 'Stock B': stock_b,
                        'Price A': price_a_last, 'Price B': price_b_last,
                        'Corr': corr, 'P-value': pvalue,
                        'Z-Score': current_z, 'Status': status,
                        'Spread': spread, 'Mean': rolling_mean, 'Std': rolling_std,
                        'Final_Ret': cum_ret[-1], 
                        'Cum_Ret_Series': cum_ret,
                        'Positions': pd.Series(positions, index=df_analysis.index),
                        'Analysis_Dates': df_analysis.index
                    })
            except: continue
            
            checked += 1
            if checked % 10 == 0:
                progress_bar.progress(min(checked / total_checks, 1.0))
                
    progress_bar.empty()
    return pd.DataFrame(pairs)

# ---------------------------------------------------------
# 5. 차트 그리기
# ---------------------------------------------------------
def plot_results(row, df_prices, window, threshold, mode):
    sa, sb = row['Stock A'], row['Stock B']
    dates = row['Analysis_Dates']
    
    if mode == "🔙 Backtest (과거 검증)":
        rows = 3
        subplot_titles = (f"Price Action ({sa} vs {sb})", "Strategy Performance", "Spread Z-Score")
        specs = [[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
        row_heights = [0.4, 0.3, 0.3]
    else:
        rows = 2
        subplot_titles = (f"Price Action ({sa} vs {sb})", "Spread Z-Score")
        specs = [[{"secondary_y": False}], [{"secondary_y": False}]]
        row_heights = [0.6, 0.4]

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=subplot_titles, row_heights=row_heights, specs=specs)

    pa = df_prices[sa].loc[dates]
    pb = df_prices[sb].loc[dates]
    pa_norm = (pa / pa.iloc[0]) * 100
    pb_norm = (pb / pb.iloc[0]) * 100
    
    fig.add_trace(go.Scatter(x=dates, y=pa_norm, name=sa, line=dict(color='#1f77b4')), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=pb_norm, name=sb, line=dict(color='#ff7f0e')), row=1, col=1)

    if mode == "🔙 Backtest (과거 검증)":
        cum_ret = row['Cum_Ret_Series'] * 100
        fig.add_trace(go.Scatter(x=dates, y=cum_ret, name='Return (%)', 
                                 line=dict(color='green', width=1.5), fill='tozeroy'), row=2, col=1)

    z_row = 3 if mode == "🔙 Backtest (과거 검증)" else 2
    
    spread = row['Spread']
    z_score = (spread - row['Mean']) / row['Std']
    z_score = z_score.loc[dates]

    fig.add_trace(go.Scatter(x=dates, y=z_score, name='Z-Score', line=dict(color='#9467bd')), row=z_row, col=1)
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", row=z_row, col=1)
    fig.add_hline(y=-threshold, line_dash="dash", line_color="blue", row=z_row, col=1)
    fig.add_hline(y=0, line_color="black", line_width=0.5, row=z_row, col=1)

    fig.update_layout(height=800 if mode == "🔙 Backtest (과거 검증)" else 600, 
                      hovermode="x unified", margin=dict(l=20, r=20, t=30, b=20))
    return fig

# ---------------------------------------------------------
# 6. 메인 실행
# ---------------------------------------------------------
if run_btn:
    df_prices = load_stock_data(start_date_input, end_date_input)
    
    if df_prices.empty:
        st.error("데이터 로드 실패.")
    else:
        results = analyze_and_backtest(df_prices, window_size, z_threshold, p_cutoff, app_mode, start_date_input, end_date_input)
        
        if results.empty:
            st.warning("조건을 만족하는 페어를 찾지 못했습니다.")
        else:
            if app_mode == "🔙 Backtest (과거 검증)":
                st.markdown(f"### 🔙 Backtest Report ({start_date_input} ~ {end_date_input})")
                
                # --- [NEW] 모의 투자 시뮬레이션 결과 ---
                # 백테스팅의 경우 '마지막 날' 기준 가격으로 수량을 계산해봄 (참고용)
                # 실제 백테스팅 수익률은 이미 계산되어 있음 (Final_Ret)
                
                top_performer = results.loc[results['Final_Ret'].idxmax()]
                avg_return = results['Final_Ret'].mean()
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg Return", f"{avg_return*100:.2f}%")
                c2.metric("Best Pair", f"{top_performer['Stock A']} - {top_performer['Stock B']}")
                c3.metric("Best Return", f"{top_performer['Final_Ret']*100:.2f}%")
                
                st.divider()
                st.subheader("🏆 Top Performing Pairs (Detail)")
                
                sorted_res = results.sort_values(by='Final_Ret', ascending=False)
                for idx, row in sorted_res.head(5).iterrows():
                    ret_color = "green" if row['Final_Ret'] > 0 else "red"
                    with st.expander(f"**:{ret_color}[{row['Final_Ret']*100:.2f}%]** | {row['Stock A']} vs {row['Stock B']}", expanded=True if idx==0 else False):
                        st.plotly_chart(plot_results(row, df_prices, window_size, z_threshold, app_mode), use_container_width=True)
                        
            else:
                # 📡 Live Analysis Mode
                st.markdown("### 📡 Live Signal Dashboard")
                
                action_items = results[results['Status'] != 'Watch']
                
                c1, c2 = st.columns(2)
                c1.metric("Analyzed Pairs", f"{len(results)}")
                c2.metric("Active Signals", f"{len(action_items)}", delta="Action Required")
                
                st.divider()
                
                tab1, tab2 = st.tabs(["🔥 Signals (with Qty)", "📋 Watchlist"])
                
                with tab1:
                    if not action_items.empty:
                        st.success(f"💰 총 투자금 {total_capital:,}원을 기준으로 계산된 수량입니다.")
                        
                        for idx, row in action_items.sort_values(by='Z-Score', key=abs, ascending=False).iterrows():
                            # --- [NEW] 수량 계산 로직 (Dollar Neutral) ---
                            # 각 종목에 자본의 50%씩 할당
                            allocation = total_capital / 2
                            qty_a = int(allocation / row['Price A'])
                            qty_b = int(allocation / row['Price B'])
                            
                            val_a = qty_a * row['Price A']
                            val_b = qty_b * row['Price B']
                            
                            status_color = "red" if row['Z-Score'] > 0 else "blue"
                            
                            # 시그널 메시지 생성
                            if "Buy A" in row['Status']:
                                trade_msg = f"🔵 **BUY** {row['Stock A']} **{qty_a:,}주** ({val_a/10000:.0f}만원)  |  🔴 **SELL** {row['Stock B']} **{qty_b:,}주** ({val_b/10000:.0f}만원)"
                            else:
                                trade_msg = f"🔴 **SELL** {row['Stock A']} **{qty_a:,}주** ({val_a/10000:.0f}만원)  |  🔵 **BUY** {row['Stock B']} **{qty_b:,}주** ({val_b/10000:.0f}만원)"

                            with st.expander(f":{status_color}[{row['Status']}] {row['Stock A']} vs {row['Stock B']} (Z: {row['Z-Score']:.2f})", expanded=True):
                                st.markdown(f"### 👉 {trade_msg}")
                                st.plotly_chart(plot_results(row, df_prices, window_size, z_threshold, app_mode), use_container_width=True)
                    else:
                        st.info("현재 진입 신호가 발생한 종목이 없습니다.")
                        
                with tab2:
                    st.dataframe(results[['Stock A', 'Stock B', 'Z-Score', 'P-value', 'Corr']].sort_values('P-value'))
else:
    st.info("👈 사이드바에서 모드를 선택하고 실행해주세요.")
