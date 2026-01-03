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
# 1. Bloomberg Style Settings
# ---------------------------------------------------------
st.set_page_config(
    page_title="QUANT TERMINAL PRO",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Mode & Bloomberg Colors CSS
st.markdown("""
<style>
    /* 전체 배경 및 폰트 */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    /* 사이드바 */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
    }
    /* 메트릭 카드 (블룸버그 스타일) */
    div[data-testid="metric-container"] {
        background-color: #1E2530;
        border: 1px solid #333;
        padding: 10px;
        border-radius: 4px;
        border-left: 5px solid #FF9800; /* 오렌지 포인트 */
    }
    label {
        color: #FF9800 !important;
        font-weight: bold;
    }
    /* 테이블 헤더 */
    thead tr th:first-child {display:none}
    tbody th {display:none}
    
    h1, h2, h3 { color: #FF9800 !important; font-family: 'Courier New', monospace; }
</style>
""", unsafe_allow_html=True)

st.title("💹 QUANT TERMINAL PRO")

# ---------------------------------------------------------
# 2. Sidebar (Advanced Strategy Controls)
# ---------------------------------------------------------
with st.sidebar:
    st.header("1. UNIVERSE")
    universe_mode = st.selectbox("Target Class", ["Futures/KOSPI200 (Hedge)", "Top 100 Large Cap (Long)"])
    
    st.divider()
    st.header("2. MODE")
    app_mode = st.radio("Operation", ["Live Monitor", "Backtest Sim"])

    st.divider()
    st.header("3. ALGO PARAMS")
    total_capital = st.number_input("Capital (KRW)", value=10000000, step=1000000)
    window_size = st.slider("Lookback Window", 20, 120, 60)
    
    st.subheader("Entry/Exit Rules")
    entry_z = st.slider("Entry Z-Score", 1.5, 3.0, 2.0)
    exit_z = st.slider("Exit Z-Score (Profit)", 0.0, 1.0, 0.0, help="0: Mean Reversion (Recommeded)")
    stop_loss_z = st.slider("Stop Loss Z-Score", 3.0, 6.0, 4.0, help="Force close if spread diverges too much")
    
    default_p = 0.05 if "Large Cap" in universe_mode else 0.10
    p_cutoff = st.slider("Coint P-value Max", 0.01, 0.20, default_p)

    st.divider()
    
    if app_mode == "Backtest Sim":
        st.header("PERIOD")
        c1, c2 = st.columns(2)
        start_input = c1.date_input("Start", datetime(2025, 1, 1))
        end_input = c2.date_input("End", datetime(2025, 12, 31))
        run_label = "INITIATE BACKTEST"
    else:
        end_input = datetime.now()
        start_input = end_input - timedelta(days=365)
        run_label = "SCAN MARKET"

    run_btn = st.button(run_label, type="primary", use_container_width=True)

# ---------------------------------------------------------
# 3. Data Feed
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_stock_data(universe_type, start_date, end_date):
    # Futures / Hedge Universe
    tickers_futures = {
        '005930.KS': 'Samsung Elec', '000660.KS': 'SK Hynix', '005380.KS': 'Hyundai Motor', 
        '000270.KS': 'Kia', '005490.KS': 'POSCO Holdings', '006400.KS': 'Samsung SDI', 
        '051910.KS': 'LG Chem', '035420.KS': 'NAVER', '035720.KS': 'Kakao', 
        '105560.KS': 'KB Financial', '055550.KS': 'Shinhan FG', '086790.KS': 'Hana FG',
        '000810.KS': 'Samsung Fire', '032830.KS': 'Samsung Life', '015760.KS': 'KEPCO', 
        '012330.KS': 'Hyundai Mobis', '009540.KS': 'HD KSOE', '042660.KS': 'Hanwha Ocean', 
        '011200.KS': 'HMM', '003490.KS': 'Korean Air', '030200.KS': 'KT', '017670.KS': 'SK Telecom',
        '009150.KS': 'Samsung Electro', '011070.KS': 'LG Innotek', '018260.KS': 'Samsung SDS', 
        '259960.KS': 'Krafton', '036570.KS': 'NCSoft', '251270.KS': 'Netmarble', 
        '090430.KS': 'Amorepacific', '097950.KS': 'CJ CheilJedang', '010130.KS': 'Korea Zinc', 
        '010950.KS': 'S-Oil', '096770.KS': 'SK Innovation', '323410.KS': 'KakaoBank', 
        '377300.KS': 'KakaoPay', '034730.KS': 'SK', '003550.KS': 'LG',
        '247540.KQ': 'Ecopro BM', '086520.KQ': 'Ecopro', '028300.KQ': 'HLB'
    }

    # Large Cap Universe
    tickers_massive = tickers_futures.copy()
    additional = {
        '373220.KS': 'LG Energy Sol', '207940.KS': 'Samsung Bio', '068270.KS': 'Celltrion', 
        '000100.KS': 'Yuhan', '128940.KS': 'Hanmi Pharm', '196170.KQ': 'Alteogen', 
        '214150.KQ': 'Classys', '145020.KQ': 'Hugel', '042700.KS': 'Hanmi Semi', 
        '403870.KQ': 'HPSP', '071050.KS': 'Korea Inv', '024110.KS': 'IBK', 
        '316140.KS': 'Woori FG', '000120.KS': 'CJ Logistics', '028670.KS': 'Pan Ocean',
        '010120.KS': 'LS ELECTRIC', '267250.KS': 'HD Hyundai Electric', '012450.KS': 'Hanwha Aero',
        '047810.KS': 'KAI', '079550.KS': 'LIG Nex1', '021240.KS': 'Coway', 
        '033780.KS': 'KT&G', '004370.KS': 'Nongshim', '007310.KS': 'Ottogi', 
        '271560.KS': 'Orion', '280360.KS': 'Lotte Wellfood', '005940.KS': 'NH Inv Sec', 
        '016360.KS': 'Samsung Sec', '039490.KS': 'Kiwoom Sec', '001450.KS': 'Hyundai Marine',
        '000150.KS': 'Doosan', '278280.KQ': 'Chunbo', '365550.KS': 'SungEel HiTech'
    }
    
    manual_tickers = {**tickers_massive, **additional} if "Large Cap" in universe_type else tickers_futures

    fetch_start = (pd.to_datetime(start_date) - timedelta(days=365)).strftime('%Y-%m-%d')
    fetch_end = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    
    tickers_list = list(manual_tickers.keys())
    all_data_list = []
    
    st_msg = st.status(f"📡 Establishing Data Feed ({len(tickers_list)} tickers)...", expanded=True)
    
    chunk_size = 5
    for i in range(0, len(tickers_list), chunk_size):
        chunk = tickers_list[i:i + chunk_size]
        try:
            st_msg.write(f"📥 Packet {i//chunk_size + 1} Received...")
            df_chunk = yf.download(chunk, start=fetch_start, end=fetch_end, progress=False)['Close']
            if isinstance(df_chunk, pd.Series): df_chunk = df_chunk.to_frame(name=chunk[0])
            all_data_list.append(df_chunk)
            time.sleep(random.uniform(0.05, 0.2))
        except: continue
        
    st_msg.update(label="Data Feed Connected.", state="complete", expanded=False)
    
    if all_data_list:
        df_final = pd.concat(all_data_list, axis=1)
        df_final = df_final.rename(columns=manual_tickers)
        df_final = df_final.ffill().bfill().dropna(axis=1, how='any')
        return df_final, manual_tickers
    
    return pd.DataFrame(), manual_tickers

# ---------------------------------------------------------
# 4. Advanced Analysis Logic (Stop Loss + Profit Taking)
# ---------------------------------------------------------
def run_analysis(df_prices, window, entry_thresh, exit_thresh, stop_loss, p_cutoff, mode, start, end):
    pairs = []
    cols = df_prices.columns
    
    if len(df_prices) < window: return pd.DataFrame()

    total_checks = len(cols) * (len(cols) - 1) // 2
    target_mask = (df_prices.index >= pd.to_datetime(start)) & (df_prices.index <= pd.to_datetime(end))
    
    prog_bar = st.progress(0)
    checked = 0
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            sa, sb = cols[i], cols[j]
            
            # 1. Correlation Filter (Optimization)
            if df_prices[sa].corr(df_prices[sb]) < 0.6: # stricter filter
                checked += 1
                continue
                
            try:
                # 2. Cointegration
                score, pval, _ = coint(df_prices[sa], df_prices[sb])
                
                if pval < p_cutoff:
                    log_a, log_b = np.log(df_prices[sa]), np.log(df_prices[sb])
                    spread = log_a - log_b
                    
                    mean = spread.rolling(window).mean()
                    std = spread.rolling(window).std()
                    z_all = (spread - mean) / std
                    z_target = z_all.loc[target_mask]
                    
                    if z_target.empty: continue
                    
                    # --- [Advanced Strategy Logic] ---
                    # Vectorized logic is hard for path-dependent strategies (Stop Loss),
                    # so we simulate state-machine for the target period.
                    
                    positions = np.zeros(len(z_target))
                    current_pos = 0 # 0: Neutral, 1: Long Spread, -1: Short Spread
                    
                    for k in range(len(z_target)):
                        z_val = z_target.iloc[k]
                        
                        if current_pos == 0:
                            # Entry
                            if z_val < -entry_thresh: current_pos = 1  # Buy A / Sell B
                            elif z_val > entry_thresh: current_pos = -1 # Sell A / Buy B
                        
                        elif current_pos == 1:
                            # Exit Long Spread
                            # 1. Take Profit (Mean Reversion)
                            if z_val >= -exit_thresh: current_pos = 0
                            # 2. Stop Loss (Divergence)
                            elif z_val < -stop_loss: current_pos = 0
                        
                        elif current_pos == -1:
                            # Exit Short Spread
                            # 1. Take Profit
                            if z_val <= exit_thresh: current_pos = 0
                            # 2. Stop Loss
                            elif z_val > stop_loss: current_pos = 0
                        
                        positions[k] = current_pos

                    # Calculate Returns
                    ret_a = df_prices[sa].loc[target_mask].pct_change().fillna(0)
                    ret_b = df_prices[sb].loc[target_mask].pct_change().fillna(0)
                    
                    # Spread Return = (RetA - RetB) * Position(t-1)
                    spr_ret = (ret_a - ret_b) * pd.Series(positions, index=z_target.index).shift(1).fillna(0).values
                    cum_ret = (1 + spr_ret).cumprod() - 1
                    
                    # Calculate Stats
                    total_trades = np.abs(np.diff(positions)).sum() / 2
                    win_rate = 0 # Simple proxy
                    sharpe = np.mean(spr_ret) / (np.std(spr_ret) + 1e-9) * np.sqrt(252)
                    
                    curr_z = z_all.iloc[-1]
                    status = "Watch"
                    if curr_z < -entry_thresh: status = "Buy A"
                    elif curr_z > entry_thresh: status = "Buy B"
                    
                    pairs.append({
                        'Stock A': sa, 'Stock B': sb,
                        'P-value': pval, 'Z-Score': curr_z,
                        'Status': status, 'Final_Ret': cum_ret[-1],
                        'Sharpe': sharpe, 'Trades': total_trades,
                        'Spread': spread, 'Mean': mean, 'Std': std,
                        'Cum_Ret_Series': cum_ret, 
                        'Daily_Ret_Series': pd.Series(spr_ret, index=z_target.index),
                        'Analysis_Dates': z_target.index,
                        'Price A': df_prices[sa].iloc[-1], 'Price B': df_prices[sb].iloc[-1]
                    })
            except: pass
            
            checked += 1
            if checked % 20 == 0: prog_bar.progress(min(checked / total_checks, 1.0))
            
    prog_bar.empty()
    return pd.DataFrame(pairs)

# ---------------------------------------------------------
# 5. Bloomberg Charts (Heatmap & Performance)
# ---------------------------------------------------------
def plot_heatmap(results):
    if results.empty: return None
    
    # Pivot for correlation matrix visualization of Top Z-scores
    top_pairs = results.sort_values(by='Z-Score', key=abs, ascending=False).head(10)
    
    data = []
    for idx, row in top_pairs.iterrows():
        data.append({'Pair': f"{row['Stock A']}/{row['Stock B']}", 'Z-Score': row['Z-Score'], 'Sharpe': row['Sharpe']})
    
    df_heat = pd.DataFrame(data)
    
    fig = go.Figure(data=go.Heatmap(
        z=df_heat['Z-Score'],
        x=df_heat['Pair'],
        y=['Z-Score Strength'],
        colorscale='RdBu_r',
        zmid=0
    ))
    fig.update_layout(title="Top Pairs Z-Score Heatmap", height=300, template="plotly_dark")
    return fig

def plot_pro_chart(row, df_prices, entry, exit, stop, mode):
    sa, sb = row['Stock A'], row['Stock B']
    dates = row['Analysis_Dates']
    
    # Bloomberg Style Layout
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.5, 0.25, 0.25])
    
    # 1. Normalized Price
    pa, pb = df_prices[sa].loc[dates], df_prices[sb].loc[dates]
    fig.add_trace(go.Scatter(x=dates, y=(pa/pa.iloc[0]-1)*100, name=sa, line=dict(color='#00E5FF', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=(pb/pb.iloc[0]-1)*100, name=sb, line=dict(color='#FF4081', width=1)), row=1, col=1)
    
    # 2. Z-Score with Bands
    z_vals = ((row['Spread'] - row['Mean']) / row['Std']).loc[dates]
    fig.add_trace(go.Scatter(x=dates, y=z_vals, name='Z-Score', line=dict(color='#FF9800', width=1.5)), row=2, col=1)
    
    # Threshold Lines
    fig.add_hline(y=entry, line_dash="dot", line_color="green", row=2, col=1)
    fig.add_hline(y=-entry, line_dash="dot", line_color="green", row=2, col=1)
    fig.add_hline(y=stop, line_color="red", row=2, col=1) # Stop Loss
    fig.add_hline(y=-stop, line_color="red", row=2, col=1)
    fig.add_hline(y=0, line_color="gray", row=2, col=1) # Mean Exit

    # 3. PnL
    if mode == "Backtest Sim":
        cum = row['Cum_Ret_Series'] * 100
        fig.add_trace(go.Scatter(x=dates, y=cum, name='PnL %', line=dict(color='#00C853', width=1), fill='tozeroy'), row=3, col=1)

    fig.update_layout(height=600, hovermode="x unified", template="plotly_dark", 
                      margin=dict(l=10, r=10, t=30, b=10), showlegend=True)
    return fig

# ---------------------------------------------------------
# 6. Main Execution
# ---------------------------------------------------------
if run_btn:
    with st.spinner("Initializing Quantitative Engine..."):
        df_prices, ticker_map = load_stock_data(universe_mode, start_input, end_input)
        name_to_code = {v: k for k, v in ticker_map.items()}

    if df_prices.empty or len(df_prices.columns) < 2:
        st.error("Feed Error. Insufficient Data.")
    else:
        results = run_analysis(df_prices, window_size, entry_z, exit_z, stop_loss_z, p_cutoff, app_mode, start_input, end_input)
        
        def fmt(name): return f"{name} ({name_to_code.get(name, '')})"

        if results.empty:
            st.warning("No opportunities found matching risk profile.")
        else:
            if app_mode == "Backtest Sim":
                # --- Portfolio Dashboard ---
                st.subheader("📊 PORTFOLIO SIMULATION")
                
                # Portfolio Calc
                all_ret = pd.DataFrame(index=pd.date_range(start=start_input, end=end_input))
                for _, row in results.iterrows():
                    s = row['Daily_Ret_Series']
                    s.index = pd.to_datetime(s.index)
                    s = s[~s.index.duplicated(keep='first')].reindex(all_ret.index).fillna(0)
                    all_ret[f"{row['Stock A']}-{row['Stock B']}"] = s
                
                port_daily = all_ret.mean(axis=1).fillna(0)
                port_cum = (1 + port_daily).cumprod() - 1
                
                # Stats
                total_ret = port_cum.iloc[-1]
                sharpe = np.mean(port_daily) / (np.std(port_daily) + 1e-9) * np.sqrt(252)
                
                wealth = (1 + port_daily).cumprod()
                mdd = ((wealth - wealth.expanding().max()) / wealth.expanding().max()).min()

                # KPI Row
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total Return", f"{total_ret*100:.2f}%")
                k2.metric("Sharpe Ratio", f"{sharpe:.2f}")
                k3.metric("Max Drawdown", f"{mdd*100:.2f}%")
                k4.metric("Active Pairs", f"{len(results)}")
                
                # Equity Curve
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(x=port_cum.index, y=port_cum*100, mode='lines', name='Equity', line=dict(color='#00C853')))
                fig_eq.update_layout(title="Equity Curve", template="plotly_dark", height=350)
                st.plotly_chart(fig_eq, use_container_width=True)
                
                # Correlation Heatmap
                st.plotly_chart(plot_heatmap(results), use_container_width=True)

                st.markdown("---")
                st.subheader("🏆 TOP PERFORMERS")
                for idx, row in results.sort_values('Final_Ret', ascending=False).head(5).iterrows():
                    with st.expander(f"{fmt(row['Stock A'])} / {fmt(row['Stock B'])} | Ret: {row['Final_Ret']*100:.2f}% | Sharpe: {row['Sharpe']:.2f}"):
                        st.plotly_chart(plot_pro_chart(row, df_prices, entry_z, exit_z, stop_loss_z, app_mode), use_container_width=True)

            else:
                # --- Live Monitor ---
                actives = results[results['Status'] != 'Watch']
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader("📡 MARKET SCANNER")
                with col2:
                    st.metric("Actionable Signals", f"{len(actives)}")
                
                tab1, tab2 = st.tabs(["🔥 SIGNALS", "👀 WATCHLIST"])
                
                with tab1:
                    if not actives.empty:
                        for idx, row in actives.sort_values(by='Z-Score', key=abs, ascending=False).iterrows():
                            alloc = total_capital / 2
                            qa = int(alloc / row['Price A'])
                            qb = int(alloc / row['Price B'])
                            sa_fmt, sb_fmt = fmt(row['Stock A']), fmt(row['Stock B'])
                            
                            is_long_a = row['Status'] == "Buy A"
                            color = "green" if is_long_a else "red"
                            
                            with st.expander(f"SIGNAL: {sa_fmt} vs {sb_fmt} (Z: {row['Z-Score']:.2f})", expanded=True):
                                c1, c2 = st.columns(2)
                                if is_long_a:
                                    c1.success(f"BUY {qa:,} shs")
                                    c2.error(f"SELL {qb:,} shs")
                                else:
                                    c1.error(f"SELL {qa:,} shs")
                                    c2.success(f"BUY {qb:,} shs")
                                
                                st.plotly_chart(plot_pro_chart(row, df_prices, entry_z, exit_z, stop_loss_z, app_mode), use_container_width=True)
                    else:
                        st.info("No divergence signals detected.")
                
                with tab2:
                    # Heatmap for Watchlist
                    st.plotly_chart(plot_heatmap(results), use_container_width=True)
                    
                    df_disp = results[['Stock A', 'Stock B', 'Z-Score', 'P-value', 'Status']].copy()
                    df_disp['Stock A'] = df_disp['Stock A'].apply(fmt)
                    df_disp['Stock B'] = df_disp['Stock B'].apply(fmt)
                    st.dataframe(df_disp.sort_values('Z-Score', key=abs, ascending=False), use_container_width=True)

else:
    st.info("Awaiting Input... Configure parameters and press RUN.")
