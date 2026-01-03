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
# 1. Page Configuration & Professional CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Pairs Trading System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #f9f9f9;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
    }
    @media (prefers-color-scheme: dark) {
        div[data-testid="metric-container"] {
            background-color: #262730;
            border: 1px solid #41424b;
        }
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Pairs Trading Strategy Dashboard")

# ---------------------------------------------------------
# 2. Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    
    universe_mode = st.selectbox(
        "Target Universe",
        ["Futures / Hedge (KOSPI 200)", "Large Cap (Top 100)"]
    )
    
    st.divider()
    
    app_mode = st.radio("Analysis Mode", ["Live Analysis", "Backtest"])

    st.divider()
    
    st.subheader("Parameters")
    total_capital = st.number_input("Capital (KRW)", value=10000000, step=1000000, format="%d")
    window_size = st.slider("Rolling Window", 20, 120, 60)
    z_threshold = st.slider("Z-Score Threshold", 1.5, 3.0, 2.0)
    
    default_p = 0.05 if "Large Cap" in universe_mode else 0.10
    p_cutoff = st.slider("Max P-value", 0.01, 0.20, default_p)

    st.divider()
    
    if app_mode == "Backtest":
        st.subheader("Period")
        c1, c2 = st.columns(2)
        start_input = c1.date_input("Start", datetime(2025, 1, 1))
        end_input = c2.date_input("End", datetime(2025, 12, 31))
        run_label = "Run Backtest"
    else:
        end_input = datetime.now()
        start_input = end_input - timedelta(days=365)
        run_label = "Run Analysis"

    run_btn = st.button(run_label, type="primary", use_container_width=True)

# ---------------------------------------------------------
# 3. Data Loading (Return Ticker Map)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_stock_data(universe_type, start_date, end_date):
    # KOSPI 200 선물 가능 종목
    tickers_futures = {
        '005930.KS': 'Samsung Elec', '000660.KS': 'SK Hynix', '005380.KS': 'Hyundai Motor', 
        '000270.KS': 'Kia', '005490.KS': 'POSCO Holdings', '006400.KS': 'Samsung SDI', 
        '051910.KS': 'LG Chem', '035420.KS': 'NAVER', '035720.KS': 'Kakao', 
        '105560.KS': 'KB Financial', '055550.KS': 'Shinhan FG', '086790.KS': 'Hana FG',
        '000810.KS': 'Samsung Fire', '032830.KS': 'Samsung Life', '015760.KS': 'KEPCO', 
        '034020.KS': 'Doosan Enerbility', '012330.KS': 'Hyundai Mobis', '009540.KS': 'HD KSOE', 
        '010140.KS': 'Samsung Heavy', '042660.KS': 'Hanwha Ocean', '011200.KS': 'HMM', 
        '003490.KS': 'Korean Air', '030200.KS': 'KT', '017670.KS': 'SK Telecom',
        '009150.KS': 'Samsung Electro', '011070.KS': 'LG Innotek', '018260.KS': 'Samsung SDS', 
        '259960.KS': 'Krafton', '036570.KS': 'NCSoft', '251270.KS': 'Netmarble', 
        '090430.KS': 'Amorepacific', '097950.KS': 'CJ CheilJedang', '010130.KS': 'Korea Zinc', 
        '004020.KS': 'Hyundai Steel', '010950.KS': 'S-Oil', '096770.KS': 'SK Innovation',
        '323410.KS': 'KakaoBank', '377300.KS': 'KakaoPay', '034730.KS': 'SK', '003550.KS': 'LG',
        '247540.KQ': 'Ecopro BM', '086520.KQ': 'Ecopro', '028300.KQ': 'HLB', 
        '293490.KQ': 'KakaoGames', '066970.KQ': 'L&F', '035900.KQ': 'JYP Ent.', 
        '041510.KQ': 'SM Ent.', '263750.KQ': 'PearlAbyss'
    }

    # 대규모 탐색용 추가 종목
    tickers_massive = tickers_futures.copy()
    additional = {
        '373220.KS': 'LG Energy Sol', '207940.KS': 'Samsung BioLogics', '068270.KS': 'Celltrion', 
        '000100.KS': 'Yuhan', '128940.KS': 'Hanmi Pharm', '196170.KQ': 'Alteogen', 
        '214150.KQ': 'Classys', '145020.KQ': 'Hugel', '042700.KS': 'Hanmi Semi', 
        '403870.KQ': 'HPSP', '071050.KS': 'Korea Inv', '024110.KS': 'IBK', 
        '316140.KS': 'Woori FG', '000120.KS': 'CJ Logistics', '028670.KS': 'Pan Ocean',
        '010120.KS': 'LS ELECTRIC', '267250.KS': 'HD Hyundai Electric', '012450.KS': 'Hanwha Aero',
        '047810.KS': 'KAI', '079550.KS': 'LIG Nex1', '021240.KS': 'Coway', 
        '033780.KS': 'KT&G', '004370.KS': 'Nongshim', '007310.KS': 'Ottogi', 
        '271560.KS': 'Orion', '280360.KS': 'Lotte Wellfood', '005940.KS': 'NH Inv Sec', 
        '016360.KS': 'Samsung Sec', '039490.KS': 'Kiwoom Sec', '001450.KS': 'Hyundai Marine',
        '000150.KS': 'Doosan', '278280.KQ': 'Chunbo', '365550.KS': 'SungEel HiTech', 
        '137400.KQ': 'PNT'
    }
    
    if "Large Cap" in universe_type:
        manual_tickers = {**tickers_massive, **additional}
    else:
        manual_tickers = tickers_futures

    fetch_start = (pd.to_datetime(start_date) - timedelta(days=365)).strftime('%Y-%m-%d')
    fetch_end = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    
    tickers_list = list(manual_tickers.keys())
    all_data_list = []
    
    status_placeholder = st.empty()
    status_placeholder.info(f"Downloading data for {len(tickers_list)} stocks...")
    
    chunk_size = 5
    for i in range(0, len(tickers_list), chunk_size):
        chunk = tickers_list[i:i + chunk_size]
        try:
            df_chunk = yf.download(chunk, start=fetch_start, end=fetch_end, progress=False)['Close']
            if isinstance(df_chunk, pd.Series): df_chunk = df_chunk.to_frame(name=chunk[0])
            all_data_list.append(df_chunk)
            time.sleep(random.uniform(0.1, 0.3))
        except: continue
        
    status_placeholder.empty()
    
    if all_data_list:
        df_final = pd.concat(all_data_list, axis=1)
        df_final = df_final.rename(columns=manual_tickers)
        df_final = df_final.ffill().bfill().dropna(axis=1, how='any')
        
        # [NEW] Return both DataFrame and Ticker Map (for code lookup)
        return df_final, manual_tickers
    
    return pd.DataFrame(), manual_tickers

# ---------------------------------------------------------
# 4. Analysis Engine
# ---------------------------------------------------------
def run_analysis(df_prices, window, threshold, p_cutoff, mode, start, end):
    pairs = []
    cols = df_prices.columns
    
    if len(df_prices) < window: return pd.DataFrame()

    total_checks = len(cols) * (len(cols) - 1) // 2
    target_mask = (df_prices.index >= pd.to_datetime(start)) & (df_prices.index <= pd.to_datetime(end))
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            sa, sb = cols[i], cols[j]
            
            if df_prices[sa].corr(df_prices[sb]) < 0.5: continue
                
            try:
                score, pval, _ = coint(df_prices[sa], df_prices[sb])
                
                if pval < p_cutoff:
                    log_a, log_b = np.log(df_prices[sa]), np.log(df_prices[sb])
                    spread = log_a - log_b
                    
                    mean = spread.rolling(window).mean()
                    std = spread.rolling(window).std()
                    z_all = (spread - mean) / std
                    
                    z_target = z_all.loc[target_mask]
                    if z_target.empty: continue
                    
                    pos = np.where(z_target < -threshold, 1, np.where(z_target > threshold, -1, 0))
                    
                    ret_a = df_prices[sa].loc[target_mask].pct_change().fillna(0)
                    ret_b = df_prices[sb].loc[target_mask].pct_change().fillna(0)
                    
                    spr_ret = (ret_a - ret_b) * pd.Series(pos).shift(1).fillna(0).values
                    cum_ret = (1 + spr_ret).cumprod() - 1
                    
                    curr_z = z_all.iloc[-1]
                    status = "Watch"
                    if curr_z < -threshold: status = "Buy A"
                    elif curr_z > threshold: status = "Buy B"
                    
                    pairs.append({
                        'Stock A': sa, 'Stock B': sb,
                        'P-value': pval, 'Z-Score': curr_z,
                        'Status': status, 'Final_Ret': cum_ret[-1],
                        'Spread': spread, 'Mean': mean, 'Std': std,
                        'Cum_Ret_Series': cum_ret, 
                        'Daily_Ret_Series': spr_ret,
                        'Analysis_Dates': z_target.index,
                        'Price A': df_prices[sa].iloc[-1], 'Price B': df_prices[sb].iloc[-1]
                    })
            except: pass
            
    return pd.DataFrame(pairs)

# ---------------------------------------------------------
# 5. Charting
# ---------------------------------------------------------
def plot_chart(row, df_prices, threshold, mode):
    sa, sb = row['Stock A'], row['Stock B']
    dates = row['Analysis_Dates']
    rows = 3 if mode == "Backtest" else 2
    
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=("Price Action", "Strategy Return", "Spread Z-Score") if rows==3 else ("Price Action", "Spread Z-Score"))
    
    pa, pb = df_prices[sa].loc[dates], df_prices[sb].loc[dates]
    fig.add_trace(go.Scatter(x=dates, y=(pa/pa.iloc[0])*100, name=sa, line=dict(color='#2962FF', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=(pb/pb.iloc[0])*100, name=sb, line=dict(color='#FF6D00', width=1.5)), row=1, col=1)
    
    if rows == 3:
        fig.add_trace(go.Scatter(x=dates, y=row['Cum_Ret_Series']*100, name='Return %', line=dict(color='#00C853', width=1.5), fill='tozeroy'), row=2, col=1)
    
    z_row = rows
    z_vals = ((row['Spread'] - row['Mean']) / row['Std']).loc[dates]
    fig.add_trace(go.Scatter(x=dates, y=z_vals, name='Z-Score', line=dict(color='#6200EA', width=1)), row=z_row, col=1)
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", row=z_row, col=1)
    fig.add_hline(y=-threshold, line_dash="dash", line_color="blue", row=z_row, col=1)
    
    fig.update_layout(height=600 if rows==3 else 450, hovermode="x unified", margin=dict(t=30, b=20, l=20, r=20), template="plotly_white")
    return fig

# ---------------------------------------------------------
# 6. Main Execution
# ---------------------------------------------------------
if run_btn:
    with st.spinner("Analyzing market data..."):
        # [NEW] Unpack Ticker Map
        df_prices, ticker_map = load_stock_data(universe_mode, start_input, end_input)
        
        # Reverse Map for Code Lookup: Name -> Code
        name_to_code = {v: k for k, v in ticker_map.items()}

    if df_prices.empty or len(df_prices.columns) < 2:
        st.error("Data Load Error. Please try again.")
    else:
        results = run_analysis(df_prices, window_size, z_threshold, p_cutoff, app_mode, start_input, end_input)
        
        # [Helper] Name Formatting with Code
        def fmt_name(name):
            code = name_to_code.get(name, "Unknown")
            return f"{name} ({code})"

        if results.empty:
            st.warning("No cointegrated pairs found matching your criteria.")
        else:
            if app_mode == "Backtest":
                # ==========================
                # Portfolio Backtest
                # ==========================
                st.subheader("📊 Portfolio Performance Report")
                
                all_returns_df = pd.DataFrame(index=pd.date_range(start=start_input, end=end_input))
                for idx, row in results.iterrows():
                    series = row['Daily_Ret_Series']
                    series.index = pd.to_datetime(series.index)
                    series = series[~series.index.duplicated(keep='first')]
                    series = series.reindex(all_returns_df.index).fillna(0)
                    all_returns_df[f"{row['Stock A']}-{row['Stock B']}"] = series

                portfolio_daily_ret = all_returns_df.mean(axis=1).fillna(0)
                portfolio_cum_ret = (1 + portfolio_daily_ret).cumprod() - 1
                
                wealth = (1 + portfolio_daily_ret).cumprod()
                peak = wealth.expanding(min_periods=1).max()
                mdd = ((wealth - peak) / peak).min()

                m1, m2, m3 = st.columns(3)
                m1.metric("Total Return", f"{portfolio_cum_ret.iloc[-1]*100:.2f}%")
                m2.metric("Max Drawdown", f"{mdd*100:.2f}%")
                m3.metric("Pairs Traded", f"{len(results)}")
                
                fig_port = go.Figure()
                fig_port.add_trace(go.Scatter(x=portfolio_cum_ret.index, y=portfolio_cum_ret*100, mode='lines', name='Equity', line=dict(color='#00C853', width=2), fill='tozeroy'))
                fig_port.update_layout(title="Portfolio Equity Curve", yaxis_title="Return (%)", hovermode="x unified", height=400, template="plotly_white")
                st.plotly_chart(fig_port, use_container_width=True)
                
                st.markdown("---")
                st.subheader("🏆 Top Performing Pairs")
                for idx, row in results.sort_values('Final_Ret', ascending=False).head(5).iterrows():
                    # Format Name with Code
                    sa_disp = fmt_name(row['Stock A'])
                    sb_disp = fmt_name(row['Stock B'])
                    
                    with st.expander(f"Return: {row['Final_Ret']*100:.2f}% | {sa_disp} vs {sb_disp}", expanded=False):
                        st.plotly_chart(plot_chart(row, df_prices, z_threshold, app_mode), use_container_width=True)

            else:
                # ==========================
                # Live Analysis (with Watchlist Tab)
                # ==========================
                actives = results[results['Status'] != 'Watch']
                st.subheader("📡 Live Market Analysis")
                
                c1, c2 = st.columns(2)
                c1.metric("Total Pairs", f"{len(results)}")
                c2.metric("Active Signals", f"{len(actives)}", delta="Action Required")
                
                st.divider()
                
                # [NEW] Tabs for Signals and Watchlist
                tab1, tab2 = st.tabs(["🔥 Active Signals", "👀 Watchlist"])
                
                with tab1:
                    if not actives.empty:
                        for idx, row in actives.sort_values(by='Z-Score', key=abs, ascending=False).iterrows():
                            # Sizing
                            alloc = total_capital / 2
                            qa = int(alloc / row['Price A'])
                            qb = int(alloc / row['Price B'])
                            
                            # Names with codes
                            sa_disp = fmt_name(row['Stock A'])
                            sb_disp = fmt_name(row['Stock B'])
                            
                            is_hedge = "Futures" in universe_mode
                            
                            if row['Status'] == "Buy A":
                                side_a, side_b = ("Long", "Short") if is_hedge else ("Buy", "Avoid")
                                clr = "green"
                            else:
                                side_a, side_b = ("Short", "Long") if is_hedge else ("Avoid", "Buy")
                                clr = "green"
                            
                            with st.expander(f"Signal: {sa_disp} vs {sb_disp} (Z: {row['Z-Score']:.2f})", expanded=True):
                                c_left, c_right = st.columns(2)
                                c_left.markdown(f"**{sa_disp}**")
                                c_left.caption(f"{side_a} {qa:,} shares")
                                
                                c_right.markdown(f"**{sb_disp}**")
                                c_right.caption(f"{side_b} {qb:,} shares")
                                
                                st.plotly_chart(plot_chart(row, df_prices, z_threshold, app_mode), use_container_width=True)
                    else:
                        st.info("No active signals at the moment.")
                        st.caption("All tracked pairs are currently within their normal statistical range.")

                with tab2:
                    st.markdown("### 📋 Full Watchlist (All Cointegrated Pairs)")
                    
                    # Watchlist Dataframe Preparation
                    watch_df = results.copy()
                    
                    # Apply Name Formatting
                    watch_df['Stock A'] = watch_df['Stock A'].apply(fmt_name)
                    watch_df['Stock B'] = watch_df['Stock B'].apply(fmt_name)
                    
                    # Columns to display
                    display_cols = ['Stock A', 'Stock B', 'Z-Score', 'P-value', 'Corr', 'Status']
                    
                    # Display with style
                    st.dataframe(
                        watch_df[display_cols].sort_values(by='Z-Score', key=abs, ascending=False),
                        use_container_width=True,
                        height=600
                    )
else:
    st.info("Please select settings and click Run.")
