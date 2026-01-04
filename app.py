import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import coint
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import FinanceDataReader as fdr
import warnings
import re

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. UI Settings
# ---------------------------------------------------------
st.set_page_config(page_title="Pair Trading Scanner", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #1A1C24; color: #E0E0E0; font-family: 'Pretendard', sans-serif; }
    section[data-testid="stSidebar"] { background-color: #111317; border-right: 1px solid #2B2D35; }
    div[data-testid="metric-container"] { background-color: #252830; border: 1px solid #363945; border-radius: 4px; padding: 15px; }
    div.stButton > button { background-color: #374151; color: white; border: 1px solid #4B5563; border-radius: 4px; font-size: 0.8rem; }
    div.stButton > button:hover { background-color: #4B5563; }
    h1, h2, h3 { color: #F3F4F6 !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

DEFAULTS = { "window_size": 60, "z_threshold": 2.0, "p_cutoff": 0.05 }

# ---------------------------------------------------------
# 2. Logic Engine (Tagging & Grouping)
# ---------------------------------------------------------
# 수동 정의된 강력한 페어들 (자동 스캔 시에도 우선 적용)
MANUAL_MAP = [
    ({'SK', 'SK하이닉스'}, '👨‍👦 SK Group (Parent)'), ({'LG', 'LG전자'}, '👨‍👦 LG Group (Parent)'),
    ({'CJ', 'CJ제일제당'}, '👨‍👦 CJ Group (Parent)'), ({'LS', 'LS ELECTRIC'}, '👨‍👦 LS Group (Parent)'),
    ({'삼성물산', '삼성전자'}, '👨‍👦 Samsung (Gov)'), ({'한화', '한화에어로스페이스'}, '👨‍👦 Hanwha (Parent)'),
    ({'HD현대', 'HD한국조선해양'}, '👨‍👦 HD Hyundai (Parent)'), ({'삼성전자', '삼성전자우'}, '⚡ Common-Pref'),
    ({'현대차', '현대차2우B'}, '⚡ Common-Pref'), ({'LG화학', 'LG화학우'}, '⚡ Common-Pref'),
    ({'SK하이닉스', '한미반도체'}, '🔗 HBM Value Chain'), ({'삼성전자', '삼성전기'}, '🔗 IT Value Chain'),
    ({'현대차', '현대모비스'}, '🔗 Auto Value Chain'), ({'한화에어로스페이스', 'LIG넥스원'}, '⚔️ Defense Rivals'),
    ({'NAVER', '카카오'}, '⚔️ Tech Rivals'), ({'현대차', '기아'}, '⚔️ Auto Rivals')
]

def get_pair_tag(stock_a, stock_b, sector_name=None):
    current_set = {stock_a, stock_b}
    # 1. 수동 맵핑 확인
    for pair_set, tag_name in MANUAL_MAP:
        if current_set == pair_set:
            return tag_name
    
    # 2. 우선주 로직
    if stock_a.replace('우', '').replace('B', '') == stock_b or stock_b.replace('우', '').replace('B', '') == stock_a:
        return "⚡ Common-Pref"
    
    # 3. 섹터 정보 활용
    if sector_name:
        return f"⚔️ Rival ({sector_name})"
    
    return "Random"

# ---------------------------------------------------------
# 3. Data Logic (Top 500 Sector Split)
# ---------------------------------------------------------
@st.cache_data(ttl=86400)
def get_market_data_info(mode):
    """
    모드에 따라 분석 대상 종목 리스트와 섹터 정보를 반환합니다.
    """
    ticker_map = {} # {code: name}
    sector_map = {} # {name: sector}
    
    try:
        if "Top 500" in mode:
            # KRX 전체 로딩 및 필터링
            df_krx = fdr.StockListing('KRX')
            df_krx = df_krx[~df_krx['Name'].str.contains('스팩|제[0-9]+호|리츠|TIGER|KODEX|ETN')]
            df_krx = df_krx.dropna(subset=['Sector', 'Marcap'])
            
            # Top 500 추출
            top500 = df_krx.sort_values('Marcap', ascending=False).head(500)
            
            for _, row in top500.iterrows():
                suffix = ".KS" if row['Market'] == 'KOSPI' else ".KQ"
                full_code = row['Code'] + suffix
                ticker_map[full_code] = row['Name']
                sector_map[row['Name']] = row['Sector']
                
        else: # Manual Core List
            manual_dict = {
                '005930.KS': '삼성전자', '000660.KS': 'SK하이닉스', '005380.KS': '현대차', '000270.KS': '기아',
                '005490.KS': 'POSCO홀딩스', '006400.KS': '삼성SDI', '051910.KS': 'LG화학', '035420.KS': 'NAVER',
                '035720.KS': '카카오', '105560.KS': 'KB금융', '055550.KS': '신한지주', '034020.KS': 'SK',
                '003550.KS': 'LG', '066570.KS': 'LG전자', '000810.KS': '삼성화재', '032830.KS': '삼성생명',
                '028260.KS': '삼성물산', '000880.KS': '한화', '267260.KS': 'HD현대', '001040.KS': 'CJ',
                '042700.KS': '한미반도체', '009150.KS': '삼성전기', '011070.KS': 'LG이노텍', '012330.KS': '현대모비스',
                '012450.KS': '한화에어로스페이스', '079550.KS': 'LIG넥스원', '003670.KS': 'POSCO퓨처엠',
                '005935.KS': '삼성전자우', '005387.KS': '현대차2우B', '051915.KS': 'LG화학우'
            }
            ticker_map = manual_dict
            # Manual 모드는 섹터 정보가 없으므로 'Core'라는 가상의 단일 섹터로 취급하거나
            # 굳이 섹터 스플릿을 안하고 전체 스캔을 돌림. 여기서는 편의상 None 처리.
            sector_map = None 

        return ticker_map, sector_map

    except Exception as e:
        st.error(f"Market Data Error: {e}")
        return {}, {}

@st.cache_data(ttl=3600)
def fetch_price_data(ticker_map, start_date, end_date):
    tickers = list(ticker_map.keys())
    if '^KS11' not in tickers: tickers.append('^KS11')
    
    # yfinance 다운로드
    try:
        # 데이터가 너무 많으면 안내 메시지
        if len(tickers) > 100:
            st.toast(f"Downloading {len(tickers)} stocks...", icon="⏳")
            
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
        data = data.dropna(axis=1, how='all') # 데이터 없는 컬럼 삭제
        
        # 지수 분리
        if '^KS11' in data.columns:
            kospi = data['^KS11'].rename('KOSPI')
            stocks = data.drop(columns=['^KS11'])
        else:
            kospi = pd.Series()
            stocks = data
            
        # 한글명 변환
        stocks = stocks.rename(columns=ticker_map)
        stocks = stocks.ffill().bfill()
        
        return stocks, kospi
    except Exception as e:
        st.error(f"Price Download Error: {e}")
        return pd.DataFrame(), pd.Series()

# ---------------------------------------------------------
# 4. Analysis Engine
# ---------------------------------------------------------
def run_analysis(df_prices, window, threshold, p_val, start, end, sector_map):
    pairs = []
    target_mask = (df_prices.index >= pd.to_datetime(start)) & (df_prices.index <= pd.to_datetime(end))
    cols = df_prices.columns
    
    prog_bar = st.progress(0, text="Initializing Analysis...")
    
    # Grouping Strategy
    groups = {}
    if sector_map:
        # Sector Split Mode: 섹터별로 묶음
        for stock in cols:
            sec = sector_map.get(stock, 'Unknown')
            if sec not in groups: groups[sec] = []
            groups[sec].append(stock)
    else:
        # Manual Mode: 전체를 하나의 그룹으로 (All-to-All)
        groups['Core'] = list(cols)
    
    total_groups = len(groups)
    processed = 0
    
    for group_name, stocks in groups.items():
        processed += 1
        n = len(stocks)
        if n < 2: continue
        
        prog_bar.progress(processed / total_groups, text=f"Scanning [{group_name}] ({n} stocks)")
        
        for i in range(n):
            for j in range(i + 1, n):
                sa, sb = stocks[i], stocks[j]
                
                # Correlation Filter (Fast)
                if df_prices[sa].corr(df_prices[sb]) < 0.6: continue
                
                try:
                    # Cointegration Test (Slow)
                    score, pval, _ = coint(df_prices[sa], df_prices[sb])
                    if pval < p_val:
                        # Signal Generation
                        log_a, log_b = np.log(df_prices[sa]), np.log(df_prices[sb])
                        spread = log_a - log_b
                        mean = spread.rolling(window).mean()
                        std = spread.rolling(window).std()
                        z_all = (spread - mean) / std
                        z_target = z_all.loc[target_mask]
                        
                        if z_target.empty: continue
                        
                        positions = np.zeros(len(z_target)); curr_pos = 0
                        for k in range(len(z_target)):
                            z = z_target.iloc[k]
                            if curr_pos == 0:
                                if z < -threshold: curr_pos = 1 
                                elif z > threshold: curr_pos = -1
                            elif curr_pos == 1:
                                if z >= 0 or z < -4.0: curr_pos = 0
                            elif curr_pos == -1:
                                if z <= 0 or z > 4.0: curr_pos = 0
                            positions[k] = curr_pos
                        
                        ret_a = df_prices[sa].loc[target_mask].pct_change().fillna(0)
                        ret_b = df_prices[sb].loc[target_mask].pct_change().fillna(0)
                        spr_ret = (ret_a - ret_b) * pd.Series(positions, index=z_target.index).shift(1).fillna(0).values
                        
                        # Tag Generation
                        tag = get_pair_tag(sa, sb, group_name if sector_map else None)
                        
                        pairs.append({
                            'Stock A': sa, 'Stock B': sb, 'Tag': tag,
                            'Z-Score': z_all.iloc[-1], 'Corr': df_prices[sa].corr(df_prices[sb]), 'P-value': pval,
                            'Final_Ret': (1 + spr_ret).prod() - 1,
                            'Daily_Ret_Series': pd.Series(spr_ret, index=z_target.index),
                            'Spread': spread, 'Mean': mean, 'Std': std, 'Analysis_Dates': z_target.index,
                            'Price A': df_prices[sa].iloc[-1], 'Price B': df_prices[sb].iloc[-1]
                        })
                except: pass
                
    prog_bar.empty()
    return pd.DataFrame(pairs)

# ---------------------------------------------------------
# 5. Visualization Functions
# ---------------------------------------------------------
def plot_pair_analysis(row, df_prices, threshold):
    sa, sb = row['Stock A'], row['Stock B']
    dates = row['Analysis_Dates']
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25])
    
    pa, pb = df_prices[sa].loc[dates], df_prices[sb].loc[dates]
    # Rebase to 100
    fig.add_trace(go.Scatter(x=dates, y=(pa/pa.iloc[0])*100, name=sa, line=dict(color='#3B82F6', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=(pb/pb.iloc[0])*100, name=sb, line=dict(color='#F59E0B', width=1.5)), row=1, col=1)
    
    z_vals = ((row['Spread'] - row['Mean']) / row['Std']).loc[dates]
    fig.add_trace(go.Scatter(x=dates, y=z_vals, name='Z-Score', line=dict(color='#9CA3AF', width=1)), row=2, col=1)
    
    # Signal Markers
    sell_sig = z_vals[z_vals > threshold]; buy_sig = z_vals[z_vals < -threshold]
    fig.add_trace(go.Scatter(x=sell_sig.index, y=sell_sig, mode='markers', marker=dict(color='#EF4444', size=5), name='Sell', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=buy_sig.index, y=buy_sig, mode='markers', marker=dict(color='#3B82F6', size=5), name='Buy', showlegend=False), row=2, col=1)
    
    fig.add_hline(y=threshold, line_dash="dash", line_color="#EF4444", row=2, col=1)
    fig.add_hline(y=-threshold, line_dash="dash", line_color="#3B82F6", row=2, col=1)
    fig.add_hrect(y0=-threshold, y1=threshold, fillcolor="gray", opacity=0.1, line_width=0, row=2, col=1)
    
    cum = (1 + row['Daily_Ret_Series']).cumprod() * 100 - 100
    fig.add_trace(go.Scatter(x=dates, y=cum, name='Return %', line=dict(color='#10B981', width=1.5), fill='tozeroy'), row=3, col=1)
    
    fig.update_layout(title=f"<b>[{row['Tag']}] {sa} vs {sb}</b>", height=600, template="plotly_dark", plot_bgcolor='#1A1C24', paper_bgcolor='#1A1C24', margin=dict(t=50, b=10))
    return fig

def plot_scatter(results):
    if results.empty: return None
    fig = px.scatter(
        results, x='Corr', y=results['Z-Score'].abs(), color='Tag',
        hover_data=['Stock A', 'Stock B'],
        title='Opportunity Map', labels={'Corr': 'Correlation', 'y': 'Abs Z-Score'},
        template='plotly_dark'
    )
    fig.add_shape(type="rect", x0=0.8, y0=2.0, x1=1.0, y1=results['Z-Score'].abs().max() + 0.5,
        line=dict(color="#10B981", width=1, dash="dot"), fillcolor="#10B981", opacity=0.1)
    fig.update_layout(height=400, plot_bgcolor='#1A1C24', paper_bgcolor='#1A1C24')
    return fig

# ---------------------------------------------------------
# 6. Sidebar & Main Execution
# ---------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    universe_mode = st.selectbox("Target Universe", ["Top 500 (Sector Split)", "Manual Core List"])
    app_mode = st.radio("Mode", ["Live Analysis", "Backtest"])
    st.divider()
    total_capital = st.number_input("Capital (KRW)", value=10000000, step=1000000, format="%d")
    
    with st.expander("Parameters", expanded=True):
        for key in DEFAULTS:
            if key not in st.session_state: st.session_state[key] = DEFAULTS[key]
        window_size = st.slider("Window Size", 20, 120, key="window_size")
        z_threshold = st.slider("Z-Score Threshold", 1.0, 4.0, key="z_threshold")
        p_cutoff = st.slider("Max P-value", 0.01, 0.20, key="p_cutoff")
        
        st.write("") 
        if st.button("Reset Parameters"):
            for key, value in DEFAULTS.items(): st.session_state[key] = value
            st.rerun()

    st.divider()
    if app_mode == "Backtest":
        c1, c2 = st.columns(2)
        start_input = c1.date_input("Start", datetime(2025, 1, 1))
        end_input = c2.date_input("End", datetime(2025, 12, 31))
        run_label = "Run Backtest"
    else:
        end_input = datetime.now(); start_input = end_input - timedelta(days=365)
        run_label = "Run Analysis"

    run_btn = st.button(run_label, type="primary", use_container_width=True)

if run_btn:
    # 1. 정보 획득 (티커, 섹터 맵)
    with st.spinner("Initializing Market Universe..."):
        ticker_map, sector_map = get_market_data_info(universe_mode)
    
    if not ticker_map:
        st.error("Failed to load ticker info.")
    else:
        # 2. 가격 데이터 다운로드
        fetch_start_str = start_input.strftime('%Y-%m-%d')
        fetch_end_str = end_input.strftime('%Y-%m-%d')
        
        stocks, kospi = fetch_price_data(ticker_map, fetch_start_str, fetch_end_str)
        
        if stocks.empty:
            st.error("Failed to download price data.")
        else:
            # 3. 분석 실행
            results = run_analysis(stocks, window_size, z_threshold, p_cutoff, start_input, end_input, sector_map)
            
            def fmt(name):
                code_list = [k for k, v in ticker_map.items() if v == name]
                code = code_list[0].split('.')[0] if code_list else "Unknown"
                return f"{name} ({code})"
            
            if results.empty:
                st.warning("No pairs found. Try relaxing P-value or Z-Score.")
            
            elif app_mode == "Backtest":
                # Benchmark (KOSPI)
                if not kospi.empty:
                    k_period = kospi.loc[start_input:end_input]; k_ret = (k_period / k_period.iloc[0]) - 1
                else: k_ret = pd.Series(0, index=pd.date_range(start_input, end_input))

                # Strategy Portfolio
                all_ret = pd.DataFrame(index=k_ret.index)
                for _, row in results.iterrows(): 
                    s = row['Daily_Ret_Series']
                    s.index = pd.to_datetime(s.index)
                    all_ret[f"{row['Stock A']}-{row['Stock B']}"] = s.reindex(all_ret.index).fillna(0)
                
                p_daily = all_ret.mean(axis=1); p_cum = (1 + p_daily).cumprod() - 1
                
                # Metrics
                st.subheader("Performance Report (vs KOSPI)")
                c1, c2, c3 = st.columns(3)
                s_final = p_cum.iloc[-1]*100 if not p_cum.empty else 0
                k_final = k_ret.iloc[-1]*100 if not k_ret.empty else 0
                c1.metric("Strategy Return", f"{s_final:.2f}%", f"{s_final-k_final:.2f}% vs Market")
                c2.metric("Benchmark Return", f"{k_final:.2f}%"); c3.metric("Alpha", f"{s_final-k_final:.2f}%p")
                
                # Comparison Chart
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(x=p_cum.index, y=p_cum*100, name='Strategy', line=dict(color='#10B981', width=3)))
                fig_comp.add_trace(go.Scatter(x=k_ret.index, y=k_ret*100, name='Benchmark', line=dict(color='#9CA3AF', width=2, dash='dot')))
                fig_comp.update_layout(title="Cumulative Return Comparison", template="plotly_dark", height=400, plot_bgcolor='#1A1C24', paper_bgcolor='#1A1C24')
                st.plotly_chart(fig_comp, use_container_width=True)
                
                st.plotly_chart(plot_scatter(results), use_container_width=True)

                st.divider()
                col_t, col_w = st.columns(2)
                with col_t:
                    st.subheader("Top Performers")
                    for _, row in results.sort_values('Final_Ret', ascending=False).head(5).iterrows():
                        with st.expander(f"{row['Tag']} | {fmt(row['Stock A'])} / {fmt(row['Stock B'])} ({row['Final_Ret']*100:.1f}%)"):
                            st.plotly_chart(plot_pair_analysis(row, stocks, z_threshold), use_container_width=True)
                with col_w:
                    st.subheader("Worst Performers")
                    for _, row in results.sort_values('Final_Ret', ascending=True).head(5).iterrows():
                        with st.expander(f"{row['Tag']} | {fmt(row['Stock A'])} / {fmt(row['Stock B'])} ({row['Final_Ret']*100:.1f}%)"):
                            st.plotly_chart(plot_pair_analysis(row, stocks, z_threshold), use_container_width=True)
            else:
                # Live Mode
                st.subheader("Live Trading Signals")
                actives = results[results['Z-Score'].abs() >= z_threshold]
                col1, col2 = st.columns([3, 1]); col1.markdown(f"**{len(results)}** pairs monitored."); col2.metric("Active Signals", f"{len(actives)}")
                
                tab1, tab2 = st.tabs(["Action Required", "Watchlist"])
                with tab1:
                    if not actives.empty:
                        for _, row in actives.sort_values(by='Z-Score', key=abs, ascending=False).iterrows():
                            with st.expander(f"🎯 [{row['Tag']}] {fmt(row['Stock A'])} / {fmt(row['Stock B'])} (Z: {row['Z-Score']:.2f})", expanded=True):
                                st.plotly_chart(plot_pair_analysis(row, stocks, z_threshold), use_container_width=True)
                    else: st.info("No signals matching current threshold.")
                with tab2:
                    st.plotly_chart(plot_scatter(results), use_container_width=True)
                    df_v = results[['Tag', 'Stock A', 'Stock B', 'Z-Score', 'Corr', 'Price A', 'Price B']].copy()
                    df_v['Stock A'] = df_v['Stock A'].apply(fmt); df_v['Stock B'] = df_v['Stock B'].apply(fmt)
                    st.dataframe(df_v.sort_values('Z-Score', key=abs, ascending=False), use_container_width=True)
else: st.info("Configure settings and click Run.")
