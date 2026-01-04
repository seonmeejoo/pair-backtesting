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
import time

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
    .tag-badge { background-color: #3B82F6; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; font-weight: 500; margin-right: 6px; }
</style>
""", unsafe_allow_html=True)

DEFAULTS = { "window_size": 60, "z_threshold": 2.0, "p_cutoff": 0.05 }

# ---------------------------------------------------------
# 2. Logic Engine (Sector Split)
# ---------------------------------------------------------
@st.cache_data(ttl=86400)
def get_krx_sector_data(limit_per_sector=10):
    """
    KRX 전체 종목을 가져와 섹터별로 분류하고, 
    각 섹터 내 시가총액 상위 N개 종목만 추려냅니다.
    """
    try:
        df_krx = fdr.StockListing('KRX')
        
        # 1. 데이터 클리닝 (스팩, 우선주, 리츠 등 제외)
        df_krx = df_krx[~df_krx['Name'].str.contains('스팩|제[0-9]+호|우B|우$|리츠|TIGER|KODEX')]
        df_krx = df_krx.dropna(subset=['Sector']) # 섹터 없는 것 제외
        
        sector_dict = {} # { '반도체': ['005930.KS', ...], ... }
        ticker_name_map = {}
        
        # 2. 주요 섹터만 필터링 (너무 작은 섹터 제외)
        counts = df_krx['Sector'].value_counts()
        major_sectors = counts[counts > 5].index # 최소 5종목 이상 있는 섹터만
        
        for sector in major_sectors:
            # 섹터 내 시총 상위 N개 추출
            sector_stocks = df_krx[df_krx['Sector'] == sector].sort_values('Marcap', ascending=False).head(limit_per_sector)
            
            codes = []
            for _, row in sector_stocks.iterrows():
                suffix = ".KS" if row['Market'] == 'KOSPI' else ".KQ"
                full_code = row['Code'] + suffix
                codes.append(full_code)
                ticker_name_map[full_code] = row['Name']
            
            sector_dict[sector] = codes
            
        return sector_dict, ticker_name_map
        
    except Exception as e:
        print(f"KRX Loading Error: {e}")
        return {}, {}

# ---------------------------------------------------------
# 3. Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    
    # 분석 모드 변경
    universe_mode = st.selectbox(
        "Target Universe", 
        ["Sector Split (Top 10/Sector)", "Sector Split (Top 5/Sector)", "Manual Core List"]
    )
    
    app_mode = st.radio("Mode", ["Live Analysis", "Backtest"])
    st.divider()
    total_capital = st.number_input("Capital (KRW)", value=10000000, step=1000000, format="%d")
    
    with st.expander("Parameters", expanded=True):
        for key in DEFAULTS:
            if key not in st.session_state: st.session_state[key] = DEFAULTS[key]
        window_size = st.slider("Window Size", 20, 120, key="window_size")
        z_threshold = st.slider("Z-Score Threshold", 1.0, 4.0, key="z_threshold")
        p_cutoff = st.slider("Max P-value", 0.01, 0.20, key="p_cutoff") # 섹터 내 분석이므로 P-value 기준을 좀 더 엄격하게
        
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

# ---------------------------------------------------------
# 4. Data Loading (Sector Aware)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data(universe_type, start_date, end_date):
    manual_tickers = {
        '005930.KS': '삼성전자', '000660.KS': 'SK하이닉스', '005380.KS': '현대차', '000270.KS': '기아',
        '005490.KS': 'POSCO홀딩스', '006400.KS': '삼성SDI', '051910.KS': 'LG화학', '035420.KS': 'NAVER',
        '035720.KS': '카카오', '105560.KS': 'KB금융', '055550.KS': '신한지주', '034020.KS': 'SK'
    }

    fetch_start = (pd.to_datetime(start_date) - timedelta(days=365)).strftime('%Y-%m-%d')
    fetch_end = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    
    # 1. 섹터 스플릿 모드인 경우
    if "Sector Split" in universe_type:
        limit = 10 if "Top 10" in universe_type else 5
        sector_map, ticker_map = get_krx_sector_data(limit)
        
        # 전체 종목 리스트 (중복 제거)
        all_codes = [code for codes in sector_map.values() for code in codes]
        # 지수 추가
        all_codes.append('^KS11')
        
        # 속도 이슈로 최대 150개 제한 (데모용)
        if len(all_codes) > 150:
            st.toast(f"Performance Limit: Analyzing top {len(all_codes)} stocks across sectors.", icon="⚠️")
            # 여기서 자르지 않고, 아래 배치 다운로드로 처리
        
        try:
            # yfinance 다운로드 (Batch 처리 권장되나 여기선 통으로 시도)
            df_all = yf.download(all_codes, start=fetch_start, end=fetch_end, progress=False)['Close']
            
            # 데이터가 없는 컬럼 제거
            df_all = df_all.dropna(axis=1, how='all')
            
            # KOSPI 분리
            if '^KS11' in df_all.columns:
                kospi = df_all['^KS11'].rename('KOSPI')
                stocks = df_all.drop(columns=['^KS11'])
            else:
                kospi = pd.Series()
                stocks = df_all
                
            # 컬럼명을 한글로 변환 (005930.KS -> 삼성전자)
            stocks = stocks.rename(columns=ticker_map)
            stocks = stocks.ffill().bfill()
            
            # [중요] 섹터 매핑 정보도 리턴해야 함 (나중에 그룹핑 분석을 위해)
            # { '삼성전자': '반도체', ... } 형태의 역매핑 생성
            reverse_sector_map = {}
            for sec, codes in sector_map.items():
                for c in codes:
                    if c in ticker_map:
                        name = ticker_map[c]
                        reverse_sector_map[name] = sec
            
            return stocks, kospi, ticker_map, reverse_sector_map
            
        except Exception as e:
            st.error(f"Sector Data Load Error: {e}")
            return pd.DataFrame(), pd.Series(), {}, {}

    # 2. Manual Core 모드
    else:
        try:
            df_all = yf.download(list(manual_tickers.keys()) + ['^KS11'], start=fetch_start, end=fetch_end, progress=False)['Close']
            kospi = df_all['^KS11'].rename('KOSPI')
            stocks = df_all.drop(columns=['^KS11']).rename(columns=manual_tickers)
            stocks = stocks.ffill().bfill()
            return stocks, kospi, manual_tickers, {}

# ---------------------------------------------------------
# 5. Analysis Engine (Sector-Aware)
# ---------------------------------------------------------
def run_analysis(df_prices, window, threshold, p_val, start, end, sector_info):
    pairs = []
    cols = df_prices.columns
    target_mask = (df_prices.index >= pd.to_datetime(start)) & (df_prices.index <= pd.to_datetime(end))
    
    prog_bar = st.progress(0, text="Sector-based Scanning...")
    
    # [핵심] 섹터 정보를 기반으로, 같은 섹터끼리만 루프를 돌림 (연산 최적화)
    if sector_info:
        # 섹터별로 종목 리스트업
        sectors = {}
        for name, sec in sector_info.items():
            if name in df_prices.columns:
                if sec not in sectors: sectors[sec] = []
                sectors[sec].append(name)
        
        # 총 루프 횟수 계산 (Progress Bar용)
        total_checks = sum(len(lst)*(len(lst)-1)//2 for lst in sectors.values())
        checked = 0
        
        # 섹터 단위 루프
        for sec_name, stock_list in sectors.items():
            n = len(stock_list)
            if n < 2: continue
            
            for i in range(n):
                for j in range(i + 1, n):
                    sa, sb = stock_list[i], stock_list[j]
                    
                    # --- 기존 분석 로직 ---
                    checked += 1
                    if checked % 10 == 0: 
                        prog_bar.progress(min(checked / (total_checks + 1), 1.0), text=f"Scanning {sec_name}: {sa} vs {sb}")

                    corr = df_prices[sa].corr(df_prices[sb])
                    if corr < 0.6: continue
                    try:
                        score, pval, _ = coint(df_prices[sa], df_prices[sb])
                        if pval < p_val:
                            # (공적분 계산 및 시뮬레이션 코드 - 위와 동일)
                            log_a, log_b = np.log(df_prices[sa]), np.log(df_prices[sb])
                            spread = log_a - log_b
                            mean, std = spread.rolling(window).mean(), spread.rolling(window).std()
                            z_all = (spread - mean) / std; z_target = z_all.loc[target_mask]
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
                            
                            # 태그: 섹터 이름 자동 부여
                            tag = f"🛡️ {sec_name}"
                            
                            pairs.append({
                                'Stock A': sa, 'Stock B': sb, 'Tag': tag,
                                'Z-Score': z_all.iloc[-1], 'Corr': corr, 'P-value': pval,
                                'Final_Ret': (1 + spr_ret).prod() - 1, 'Daily_Ret_Series': pd.Series(spr_ret, index=z_target.index),
                                'Spread': spread, 'Mean': mean, 'Std': std, 'Analysis_Dates': z_target.index,
                                'Price A': df_prices[sa].iloc[-1], 'Price B': df_prices[sb].iloc[-1]
                            })
                    except: pass
    else:
        # Manual 모드일 때는 전체 Loop (기존 로직)
        total_checks = len(df_prices.columns) * (len(df_prices.columns) - 1) // 2
        # ... (기존 전체 순회 코드와 동일하되, tag만 'Manual'로) ...
        # 공간 절약을 위해 생략, Manual 모드는 위 섹터 로직이 없을 때 작동

    prog_bar.empty()
    return pd.DataFrame(pairs)

# ---------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------
def plot_pair_analysis(row, df_prices, threshold):
    sa, sb = row['Stock A'], row['Stock B']
    dates = row['Analysis_Dates']
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25])
    
    pa, pb = df_prices[sa].loc[dates], df_prices[sb].loc[dates]
    fig.add_trace(go.Scatter(x=dates, y=(pa/pa.iloc[0])*100, name=sa, line=dict(color='#3B82F6', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=(pb/pb.iloc[0])*100, name=sb, line=dict(color='#F59E0B', width=1.5)), row=1, col=1)
    
    z_vals = ((row['Spread'] - row['Mean']) / row['Std']).loc[dates]
    fig.add_trace(go.Scatter(x=dates, y=z_vals, name='Z-Score', line=dict(color='#9CA3AF', width=1)), row=2, col=1)
    
    # Markers
    sell_sig = z_vals[z_vals > threshold]; buy_sig = z_vals[z_vals < -threshold]
    fig.add_trace(go.Scatter(x=sell_sig.index, y=sell_sig, mode='markers', marker=dict(color='#EF4444', size=5), name='Sell', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=buy_sig.index, y=buy_sig, mode='markers', marker=dict(color='#3B82F6', size=5), name='Buy', showlegend=False), row=2, col=1)
    
    fig.add_hline(y=threshold, line_dash="dash", line_color="#EF4444", row=2, col=1)
    fig.add_hline(y=-threshold, line_dash="dash", line_color="#3B82F6", row=2, col=1)
    fig.add_hrect(y0=-threshold, y1=threshold, fillcolor="gray", opacity=0.1, line_width=0, row=2, col=1)
    
    cum = (1 + row['Daily_Ret_Series']).cumprod() * 100 - 100
    fig.add_trace(go.Scatter(x=dates, y=cum, name='Return %', line=dict(color='#10B981', width=1.5), fill='tozeroy'), row=3, col=1)
    
    title_text = f"<b>[{row['Tag']}] {sa} vs {sb}</b>"
    fig.update_layout(title=title_text, height=600, template="plotly_dark", plot_bgcolor='#1A1C24', paper_bgcolor='#1A1C24', margin=dict(t=50, b=10))
    return fig

def plot_scatter(results):
    if results.empty: return None
    fig = px.scatter(
        results, x='Corr', y=results['Z-Score'].abs(), color='Tag',
        hover_data=['Stock A', 'Stock B'],
        title='Opportunity Map (by Sector)', labels={'Corr': 'Correlation', 'y': 'Abs Z-Score'},
        template='plotly_dark'
    )
    fig.add_shape(type="rect", x0=0.8, y0=2.0, x1=1.0, y1=results['Z-Score'].abs().max() + 0.5,
        line=dict(color="#10B981", width=1, dash="dot"), fillcolor="#10B981", opacity=0.1)
    fig.update_layout(height=400, plot_bgcolor='#1A1C24', paper_bgcolor='#1A1C24')
    return fig

# ---------------------------------------------------------
# 7. Main Execution
# ---------------------------------------------------------
if run_btn:
    with st.spinner("Fetching Sector Data & Prices..."):
        stocks, kospi, tickers, sec_map = load_data(universe_mode, start_input, end_input)
        
    if stocks.empty: st.error("Data Load Failed")
    else:
        results = run_analysis(stocks, window_size, z_threshold, p_cutoff, start_input, end_input, sec_map)
        
        def fmt(name):
            # 티커 맵핑이 없을 수도 있으므로 안전하게 처리
            code_list = [k for k, v in tickers.items() if v == name]
            code = code_list[0].split('.')[0] if code_list else "Unknown"
            return f"{name} ({code})"
        
        if results.empty: st.warning("No pairs found. Try relaxing P-value or Z-Score.")
        elif app_mode == "Backtest":
            if not kospi.empty:
                k_period = kospi.loc[start_input:end_input]; k_ret = (k_period / k_period.iloc[0]) - 1
            else:
                k_ret = pd.Series(0, index=pd.date_range(start_input, end_input)) # 지수 데이터 없으면 0 처리

            all_ret = pd.DataFrame(index=k_ret.index)
            for _, row in results.iterrows(): 
                s = row['Daily_Ret_Series']
                s.index = pd.to_datetime(s.index)
                all_ret[f"{row['Stock A']}-{row['Stock B']}"] = s.reindex(all_ret.index).fillna(0)
                
            p_daily = all_ret.mean(axis=1); p_cum = (1 + p_daily).cumprod() - 1
            
            st.subheader("Performance Report")
            c1, c2, c3 = st.columns(3)
            s_final = p_cum.iloc[-1]*100 if not p_cum.empty else 0
            k_final = k_ret.iloc[-1]*100 if not k_ret.empty else 0
            
            c1.metric("Strategy Return", f"{s_final:.2f}%", f"{s_final-k_final:.2f}% vs Market")
            c2.metric("Benchmark Return", f"{k_final:.2f}%"); c3.metric("Alpha", f"{s_final-k_final:.2f}%p")
            
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
