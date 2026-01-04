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
# 2. Logic Engine (Top 500 Sector Split)
# ---------------------------------------------------------
@st.cache_data(ttl=86400)
def get_krx_top500_sectors():
    """
    KRX 시가총액 상위 500개 종목을 추출한 뒤,
    섹터별로 그룹핑하여 리턴합니다.
    """
    try:
        # 1. 전체 리스트 가져오기
        df_krx = fdr.StockListing('KRX')
        
        # 2. 데이터 클리닝 (스팩, 우선주, 리츠 등 제외)
        # 우선주는 별도 로직이 없다면 제외하는 것이 일반적이나, 시총 상위 포함을 원하면 유지 가능.
        # 여기서는 노이즈 제거를 위해 일반적인 우선주/스팩 제거
        df_krx = df_krx[~df_krx['Name'].str.contains('스팩|제[0-9]+호|우B|우$|리츠|TIGER|KODEX|ETN')]
        df_krx = df_krx.dropna(subset=['Sector', 'Marcap'])
        
        # 3. 시가총액 상위 500개 추출 (The Global Top 500)
        top500 = df_krx.sort_values('Marcap', ascending=False).head(500)
        
        sector_dict = {}     # { '반도체': ['005930.KS', ...], ... }
        ticker_name_map = {} # { '005930.KS': '삼성전자', ... }
        
        # 4. 섹터별 그룹핑
        # Top 500 안에 포함된 종목들만 가지고 섹터를 나눕니다.
        for sector, group in top500.groupby('Sector'):
            # 해당 섹터에 종목이 2개 이상이어야 페어링 가능
            if len(group) < 2: continue
            
            codes = []
            for _, row in group.iterrows():
                suffix = ".KS" if row['Market'] == 'KOSPI' else ".KQ"
                full_code = row['Code'] + suffix
                codes.append(full_code)
                ticker_name_map[full_code] = row['Name']
            
            sector_dict[sector] = codes
            
        return sector_dict, ticker_name_map
        
    except Exception as e:
        st.error(f"KRX Data Error: {e}")
        return {}, {}

# ---------------------------------------------------------
# 3. Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    
    # [NEW] TOP 500 모드 추가
    universe_mode = st.selectbox(
        "Target Universe", 
        ["Top 500 (Sector Split)", "Manual Core List"]
    )
    
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

# ---------------------------------------------------------
# 4. Data Loading (Batch Optimized)
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
    
    # 1. TOP 500 Sector Split 모드
    if "Top 500" in universe_type:
        sector_map, ticker_map = get_krx_top500_sectors()
        
        all_codes = [code for codes in sector_map.values() for code in codes]
        # 지수 추가
        if '^KS11' not in all_codes: all_codes.append('^KS11')
        
        # 500개 종목 다운로드는 시간이 걸리므로 진행상황 표시
        st.toast(f"Downloading Top {len(all_codes)} stocks... Please wait.", icon="⏳")
        
        try:
            # yfinance 대량 다운로드
            df_all = yf.download(all_codes, start=fetch_start, end=fetch_end, progress=False)['Close']
            df_all = df_all.dropna(axis=1, how='all') # 데이터 없는 것 제거
            
            if '^KS11' in df_all.columns:
                kospi = df_all['^KS11'].rename('KOSPI')
                stocks = df_all.drop(columns=['^KS11'])
            else:
                kospi = pd.Series()
                stocks = df_all
                
            stocks = stocks.rename(columns=ticker_map)
            stocks = stocks.ffill().bfill()
            
            # 역매핑 (종목명 -> 섹터)
            reverse_sector_map = {}
            for sec, codes in sector_map.items():
                for c in codes:
                    if c in ticker_map:
                        name = ticker_map[c]
                        reverse_sector_map[name] = sec
            
            return stocks, kospi, ticker_map, reverse_sector_map
            
        except Exception as e:
            st.error(f"Download Error: {e}")
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
# 5. Analysis Engine (Sector Loop)
# ---------------------------------------------------------
def run_analysis(df_prices, window, threshold, p_val, start, end, sector_info):
    pairs = []
    target_mask = (df_prices.index >= pd.to_datetime(start)) & (df_prices.index <= pd.to_datetime(end))
    
    prog_bar = st.progress(0, text="Sector Analysis Initializing...")
    
    if sector_info:
        # 섹터별 종목 분류
        sectors = {}
        for name, sec in sector_info.items():
            if name in df_prices.columns:
                if sec not in sectors: sectors[sec] = []
                sectors[sec].append(name)
        
        total_sectors = len(sectors)
        processed_sec = 0
        
        # 섹터 단위 루프 (효율성 극대화)
        for sec_name, stock_list in sectors.items():
            processed_sec += 1
            n = len(stock_list)
            if n < 2: continue
            
            # Progress Update
            prog_bar.progress(processed_sec / total_sectors, text=f"Analyzing [{sec_name}]: {n} stocks")
            
            for i in range(n):
                for j in range(i + 1, n):
                    sa, sb = stock_list[i], stock_list[j]
                    
                    # Correlation Filter
                    if df_prices[sa].corr(df_prices[sb]) < 0.6: continue
                    
                    try:
                        score, pval, _ = coint(df_prices[sa], df_prices[sb])
                        if pval < p_val:
                            log_a, log_b = np.log(df_prices[sa]), np.log(df_prices[sb])
                            spread = log_a - log_b
                            mean, std = spread.rolling(window).mean(), spread.rolling(window).std()
                            z_all = (spread - mean) / std; z_target = z_all.loc[target_mask]
                            if z_target.empty: continue
                            
                            # Backtest logic
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
                            
                            tag = f"🛡️ {sec_name}"
                            
                            pairs.append({
                                'Stock A': sa, 'Stock B': sb, 'Tag': tag,
                                'Z-Score': z_all.iloc[-1], 'Corr': df_prices[sa].corr(df_prices[sb]), 'P-value': pval,
                                'Final_Ret': (1 + spr_ret).prod() - 1, 'Daily_Ret_Series': pd.Series(spr_ret, index=z_target.index),
                                'Spread': spread, 'Mean': mean, 'Std': std, 'Analysis_Dates': z_target.index,
                                'Price A': df_prices[sa].iloc[-1], 'Price B': df_prices[sb].iloc[-1]
                            })
                    except: pass
    else:
        # Fallback for Manual Mode
        cols = df_prices.columns
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                # ... (Simple Loop Logic) ...
