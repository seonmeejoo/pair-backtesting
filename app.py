import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from itertools import combinations
from datetime import datetime, timedelta
import os
import time

# ==========================================
# 🎨 0. 스타일 설정 (Dark & Neon)
# ==========================================
def init_settings():
    font_path = "NanumGothic.ttf"
    if not os.path.exists(font_path):
        url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
        response = requests.get(url)
        with open(font_path, "wb") as f:
            f.write(response.content)
    fm.fontManager.addfont(font_path)
    plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False 

    plt.style.use('dark_background')
    plt.rcParams.update({
        'axes.facecolor': '#111111',
        'figure.facecolor': '#111111',
        'grid.color': '#444444',
        'grid.linestyle': '--',
        'grid.alpha': 0.5,
        'axes.edgecolor': '#888888',
        'text.color': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'axes.labelcolor': '#ff9900',
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'lines.linewidth': 1.5
    })

init_settings()

# ==========================================
# 📡 1. 데이터 수집
# ==========================================
@st.cache_data(ttl=3600*12)
def fetch_all_naver_stocks():
    base_url = "https://finance.naver.com/sise/sise_group.naver?type=upjong"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        res = requests.get(base_url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        table = soup.find('table', {'class': 'type_1'})
        rows = table.find_all('tr')
        
        sector_links = []
        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 2: continue
            link_tag = cols[0].find('a')
            if link_tag:
                sector_links.append((link_tag.text.strip(), "https://finance.naver.com" + link_tag['href']))
        
        all_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_sectors = len(sector_links)
        
        for idx, (sec_name, sec_url) in enumerate(sector_links):
            status_text.text(f"📡 System Scanning... [{idx+1}/{total_sectors}] {sec_name}")
            progress_bar.progress((idx + 1) / total_sectors)
            
            res_sec = requests.get(sec_url, headers=headers)
            soup_sec = BeautifulSoup(res_sec.text, 'html.parser')
            sub_table = soup_sec.find('table', {'class': 'type_5'})
            if not sub_table: continue
            
            for s_row in sub_table.find_all('tr'):
                s_cols = s_row.find_all('td')
                if len(s_cols) < 2: continue 
                name_tag = s_cols[0].find('a')
                if name_tag:
                    all_data.append({
                        'Sector': sec_name,
                        'Name': name_tag.text.strip(),
                        'Code': name_tag['href'].split('code=')[-1],
                        'Price': s_cols[1].text.strip()
                    })
            time.sleep(0.02)
            
        progress_bar.empty()
        status_text.empty()
        
        df_naver = pd.DataFrame(all_data).drop_duplicates(subset=['Code'])
        
        status_text.text("💰 Fetching Market Cap Data...")
        df_krx = fdr.StockListing('KRX')[['Code', 'Marcap']]
        df_merged = pd.merge(df_naver, df_krx, on='Code', how='left').fillna({'Marcap': 0})
        
        df_merged = df_merged.sort_values(by=['Sector', 'Marcap'], ascending=[True, False])
        
        def format_marcap(val):
            if val == 0: return "-"
            val = int(val)
            jo = val // 1000000000000
            uk = (val % 1000000000000) // 100000000
            return f"{jo}조 {uk}억" if jo > 0 else f"{uk}억"
            
        df_merged['Market Cap'] = df_merged['Marcap'].apply(format_marcap)
        status_text.empty()
        return df_merged

    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_price_history(codes_list, start_date):
    data_dict = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(codes_list)
    
    for i, code in enumerate(codes_list):
        if i % 5 == 0: 
            status_text.text(f"📉 Downloading Prices: {i+1}/{total}")
            progress_bar.progress((i + 1) / total)
        try:
            df = fdr.DataReader(code, start_date)
            if not df.empty:
                data_dict[code] = df['Close']
        except: continue
    
    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(data_dict).dropna()

def run_pair_analysis(price_df, stocks_info, p_thresh, z_thresh):
    pairs = []
    sectors = stocks_info['Sector'].unique()
    
    for sector in sectors:
        sector_stocks = stocks_info[stocks_info['Sector'] == sector]
        valid_codes = [c for c in sector_stocks['Code'] if c in price_df.columns]
        
        if len(valid_codes) < 2: continue
        
        for s1, s2 in combinations(valid_codes, 2):
            # [핵심 변경 1] 가격 자체(Price) 대신 로그 가격(Log Price) 사용
            # 이유: 10만원짜리와 1만원짜리 주식의 등락폭 왜곡을 없앰
            series1 = np.log(price_df[s1])
            series2 = np.log(price_df[s2])
            
            if len(series1) < 30 or series1.std() == 0 or series2.std() == 0: continue
            
            # [핵심 변경 2] 상관계수 기준 완화 (0.8 -> 0.7)
            if series1.corr(series2) < 0.7: continue

            try:
                score, p_value, _ = coint(series1, series2)
                if p_value < p_thresh:
                    name1 = sector_stocks[sector_stocks['Code'] == s1]['Name'].values[0]
                    name2 = sector_stocks[sector_stocks['Code'] == s2]['Name'].values[0]
                    
                    x = sm.add_constant(series2)
                    model = sm.OLS(series1, x).fit()
                    
                    if len(model.params) < 2: continue
                    hedge_ratio = model.params.iloc[1]
                    
                    # Spread = Log(A) - beta * Log(B)
                    spread = series1 - (hedge_ratio * series2)
                    z_score = (spread.iloc[-1] - spread.mean()) / spread.std()
                    
                    pairs.append({
                        'Sector': sector, 
                        'Stock1': name1, 'Stock2': name2,
                        'Code1': s1, 'Code2': s2,
                        'P_value': p_value, 'Current_Z': z_score,
                        'Spread_Series': spread 
                    })
            except: continue
    return pd.DataFrame(pairs)
    
# ==========================================
# 🖥️ UI: Pair Scanner Terminal
# ==========================================
st.set_page_config(page_title="Pair Scanner Terminal", layout="wide", page_icon="📈")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    h1 { color: #ff9900; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Pair Scanner Terminal")
st.markdown("Top-Down Approach: **Scan Market** ➔ **Select Sector** ➔ **Find Alpha**")

if 'all_market_data' not in st.session_state:
    st.session_state.all_market_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# -------------------------------------------------------------------
# [STEP 1] Data Fetching
# -------------------------------------------------------------------
st.header("1️⃣ Market Scan (Data Ingestion)")
col_btn, col_msg = st.columns([1, 4])

with col_btn:
    if st.button("🔄 Scan Entire Market", type="primary"):
        df = fetch_all_naver_stocks()
        st.session_state.all_market_data = df
        st.session_state.analysis_results = None 

with col_msg:
    if st.session_state.all_market_data is not None:
        raw_df = st.session_state.all_market_data
        st.success(f"✅ Data Ready: {raw_df['Sector'].nunique()} Sectors / {len(raw_df)} Stocks")
    else:
        st.info("Click the button to fetch market data (Approx. 30s)")

# 요약 보기
if st.session_state.all_market_data is not None:
    raw_df = st.session_state.all_market_data
    
    sector_counts = raw_df['Sector'].value_counts()
    
    tab_chart, tab_table = st.tabs(["📊 Sector Distribution (Count)", "📂 Market Leaders (Top 5)"])
    
    with tab_chart:
        st.bar_chart(sector_counts, color="#ff9900")
        
    with tab_table:
        # 🌟 [NEW] 섹터별 종목 수 요약 테이블 (Top 5 위에 배치)
        st.markdown("##### 🔢 Sector Stock Counts (Summary)")
        
        # 카운트 데이터프레임 생성
        count_df = sector_counts.reset_index()
        count_df.columns = ['Sector', 'Total Stocks']
        
        # 테이블 높이를 제한(height=200)하여 너무 많은 공간을 차지하지 않게 함
        st.dataframe(
            count_df, 
            use_container_width=True, 
            hide_index=True, 
            height=200 
        )
        
        st.divider() # 구분선
        
        # 기존 Top 5 테이블
        st.markdown("##### 🏆 Market Leaders (Top 5 by Market Cap)")
        display_df = raw_df.groupby('Sector').head(5)
        st.dataframe(
            display_df[['Sector', 'Name', 'Price', 'Market Cap']], 
            use_container_width=True,
            column_config={
                "Sector": "Sector", "Name": "Name",
                "Price": "Price", "Market Cap": "Market Cap"
            },
            hide_index=True
        )

# -------------------------------------------------------------------
# [STEP 2] Analysis
# -------------------------------------------------------------------
st.divider()
st.header("2️⃣ Sector Deep Dive")

if st.session_state.all_market_data is not None:
    raw_df = st.session_state.all_market_data
    all_sectors = raw_df['Sector'].unique().tolist()
    
    sector_count_map = raw_df['Sector'].value_counts().to_dict()
    
    selected_sectors = st.multiselect(
        "Select Target Sectors:", all_sectors,
        format_func=lambda x: f"{x} ({sector_count_map.get(x, 0)} stocks)",
        default=all_sectors[:1] if len(all_sectors) > 0 else None
    )
    
    c1, c2, c3 = st.columns(3)
    lookback = c1.slider("Lookback Period (Days)", 100, 730, 365)
    z_thresh = c2.number_input("Z-Score Threshold", 1.5, 4.0, 2.0, 0.1)
    p_thresh = c3.number_input("P-value Threshold", 0.01, 0.1, 0.05, 0.01)
    
    if st.button("🚀 Run Pair Analysis", type="primary"):
        if not selected_sectors:
            st.warning("Please select at least one sector.")
        else:
            target_stocks_info = raw_df[raw_df['Sector'].isin(selected_sectors)]
            st.info(f"🧐 Analyzing {len(target_stocks_info)} stocks in selected sectors...")
            
            start_date = (datetime.now() - timedelta(days=lookback)).strftime('%Y-%m-%d')
            price_df = fetch_price_history(target_stocks_info['Code'].tolist(), start_date)
            
            if price_df.empty:
                st.error("Failed to fetch price data.")
            else:
                with st.spinner("Calculating Correlations & Cointegration..."):
                    results = run_pair_analysis(price_df, target_stocks_info, p_thresh, z_thresh)
                    st.session_state.analysis_results = (results, price_df)

# -------------------------------------------------------------------
# [STEP 3] Visualization
# -------------------------------------------------------------------
if st.session_state.analysis_results is not None:
    results, price_df = st.session_state.analysis_results
    
    if not results.empty:
        signals = results[abs(results['Current_Z']) >= z_thresh].copy()
        signals['Signal'] = np.where(signals['Current_Z'] > 0, "SHORT A / LONG B", "LONG A / SHORT B")
        
        st.divider()
        st.subheader(f"📊 Results: {len(results)} Pairs Identified")
        
        tab1, tab2 = st.tabs(["🔥 ACTIVE SIGNALS", "👀 WATCHLIST"])
        
        def draw_pair_chart(pair_data, price_df, z_limit):
            s1, s2 = pair_data['Code1'], pair_data['Code2']
            n1, n2 = pair_data['Stock1'], pair_data['Stock2']
            spread = pair_data['Spread_Series']
            
            # 시각화는 직관적인 '누적 수익률(%)'로 변환해서 보여줌
            p1 = (price_df[s1] / price_df[s1].iloc[0] - 1) * 100
            p2 = (price_df[s2] / price_df[s2].iloc[0] - 1) * 100
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
            
            # Neon Style
            ax1.plot(p1, color='#00ffcc', label=f"{n1} (Returns %)", linewidth=2) 
            ax1.plot(p2, color='#ff00ff', label=f"{n2} (Returns %)", linewidth=2)
            ax1.set_title(f"CUMULATIVE RETURNS: {n1} vs {n2}", color='#ff9900', fontsize=16, pad=15)
            ax1.legend(facecolor='#1e1e1e', edgecolor='#444444')
            ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
            
            z_score = (spread - spread.mean()) / spread.std()
            ax2.plot(z_score, color='#ffff00', label='Spread Z-Score', linewidth=1.5)
            ax2.axhline(z_limit, color='red', linestyle='--', linewidth=1)
            ax2.axhline(-z_limit, color='red', linestyle='--', linewidth=1)
            ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)
            
            ax2.fill_between(z_score.index, z_limit, z_score, where=(z_score >= z_limit), color='red', alpha=0.3)
            ax2.fill_between(z_score.index, -z_limit, z_score, where=(z_score <= -z_limit), color='red', alpha=0.3)
            ax2.set_title(f"LOG-SPREAD Z-SCORE (Current: {pair_data['Current_Z']:.2f})", color='#ff9900', fontsize=12)
            
            plt.tight_layout()
            st.pyplot(fig)

        with tab1:
            if signals.empty:
                st.info("No active signals found matching criteria.")
            else:
                st.dataframe(signals[['Sector', 'Stock1', 'Stock2', 'Current_Z', 'Signal', 'P_value']], use_container_width=True, hide_index=True)
                sel_sig = st.selectbox("Select Pair to Visualize:", signals.index, format_func=lambda i: f"{signals.loc[i,'Stock1']} vs {signals.loc[i,'Stock2']}", key='sig_sel')
                draw_pair_chart(signals.loc[sel_sig], price_df, z_thresh)

        with tab2:
            watchlist = results[abs(results['Current_Z']) < z_thresh].sort_values('P_value')
            if watchlist.empty:
                st.info("Watchlist empty.")
            else:
                st.dataframe(watchlist[['Sector', 'Stock1', 'Stock2', 'Current_Z', 'P_value']], use_container_width=True)
                sel_watch = st.selectbox("Select Pair to Visualize:", watchlist.index, format_func=lambda i: f"{watchlist.loc[i,'Stock1']} vs {watchlist.loc[i,'Stock2']}", key='watch_sel')
                draw_pair_chart(watchlist.loc[sel_watch], price_df, z_thresh)
    else:
        st.warning("No pairs found. Try relaxing the correlation or p-value thresholds.")
