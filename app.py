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
# 🎨 0. 스타일 설정
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
                sec_name = link_tag.text.strip()
                if "기타" in sec_name: continue
                sector_links.append((sec_name, "https://finance.naver.com" + link_tag['href']))
        
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
            time.sleep(0.01)
            
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
        if i % 10 == 0: 
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
    """ 
    [복구됨] 공적분(Cointegration) 기반 로직
    - 로그 수익률 사용
    - 상관계수 0.7 이상 (1차 필터)
    - 공적분 P-value < 0.1 (2차 필터 - 필수)
    """
    pairs = []
    sectors = stocks_info['Sector'].unique()
    
    for sector in sectors:
        sector_stocks = stocks_info[stocks_info['Sector'] == sector]
        valid_codes = [c for c in sector_stocks['Code'] if c in price_df.columns]
        
        if len(valid_codes) < 2: continue
        
        for s1, s2 in combinations(valid_codes, 2):
            series1 = np.log(price_df[s1])
            series2 = np.log(price_df[s2])
            
            if len(series1) < 20 or series1.std() == 0 or series2.std() == 0: continue
            
            # 1. 상관계수 필터 (기본적인 방향성 확인)
            if series1.corr(series2) < 0.7: continue

            try:
                # 2. 공적분 테스트 (통계적 유의성 검증)
                score, p_value, _ = coint(series1, series2)
                
                # 🔥 P-value 기준 만족 못하면 탈락 (Watchlist에도 안 넣음)
                # 다만 기준을 0.05가 아니라 0.1 정도로 잡으면 많이 나옴
                if p_value < p_thresh:
                    name1 = sector_stocks[sector_stocks['Code'] == s1]['Name'].values[0]
                    name2 = sector_stocks[sector_stocks['Code'] == s2]['Name'].values[0]
                    
                    x = sm.add_constant(series2)
                    model = sm.OLS(series1, x).fit()
                    
                    if len(model.params) < 2: continue
                    hedge_ratio = model.params.iloc[1]
                    
                    spread = series1 - (hedge_ratio * series2)
                    z_score = (spread.iloc[-1] - spread.mean()) / spread.std()
                    
                    pairs.append({
                        'Sector': sector, 
                        'Stock1': name1, 'Stock2': name2,
                        'Code1': s1, 'Code2': s2,
                        'P_value': p_value,
                        'Current_Z': z_score,
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
st.markdown("Method: **Log-Price Cointegration** (Mean Reversion Strategy)")

if 'all_market_data' not in st.session_state:
    st.session_state.all_market_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# -------------------------------------------------------------------
# [STEP 1] Data Fetching
# -------------------------------------------------------------------
st.header("1️⃣ Market Scan")
col_btn, col_msg = st.columns([1, 4])

with col_btn:
    if st.button("🔄 Scan Entire Market", type="primary"):
        df = fetch_all_naver_stocks()
        st.session_state.all_market_data = df
        st.session_state.analysis_results = None 

with col_msg:
    if st.session_state.all_market_data is not None:
        raw_df = st.session_state.all_market_data
        st.success(f"✅ Data Ready: {len(raw_df)} stocks / {raw_df['Sector'].nunique()} sectors")
    else:
        st.info("Start by scanning the market data.")

# 요약 보기
if st.session_state.all_market_data is not None:
    raw_df = st.session_state.all_market_data
    sector_counts = raw_df['Sector'].value_counts()
    
    tab_chart, tab_table = st.tabs(["📊 Sector Count", "📂 Top 5 Leaders"])
    
    with tab_chart:
        st.bar_chart(sector_counts, color="#ff9900")
        
    with tab_table:
        st.markdown("##### 🔢 Sector Stock Counts")
        count_df = sector_counts.reset_index()
        count_df.columns = ['Sector', 'Count']
        st.dataframe(count_df, use_container_width=True, hide_index=True, height=200)
        
        st.divider()
        st.markdown("##### 🏆 Sector Leaders (Top 5)")
        display_df = raw_df.groupby('Sector').head(5)
        st.dataframe(display_df[['Sector', 'Name', 'Price', 'Market Cap']], use_container_width=True, hide_index=True)

# -------------------------------------------------------------------
# [STEP 2] Analysis
# -------------------------------------------------------------------
st.divider()
st.header("2️⃣ Deep Dive & Analysis")

if st.session_state.all_market_data is not None:
    raw_df = st.session_state.all_market_data
    all_sectors = raw_df['Sector'].unique().tolist()
    sector_count_map = raw_df['Sector'].value_counts().to_dict()
    
    selected_sectors = st.multiselect(
        "Select Target Sectors:", all_sectors,
        format_func=lambda x: f"{x} ({sector_count_map.get(x, 0)})",
        default=all_sectors[:1] if len(all_sectors) > 0 else None
    )
    
    c1, c2, c3 = st.columns(3)
    
    # 🌟 [요청 반영] Lookback Days Default = 60일
    lookback = c1.slider("Lookback (Days)", 30, 365, 60, help="짧을수록(60일) 최근 동조화 현상을 잘 포착합니다.") 
    z_thresh = c2.number_input("Z-Score Threshold (Signal)", 1.5, 4.0, 2.0, 0.1)
    # 🌟 [요청 반영] P-value Filter 복구 (Default 0.05 ~ 0.1)
    p_thresh = c3.number_input("Max P-value (Cointegration)", 0.01, 0.2, 0.1, 0.01, help="0.05: 엄격, 0.1: 보통")
    
    if st.button("🚀 Run Pair Analysis", type="primary"):
        if not selected_sectors:
            st.warning("Select a sector first.")
        else:
            target_stocks_info = raw_df[raw_df['Sector'].isin(selected_sectors)]
            st.info(f"Scanning {len(target_stocks_info)} stocks... (Lookback: {lookback} days)")
            
            start_date = (datetime.now() - timedelta(days=lookback)).strftime('%Y-%m-%d')
            price_df = fetch_price_history(target_stocks_info['Code'].tolist(), start_date)
            
            if price_df.empty:
                st.error("No price data.")
            else:
                with st.spinner("Calculating Cointegration (Mean Reversion)..."):
                    results = run_pair_analysis(price_df, target_stocks_info, p_thresh, z_thresh)
                    st.session_state.analysis_results = (results, price_df)

# -------------------------------------------------------------------
# [STEP 3] Results
# -------------------------------------------------------------------
if st.session_state.analysis_results is not None:
    results, price_df = st.session_state.analysis_results
    
    if not results.empty:
        # Z-Score 기준으로 Signal / Watchlist 분리
        # Watchlist의 정의: 공적분(P-val < Thresh)은 통과했으나, Z-Score가 낮아 아직 진입 시점이 아닌 것
        
        signals = results[abs(results['Current_Z']) >= z_thresh].copy()
        signals['Signal'] = np.where(signals['Current_Z'] > 0, "SHORT A / LONG B", "LONG A / SHORT B")
        signals = signals.sort_values(by='P_value') # 통계적 유의성 높은 순
        
        watchlist = results[abs(results['Current_Z']) < z_thresh].sort_values(by='Current_Z', key=abs, ascending=False)
        
        st.divider()
        st.subheader(f"📊 Identified Pairs: {len(results)} Total (Cointegrated)")
        
        tab1, tab2 = st.tabs(["🔥 Active Signals (Trade Now)", "👀 Watchlist (Wait for Spread)"])
        
        def draw_pair_chart(pair_data, price_df, z_limit):
            s1, s2 = pair_data['Code1'], pair_data['Code2']
            n1, n2 = pair_data['Stock1'], pair_data['Stock2']
            spread = pair_data['Spread_Series']
            
            p1 = (price_df[s1] / price_df[s1].iloc[0] - 1) * 100
            p2 = (price_df[s2] / price_df[s2].iloc[0] - 1) * 100
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
            
            ax1.plot(p1, color='#00ffcc', label=f"{n1}", linewidth=2) 
            ax1.plot(p2, color='#ff00ff', label=f"{n2}", linewidth=2)
            ax1.set_title(f"Cumulative Returns: {n1} vs {n2}", color='#ff9900', fontsize=16)
            ax1.legend(facecolor='#1e1e1e')
            ax1.grid(True, alpha=0.3)
            
            z_score = (spread - spread.mean()) / spread.std()
            ax2.plot(z_score, color='#ffff00', label='Spread Z-Score')
            ax2.axhline(z_limit, c='r', ls='--'); ax2.axhline(-z_limit, c='r', ls='--'); ax2.axhline(0, c='gray')
            ax2.fill_between(z_score.index, z_limit, z_score, where=(z_score>=z_limit), color='red', alpha=0.3)
            ax2.fill_between(z_score.index, -z_limit, z_score, where=(z_score<=-z_limit), color='red', alpha=0.3)
            ax2.set_title(f"Spread Z-Score: {pair_data['Current_Z']:.2f} (P-val: {pair_data['P_value']:.4f})", color='#ff9900')
            
            st.pyplot(fig)

        with tab1:
            if signals.empty:
                st.info("No cointegrated pairs are currently deviating enough (Z-score low). Check Watchlist.")
            else:
                st.dataframe(
                    signals[['Stock1', 'Stock2', 'Current_Z', 'Signal', 'P_value']], 
                    use_container_width=True, hide_index=True
                )
                sel = st.selectbox("Visualize Signal:", signals.index, format_func=lambda i: f"{signals.loc[i,'Stock1']} - {signals.loc[i,'Stock2']}", key='s1')
                draw_pair_chart(signals.loc[sel], price_df, z_thresh)

        with tab2:
            if watchlist.empty:
                st.info("No pairs passed the cointegration test.")
            else:
                st.markdown("**These pairs move together (Cointegrated) but spread is currently normal.**")
                st.dataframe(
                    watchlist[['Stock1', 'Stock2', 'Current_Z', 'P_value']], 
                    use_container_width=True, hide_index=True
                )
                sel = st.selectbox("Visualize Watchlist:", watchlist.index, format_func=lambda i: f"{watchlist.loc[i,'Stock1']} - {watchlist.loc[i,'Stock2']}", key='w1')
                draw_pair_chart(watchlist.loc[sel], price_df, z_thresh)
    else:
        st.warning("No cointegrated pairs found. Try increasing P-value threshold or changing the sector.")
