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
# 📡 함수 정의
# ==========================================
@st.cache_data(ttl=3600*12)
def fetch_all_naver_stocks():
    base_url = "https://finance.naver.com/sise/sise_group.naver?type=upjong"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(base_url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        rows = soup.find('table', {'class': 'type_1'}).find_all('tr')
        
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
        total = len(sector_links)
        
        for idx, (sec_name, sec_url) in enumerate(sector_links):
            status_text.text(f"📡 System Scanning... [{idx+1}/{total}] {sec_name}")
            progress_bar.progress((idx + 1) / total)
            
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
        
        status_text.text("💰 Mapping Market Cap...")
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
        st.error(f"Error: {e}"); return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_price_history(codes_list, start_date):
    data_dict = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    total = len(codes_list)
    
    for i, code in enumerate(codes_list):
        if i % 10 == 0: 
            status_text.text(f"📉 Downloading Data: {i+1}/{total}")
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
            series1 = np.log(price_df[s1])
            series2 = np.log(price_df[s2])
            
            if len(series1) < 20 or series1.std() == 0 or series2.std() == 0: continue
            if series1.corr(series2) < 0.7: continue # Basic correlation check

            try:
                score, p_value, _ = coint(series1, series2)
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

# Session State 초기화
if 'step1_data' not in st.session_state: st.session_state.step1_data = None
if 'step2_price' not in st.session_state: st.session_state.step2_price = None
if 'step2_target' not in st.session_state: st.session_state.step2_target = None
if 'step3_results' not in st.session_state: st.session_state.step3_results = None

# -------------------------------------------------------------------
# [STEP 1] Market Scan (전체 시장 스캔)
# -------------------------------------------------------------------
st.header("1️⃣ Market Scan")
col_1, col_2 = st.columns([1, 4])

with col_1:
    if st.button("🔄 Scan Entire Market", type="primary"):
        df = fetch_all_naver_stocks()
        st.session_state.step1_data = df
        # 상위 데이터가 바뀌면 하위 데이터 초기화
        st.session_state.step2_price = None 
        st.session_state.step3_results = None

with col_2:
    if st.session_state.step1_data is not None:
        raw_df = st.session_state.step1_data
        st.success(f"✅ Market Data Loaded: {len(raw_df)} Stocks across {raw_df['Sector'].nunique()} Sectors")
    else:
        st.info("Start by scanning the market.")

if st.session_state.step1_data is not None:
    with st.expander("📂 Market Overview (Top 5)", expanded=False):
        raw_df = st.session_state.step1_data
        sector_counts = raw_df['Sector'].value_counts().reset_index()
        sector_counts.columns = ['Sector', 'Count']
        
        c1, c2 = st.columns([1, 2])
        c1.dataframe(sector_counts, use_container_width=True, hide_index=True, height=300)
        c2.dataframe(raw_df.groupby('Sector').head(5)[['Sector', 'Name', 'Price', 'Market Cap']], use_container_width=True, hide_index=True, height=300)

st.divider()

# -------------------------------------------------------------------
# [STEP 2] Data Prep (섹터 선택 & 가격 다운로드)
# -------------------------------------------------------------------
st.header("2️⃣ Data Preparation (Download)")

if st.session_state.step1_data is not None:
    raw_df = st.session_state.step1_data
    all_sectors = raw_df['Sector'].unique().tolist()
    sector_map = raw_df['Sector'].value_counts().to_dict()
    
    col_s1, col_s2 = st.columns([3, 1])
    
    with col_s1:
        selected_sectors = st.multiselect(
            "Select Target Sectors:", all_sectors,
            format_func=lambda x: f"{x} ({sector_map.get(x, 0)})",
            default=all_sectors[:1] if len(all_sectors) > 0 else None
        )
        
    with col_s2:
        # 💡 [Default] 60일 (약 3개월) 설정
        lookback = st.slider("Lookback (Days)", 30, 365, 60, help="짧을수록(60일) 최근 동조화 추세를 잘 반영합니다.")
    
    if st.button("📥 Download Price Data", type="primary"):
        if not selected_sectors:
            st.warning("Please select a sector.")
        else:
            target_stocks = raw_df[raw_df['Sector'].isin(selected_sectors)]
            start_date = (datetime.now() - timedelta(days=lookback)).strftime('%Y-%m-%d')
            
            # 다운로드 실행
            price_df = fetch_price_history(target_stocks['Code'].tolist(), start_date)
            
            if not price_df.empty:
                # 세션에 저장
                st.session_state.step2_price = price_df
                st.session_state.step2_target = target_stocks
                st.session_state.step3_results = None # 가격 데이터 바뀌면 결과 초기화
                st.rerun() # 화면 갱신
            else:
                st.error("Failed to download data.")

    # 다운로드 완료 상태 표시
    if st.session_state.step2_price is not None:
        n_stocks = len(st.session_state.step2_price.columns)
        st.success(f"✅ Price Data Ready: {n_stocks} stocks downloaded ({lookback} days history)")
    else:
        st.info("Select sectors and click Download.")

st.divider()

# -------------------------------------------------------------------
# [STEP 3] Statistical Analysis (파라미터 설정 & 분석)
# -------------------------------------------------------------------
st.header("3️⃣ Find Pairs")

if st.session_state.step2_price is not None:
    # 파라미터 튜닝 패널
    c1, c2, c3 = st.columns(3)
    z_thresh = c1.number_input("Z-Score (Signal)", 1.5, 4.0, 2.0, 0.1)
    p_thresh = c2.number_input("Max P-value", 0.01, 0.2, 0.1, 0.01) # 0.1 정도로 완화 추천
    
    if st.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Calculating Cointegration..."):
            results = run_pair_analysis(
                st.session_state.step2_price, 
                st.session_state.step2_target, 
                p_thresh, 
                z_thresh
            )
            st.session_state.step3_results = results

    # -------------------------------------------------------------------
    # [STEP 4] Results & Visualization
    # -------------------------------------------------------------------
    if st.session_state.step3_results is not None:
        results = st.session_state.step3_results
        
        if not results.empty:
            # Signal / Watchlist 분리
            signals = results[abs(results['Current_Z']) >= z_thresh].copy()
            signals['Signal'] = np.where(signals['Current_Z'] > 0, "SHORT A / LONG B", "LONG A / SHORT B")
            signals = signals.sort_values('P_value')
            
            watchlist = results[abs(results['Current_Z']) < z_thresh].sort_values('Current_Z', key=abs, ascending=False)
            
            st.write(f"### 🎯 Found {len(results)} Cointegrated Pairs")
            
            tab1, tab2 = st.tabs(["🔥 Active Signals", "👀 Watchlist"])
            
            def draw_pair_chart(pair_data, price_df, z_limit):
                s1, s2 = pair_data['Code1'], pair_data['Code2']
                n1, n2 = pair_data['Stock1'], pair_data['Stock2']
                spread = pair_data['Spread_Series']
                
                p1 = (price_df[s1] / price_df[s1].iloc[0] - 1) * 100
                p2 = (price_df[s2] / price_df[s2].iloc[0] - 1) * 100
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
                
                ax1.plot(p1, color='#00ffcc', label=n1, linewidth=2) 
                ax1.plot(p2, color='#ff00ff', label=n2, linewidth=2)
                ax1.set_title(f"Cumulative Returns: {n1} vs {n2}", color='#ff9900', fontsize=16)
                ax1.legend(facecolor='#1e1e1e'); ax1.grid(True, alpha=0.3)
                
                z_score = (spread - spread.mean()) / spread.std()
                ax2.plot(z_score, color='#ffff00', label='Z-Score')
                ax2.axhline(z_limit, c='r', ls='--'); ax2.axhline(-z_limit, c='r', ls='--'); ax2.axhline(0, c='gray')
                ax2.fill_between(z_score.index, z_limit, z_score, where=(z_score>=z_limit), color='red', alpha=0.3)
                ax2.fill_between(z_score.index, -z_limit, z_score, where=(z_score<=-z_limit), color='red', alpha=0.3)
                ax2.set_title(f"Spread Z-Score: {pair_data['Current_Z']:.2f} (P-val: {pair_data['P_value']:.4f})", color='#ff9900')
                st.pyplot(fig)

            with tab1:
                if signals.empty: st.info("No active signals.")
                else:
                    st.dataframe(signals[['Stock1', 'Stock2', 'Current_Z', 'Signal', 'P_value']], use_container_width=True, hide_index=True)
                    sel = st.selectbox("Visualize Signal:", signals.index, format_func=lambda i: f"{signals.loc[i,'Stock1']} - {signals.loc[i,'Stock2']}", key='s1')
                    draw_pair_chart(signals.loc[sel], st.session_state.step2_price, z_thresh)

            with tab2:
                if watchlist.empty: st.info("No watchlist items.")
                else:
                    st.dataframe(watchlist[['Stock1', 'Stock2', 'Current_Z', 'P_value']], use_container_width=True, hide_index=True)
                    sel = st.selectbox("Visualize Watchlist:", watchlist.index, format_func=lambda i: f"{watchlist.loc[i,'Stock1']} - {watchlist.loc[i,'Stock2']}", key='w1')
                    draw_pair_chart(watchlist.loc[sel], st.session_state.step2_price, z_thresh)
        else:
            st.warning("No pairs found. Try increasing Max P-value (e.g., 0.15).")

else:
    st.info("Waiting for Price Data... (Complete Step 2)")
