import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from itertools import combinations
from datetime import datetime, timedelta
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 0. 환경 및 시각화 설정
# ==========================================
def init_settings():
    font_path = "NanumGothic.ttf"
    if not os.path.exists(font_path):
        url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
        response = requests.get(url)
        with open(font_path, "wb") as f: f.write(response.content)
    fm.fontManager.addfont(font_path)
    plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False 
    plt.style.use('dark_background')

init_settings()

# ==========================================
# 1. 핵심 분석 엔진 (ADF, Half-life 포함)
# ==========================================

def calculate_half_life(spread):
    spread_lag = spread.shift(1)
    spread_diff = spread.diff()
    valid = ~spread_lag.isna() & ~spread_diff.isna()
    if valid.sum() < 10: return np.nan
    res = sm.OLS(spread_diff[valid], sm.add_constant(spread_lag[valid])).fit()
    lambda_val = -res.params.iloc[1]
    if lambda_val <= 0: return np.nan
    return np.log(2) / lambda_val

@st.cache_data(ttl=3600*12)
def fetch_market_structure():
    base_url = "https://finance.naver.com/sise/sise_group.naver?type=upjong"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(base_url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        rows = soup.find('table', {'class': 'type_1'}).find_all('tr')
        sector_links = [(r.find('a').text.strip(), "https://finance.naver.com" + r.find('a')['href']) 
                        for r in rows if r.find('a') and "기타" not in r.find('a').text]
        all_stocks = []
        for idx, (name, url) in enumerate(sector_links):
            r = requests.get(url, headers=headers)
            soup_sec = BeautifulSoup(r.text, 'html.parser')
            table = soup_sec.find('table', {'class': 'type_5'})
            if not table: continue
            for tr in table.find_all('tr'):
                tds = tr.find_all('td')
                if len(tds) < 2 or not tds[0].find('a'): continue
                a = tds[0].find('a')
                code = a['href'].split('code=')[-1]
                all_stocks.append({'Sector': name, 'Name': f"{a.text.strip()} ({code})", 'Code': code})
        df = pd.merge(pd.DataFrame(all_stocks).drop_duplicates('Code'), fdr.StockListing('KRX')[['Code', 'Marcap']], on='Code', how='left').fillna(0)
        return df.sort_values(['Sector', 'Marcap'], ascending=[True, False])
    except: return pd.DataFrame()

def download_unit(code, start_date):
    try:
        df = fdr.DataReader(code, start_date)
        return (code, df['Close']) if not df.empty else (code, None)
    except: return (code, None)

@st.cache_data(ttl=3600)
def fetch_prices_parallel(codes, start_date):
    data = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_code = {executor.submit(download_unit, c, start_date): c for c in codes}
        for future in as_completed(future_to_code):
            code, res = future.result()
            if res is not None: data[code] = res
    return pd.DataFrame(data)

def analyze_pairs_refined(price_df, stocks_info, p_thresh, z_thresh, corr_limit):
    results = []
    sectors = stocks_info['Sector'].unique()
    for sector in sectors:
        sec_stocks = stocks_info[stocks_info['Sector'] == sector]
        codes = [c for c in sec_stocks['Code'] if c in price_df.columns]
        if len(codes) < 2: continue
        for s1, s2 in combinations(codes, 2):
            pair_data = price_df[[s1, s2]].dropna()
            if len(pair_data) < 30: continue 
            corr = pair_data[s1].corr(pair_data[s2])
            if corr < corr_limit: continue
            try:
                y, x_val = np.log(pair_data[s1]), np.log(pair_data[s2])
                model = sm.OLS(y, sm.add_constant(x_val)).fit()
                alpha, beta = model.params.iloc[0], model.params.iloc[1]
                spread = y - (beta * x_val + alpha)
                adf_p = adfuller(spread)[1]
                
                if adf_p < p_thresh:
                    half_life = calculate_half_life(spread)
                    spread_mean, spread_std = spread.mean(), spread.std()
                    z_score = (spread.iloc[-1] - spread_mean) / spread_std
                    results.append({
                        'Sector': sector, 'Stock1': sec_stocks[sec_stocks['Code']==s1]['Name'].values[0],
                        'Stock2': sec_stocks[sec_stocks['Code']==s2]['Name'].values[0],
                        'Code1': s1, 'Code2': s2, 'Alpha': alpha, 'Beta': beta,
                        'ADF_P': adf_p, 'Half_Life': half_life, 'Spread_Mean': spread_mean,
                        'Spread_Std': spread_std, 'Current_Z': z_score, 'Spread': spread
                    })
            except: continue
    return pd.DataFrame(results)

# ==========================================
# 2. 메인 UI 및 시각화
# ==========================================
st.set_page_config(page_title="Pair Scanner Terminal", layout="wide")
st.title("📊 Pair Scanner Terminal")

if 'm_df' not in st.session_state: st.session_state.m_df = None
if 'p_df' not in st.session_state: st.session_state.p_df = None

# Step 1 & 2 (생략 - 이전과 동일)
st.header("1. 시장 데이터 스캔")
if st.button("전체 종목 현황 스캔", type="primary"):
    st.session_state.m_df = fetch_market_structure()

if st.session_state.m_df is not None:
    st.header("2. 데이터 로드 및 전처리")
    mode = st.radio("분석 범위", ["전체 섹터 TOP 10", "특정 섹터 선택"])
    lookback = st.slider("조회 기간 (일)", 30, 500, 180)
    
    target = pd.DataFrame()
    if mode == "전체 섹터 TOP 10":
        target = st.session_state.m_df.groupby('Sector').head(10)
    else:
        sel = st.multiselect("섹터 선택", st.session_state.m_df['Sector'].unique())
        if sel: target = st.session_state.m_df[st.session_state.m_df['Sector'].isin(sel)]

    if st.button("주가 데이터 다운로드"):
        start = (datetime.now() - timedelta(days=lookback)).strftime('%Y-%m-%d')
        st.session_state.p_df = fetch_prices_parallel(target['Code'].tolist(), start)
        st.session_state.target_info = target
        st.success(f"{len(st.session_state.p_df.columns)}개 종목 로드 완료")

st.divider()

# Step 3: Action 목록 vs Watchlist 분리
st.header("3. 페어 분석 전략 리포트")
if st.session_state.p_df is not None:
    c1, c2, c3 = st.columns(3)
    p_crit = c1.number_input("Max ADF P-value", 0.01, 0.5, 0.1)
    z_crit = c2.number_input("Z-Score Threshold (Action 기준)", 1.0, 5.0, 2.0)
    corr_crit = c3.slider("Min Correlation", 0.5, 0.99, 0.8)
    
    if st.button("분석 실행", type="primary"):
        with st.spinner("퀀트 분석 엔진 가동 중..."):
            res = analyze_pairs_refined(st.session_state.p_df, st.session_state.target_info, p_crit, z_crit, corr_crit)
            st.session_state.res = res

    if 'res' in st.session_state and not st.session_state.res.empty:
        results = st.session_state.res
        
        # 💡 [핵심] Action 목록과 Watchlist 분리 로직
        action_list = results[abs(results['Current_Z']) >= z_crit].sort_values('Current_Z', ascending=False)
        watchlist = results[abs(results['Current_Z']) < z_crit].sort_values('ADF_P')

        tab1, tab2 = st.tabs([f"🔥 Action 목록 ({len(action_list)})", f"🔍 Watchlist ({len(watchlist)})"])
        
        def display_pair_detail(pair_row, key_prefix):
            # 1. 차트
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
            common_idx = pair_row['Spread'].index
            p1_raw = st.session_state.p_df[pair_row['Code1']].loc[common_idx]
            p2_raw = st.session_state.p_df[pair_row['Code2']].loc[common_idx]
            p1_norm = (p1_raw / p1_raw.iloc[0] - 1) * 100
            p2_norm = (p2_raw / p2_raw.iloc[0] - 1) * 100
            
            ax1.plot(p1_norm, color='#00ffcc', label=pair_row['Stock1'])
            ax1.plot(p2_norm, color='#ff00ff', label=pair_row['Stock2'])
            ax1.set_title("Relative Returns (%)"); ax1.legend(); ax1.grid(True, alpha=0.3)
            
            z_s = (pair_row['Spread'] - pair_row['Spread_Mean']) / pair_row['Spread_Std']
            ax2.plot(z_s, color='#ffff00')
            ax2.axhline(z_crit, color='red', ls='--'); ax2.axhline(-z_crit, color='red', ls='--')
            ax2.set_title(f"Z-Score (Alpha: {pair_row['Alpha']:.4f}, Beta: {pair_row['Beta']:.4f}, Half-life: {pair_row['Half_Life']:.1f}d)"); ax2.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # 2. 데이터 상세보기 (Date, P1, P2, Zscore)
            st.subheader("📋 분석 데이터 상세보기")
            detail_df = pd.DataFrame({
                'Date': common_idx.strftime('%Y-%m-%d'),
                'P1': p1_raw.values,
                'P2': p2_raw.values,
                'Zscore': z_s.values
            }).sort_values('Date', ascending=False)
            st.dataframe(detail_df, use_container_width=True, hide_index=True)

        with tab1:
            if action_list.empty:
                st.info("현재 진입 임계치(Z-Score)를 초과한 종목이 없습니다.")
            else:
                st.dataframe(action_list[['Sector', 'Stock1', 'Stock2', 'Beta', 'ADF_P', 'Half_Life', 'Current_Z']], use_container_width=True, hide_index=True)
                sel_action = st.selectbox("Action 대상 선택", action_list.index, format_func=lambda x: f"{action_list.loc[x, 'Stock1']} - {action_list.loc[x, 'Stock2']}", key="action_sel")
                display_pair_detail(action_list.loc[sel_action], "action")

        with tab2:
            if watchlist.empty:
                st.info("감시 리스트가 비어 있습니다.")
            else:
                st.dataframe(watchlist[['Sector', 'Stock1', 'Stock2', 'Beta', 'ADF_P', 'Half_Life', 'Current_Z']], use_container_width=True, hide_index=True)
                sel_watch = st.selectbox("Watchlist 대상 선택", watchlist.index, format_func=lambda x: f"{watchlist.loc[x, 'Stock1']} - {watchlist.loc[x, 'Stock2']}", key="watch_sel")
                display_pair_detail(watchlist.loc[sel_watch], "watch")
