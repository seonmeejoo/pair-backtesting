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
# 1. 고속 데이터 처리 엔진
# ==========================================

@st.cache_data(ttl=3600*12)
def fetch_market_structure():
    """네이버 업종별 데이터 수집"""
    base_url = "https://finance.naver.com/sise/sise_group.naver?type=upjong"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(base_url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        rows = soup.find('table', {'class': 'type_1'}).find_all('tr')
        sector_links = [(r.find('a').text.strip(), "https://finance.naver.com" + r.find('a')['href']) 
                        for r in rows if r.find('a') and "기타" not in r.find('a').text]
        
        all_stocks = []
        p_bar = st.progress(0)
        for idx, (name, url) in enumerate(sector_links):
            p_bar.progress((idx + 1) / len(sector_links))
            r = requests.get(url, headers=headers)
            soup_sec = BeautifulSoup(r.text, 'html.parser')
            table = soup_sec.find('table', {'class': 'type_5'})
            if not table: continue
            for tr in table.find_all('tr'):
                tds = tr.find_all('td')
                if len(tds) < 2 or not tds[0].find('a'): continue
                a = tds[0].find('a')
                all_stocks.append({'Sector': name, 'Name': a.text.strip(), 'Code': a['href'].split('code=')[-1], 'Price': tds[1].text.strip()})
            time.sleep(0.01)
        p_bar.empty()
        df = pd.merge(pd.DataFrame(all_stocks).drop_duplicates('Code'), fdr.StockListing('KRX')[['Code', 'Marcap']], on='Code', how='left').fillna(0)
        df['Market Cap Text'] = df['Marcap'].apply(lambda v: f"{int(v)//1000000000000}조 {int(v)%1000000000000//100000000}억" if v >= 1000000000000 else f"{int(v)//100000000}억")
        return df.sort_values(['Sector', 'Marcap'], ascending=[True, False])
    except: return pd.DataFrame()

def download_unit(code, start_date):
    try:
        df = fdr.DataReader(code, start_date)
        return (code, df['Close']) if not df.empty else (code, None)
    except: return (code, None)

@st.cache_data(ttl=3600)
def fetch_prices_parallel(codes, start_date):
    """병렬 다운로드 (dropna 제거로 데이터 보존)"""
    data = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_code = {executor.submit(download_unit, c, start_date): c for c in codes}
        for future in as_completed(future_to_code):
            code, res = future.result()
            if res is not None: data[code] = res
    return pd.DataFrame(data) # 🚨 여기서 dropna()를 절대 하지 않음

def analyze_pairs_refined(price_df, stocks_info, p_thresh, z_thresh, corr_limit):
    """페어별 개별 정렬 분석 엔진"""
    results = []
    sectors = stocks_info['Sector'].unique()
    
    for sector in sectors:
        sec_stocks = stocks_info[stocks_info['Sector'] == sector]
        codes = [c for c in sec_stocks['Code'] if c in price_df.columns]
        if len(codes) < 2: continue
        
        for s1, s2 in combinations(codes, 2):
            # 🚨 [핵심] 두 종목만 따로 떼어내서 날짜를 맞춤
            pair_data = price_df[[s1, s2]].dropna()
            if len(pair_data) < 30: continue 
            
            # 1차 상관계수 필터
            corr = pair_data[s1].corr(pair_data[s2])
            if corr < corr_limit: continue
            
            try:
                # 2차 공적분 검정 (로그 가격 모델)
                y, x = np.log(pair_data[s1]), np.log(pair_data[s2])
                score, p_val, _ = coint(y, x)
                
                if p_val < p_thresh:
                    model = sm.OLS(y, sm.add_constant(x)).fit()
                    spread = y - (model.params.iloc[1] * x)
                    z = (spread.iloc[-1] - spread.mean()) / spread.std()
                    
                    results.append({
                        'Sector': sector, 'Stock1': sec_stocks[sec_stocks['Code']==s1]['Name'].values[0],
                        'Stock2': sec_stocks[sec_stocks['Code']==s2]['Name'].values[0],
                        'Code1': s1, 'Code2': s2, 'Correlation': corr, 'P_value': p_val, 'Current_Z': z, 'Spread': spread
                    })
            except: continue
    return pd.DataFrame(results)

# ==========================================
# 2. 메인 UI 구성
# ==========================================
st.set_page_config(page_title="Pair Scanner Terminal", layout="wide")
st.title("Pair Scanner Terminal")

if 'm_df' not in st.session_state: st.session_state.m_df = None
if 'p_df' not in st.session_state: st.session_state.p_df = None

# Step 1
st.header("1. 시장 데이터 스캔")
if st.button("전체 종목 조회", type="primary"):
    st.session_state.m_df = fetch_market_structure()

if st.session_state.m_df is not None:
    st.success(f"스캔 완료: {st.session_state.m_df['Sector'].nunique()}개 섹터")
    with st.expander("섹터별 시총 상위 5개 미리보기"):
        st.dataframe(st.session_state.m_df.groupby('Sector').head(5)[['Sector', 'Name', 'Price', 'Market Cap Text']], use_container_width=True, hide_index=True)

st.divider()

# Step 2
st.header("2. 분석 대상 및 주가 데이터")
if st.session_state.m_df is not None:
    mode = st.radio("분석 모드", ["전체 섹터 (섹터별 시총 상위 30개)", "특정 섹터 집중 (선택 섹터 전 종목)"])
    lookback = st.slider("조회 기간 (일)", 30, 365, 60)
    
    target = pd.DataFrame()
    if mode == "전체 섹터 (섹터별 시총 상위 30개)":
        target = st.session_state.m_df.groupby('Sector').head(30)
    else:
        sel = st.multiselect("섹터 선택", st.session_state.m_df['Sector'].unique())
        if sel: target = st.session_state.m_df[st.session_state.m_df['Sector'].isin(sel)]

    if st.button("주가 다운로드 시작"):
        if not target.empty:
            start = (datetime.now() - timedelta(days=lookback)).strftime('%Y-%m-%d')
            with st.spinner("고속 병렬 다운로드 중..."):
                st.session_state.p_df = fetch_prices_parallel(target['Code'].tolist(), start)
                st.session_state.target_info = target
            st.success(f"{len(st.session_state.p_df.columns)}개 종목 로드 완료")

st.divider()

# Step 3
st.header("3. 페어 분석 및 전략 실행")
if st.session_state.p_df is not None:
    c1, c2, c3 = st.columns(3)
    p_crit = c1.number_input("Max P-value (Cointegration)", 0.01, 0.5, 0.1)
    z_crit = c2.number_input("Z-Score Threshold", 1.0, 5.0, 2.0)
    corr_crit = c3.slider("Min Correlation", 0.5, 0.99, 0.8)
    
    if st.button("분석 실행", type="primary"):
        with st.spinner("공적분 연산 중..."):
            res = analyze_pairs_refined(st.session_state.p_df, st.session_state.target_info, p_crit, z_crit, corr_crit)
            st.session_state.res = res

    if 'res' in st.session_state and not st.session_state.res.empty:
        results = st.session_state.res
        tab1, tab2 = st.tabs(["🔥 진입 시그널", "🔍 전체 Watchlist"])
        
        def draw_bloomberg(pair):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
            p1 = (st.session_state.p_df[pair['Code1']].loc[pair['Spread'].index] / st.session_state.p_df[pair['Code1']].loc[pair['Spread'].index].iloc[0] - 1) * 100
            p2 = (st.session_state.p_df[pair['Code2']].loc[pair['Spread'].index] / st.session_state.p_df[pair['Code2']].loc[pair['Spread'].index].iloc[0] - 1) * 100
            ax1.plot(p1, color='#00ffcc', label=pair['Stock1']); ax1.plot(p2, color='#ff00ff', label=pair['Stock2'])
            ax1.set_title(f"Returns: {pair['Stock1']} vs {pair['Stock2']}"); ax1.legend(); ax1.grid(True, alpha=0.3)
            z_s = (pair['Spread'] - pair['Spread'].mean()) / pair['Spread'].std()
            ax2.plot(z_s, color='#ffff00'); ax2.axhline(z_crit, color='red', ls='--'); ax2.axhline(-z_crit, color='red', ls='--')
            ax2.fill_between(z_s.index, z_crit, z_s, where=(z_s>=z_crit), color='red', alpha=0.3)
            ax2.fill_between(z_s.index, -z_crit, z_s, where=(z_s<=-z_crit), color='red', alpha=0.3)
            ax2.set_title(f"Z-Score Spread: {pair['Current_Z']:.2f}"); ax2.grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig)

        with tab1:
            sig = results[abs(results['Current_Z']) >= z_crit]
            if not sig.empty:
                st.dataframe(sig[['Sector', 'Stock1', 'Stock2', 'Correlation', 'Current_Z', 'P_value']], use_container_width=True, hide_index=True)
                sel = st.selectbox("시그널 차트", sig.index, format_func=lambda x: f"{sig.loc[x, 'Stock1']} - {sig.loc[x, 'Stock2']}")
                draw_bloomberg(sig.loc[sel])
            else: st.info("현재 기준 충족 페어 없음")
        with tab2:
            st.dataframe(results[['Sector', 'Stock1', 'Stock2', 'Correlation', 'Current_Z', 'P_value']], use_container_width=True, hide_index=True)
            sel_w = st.selectbox("관심 페어 차트", results.index, format_func=lambda x: f"{results.loc[x, 'Stock1']} - {results.loc[x, 'Stock2']}")
            draw_bloomberg(results.loc[sel_w])
    elif 'res' in st.session_state: st.warning("발견된 페어 없음")
