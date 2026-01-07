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
# 0. 시스템 설정 및 시각화 테마
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
        'axes.facecolor': '#111111', 'figure.facecolor': '#111111',
        'grid.color': '#444444', 'grid.linestyle': '--', 'grid.alpha': 0.5,
        'axes.edgecolor': '#888888', 'text.color': 'white',
        'xtick.color': 'white', 'ytick.color': 'white',
        'axes.labelcolor': '#ff9900', 'axes.titlesize': 14,
        'axes.titleweight': 'bold', 'lines.linewidth': 1.5
    })

init_settings()

# ==========================================
# 1. 핵심 데이터 처리 엔진
# ==========================================

@st.cache_data(ttl=3600*12)
def fetch_all_market_data():
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
                name = link_tag.text.strip()
                if "기타" not in name:
                    sector_links.append((name, "https://finance.naver.com" + link_tag['href']))
        
        all_stocks = []
        p_bar = st.progress(0)
        for idx, (name, url) in enumerate(sector_links):
            p_bar.progress((idx + 1) / len(sector_links))
            r = requests.get(url, headers=headers)
            s = BeautifulSoup(r.text, 'html.parser')
            t = s.find('table', {'class': 'type_5'})
            if not t: continue
            for tr in t.find_all('tr'):
                tds = tr.find_all('td')
                if len(tds) < 2: continue
                a = tds[0].find('a')
                if a:
                    all_stocks.append({
                        'Sector': name, 'Name': a.text.strip(), 
                        'Code': a['href'].split('code=')[-1], 'Price': tds[1].text.strip()
                    })
            time.sleep(0.01)
        p_bar.empty()
        df_naver = pd.DataFrame(all_stocks).drop_duplicates('Code')
        df_krx = fdr.StockListing('KRX')[['Code', 'Marcap']]
        df = pd.merge(df_naver, df_krx, on='Code', how='left').fillna(0)
        def format_m(v):
            v = int(v); jo = v // 1000000000000; uk = (v % 1000000000000) // 100000000
            return f"{jo}조 {uk}억" if jo > 0 else f"{uk}억"
        df['Market Cap Value'] = df['Marcap']
        df['Market Cap'] = df['Marcap'].apply(format_m)
        return df.sort_values(['Sector', 'Market Cap Value'], ascending=[True, False])
    except: return pd.DataFrame()

# 🚀 고속 병렬 다운로드 함수
def download_single_stock(code, start_date):
    try:
        df = fdr.DataReader(code, start_date)
        if not df.empty:
            return code, df['Close']
    except:
        pass
    return code, None

@st.cache_data(ttl=3600)
def fetch_prices_parallel(codes, start_date):
    """멀티스레딩을 이용한 고속 주가 수집"""
    data = {}
    total = len(codes)
    p_bar = st.progress(0)
    status_text = st.empty()
    
    # 최대 10개의 스레드를 동시에 사용
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_code = {executor.submit(download_single_stock, code, start_date): code for code in codes}
        
        completed = 0
        for future in as_completed(future_to_code):
            code, result = future.result()
            if result is not None:
                data[code] = result
            completed += 1
            p_bar.progress(completed / total)
            status_text.text(f"진행 상황: {completed}/{total} 종목 완료")
            
    p_bar.empty()
    status_text.empty()
    return pd.DataFrame(data).dropna()

def analyze_pairs(price_df, stocks_info, p_thresh, z_thresh, corr_limit):
    results = []
    corr_matrix = price_df.corr()
    sectors = stocks_info['Sector'].unique()
    for sector in sectors:
        sec_stocks = stocks_info[stocks_info['Sector'] == sector]
        codes = [c for c in sec_stocks['Code'] if c in price_df.columns]
        if len(codes) < 2: continue
        for s1, s2 in combinations(codes, 2):
            if corr_matrix.loc[s1, s2] < corr_limit: continue
            try:
                y, x_val = np.log(price_df[s1]), np.log(price_df[s2])
                score, p_val, _ = coint(y, x_val)
                if p_val < p_thresh:
                    model = sm.OLS(y, sm.add_constant(x_val)).fit()
                    hedge_ratio = model.params.iloc[1]
                    spread = y - (hedge_ratio * x_val)
                    z = (spread.iloc[-1] - spread.mean()) / spread.std()
                    results.append({
                        'Sector': sector, 'Stock1': sec_stocks[sec_stocks['Code']==s1]['Name'].values[0],
                        'Stock2': sec_stocks[sec_stocks['Code']==s2]['Name'].values[0],
                        'Code1': s1, 'Code2': s2, 'Correlation': corr_matrix.loc[s1, s2],
                        'P_value': p_val, 'Current_Z': z, 'Spread': spread
                    })
            except: continue
    return pd.DataFrame(results)

# ==========================================
# 2. 메인 화면 구성
# ==========================================
st.set_page_config(page_title="Pair Scanner Terminal", layout="wide")
st.title("Pair Scanner Terminal")

if 'market_df' not in st.session_state: st.session_state.market_df = None
if 'price_df' not in st.session_state: st.session_state.price_df = None

# --- Step 1 ---
st.header("Step 1. 시장 데이터 스캔")
if st.button("전체 종목 및 섹터 정보 조회", type="primary"):
    with st.spinner("섹터 정보를 수집 중..."):
        st.session_state.market_df = fetch_all_market_data()

if st.session_state.market_df is not None:
    st.success(f"스캔 완료: {st.session_state.market_df['Sector'].nunique()}개 섹터 확보")
    with st.expander("섹터별 시가총액 상위 종목 미리보기"):
        st.dataframe(st.session_state.market_df.groupby('Sector').head(5)[['Sector', 'Name', 'Price', 'Market Cap']], use_container_width=True, hide_index=True)

st.divider()

# --- Step 2 ---
st.header("Step 2. 데이터 다운로드 설정")
if st.session_state.market_df is not None:
    mode = st.radio("다운로드 모드 선택", 
                    ["전체 섹터 스캔 (섹터별 시총 상위 30개)", "특정 섹터 집중 스캔 (선택 섹터 내 전체 종목)"])
    
    lookback = st.slider("데이터 조회 기간 (일)", 30, 200, 60)
    
    target_info = pd.DataFrame()
    if mode == "전체 섹터 스캔 (섹터별 시총 상위 30개)":
        target_info = st.session_state.market_df.groupby('Sector').head(30)
    else:
        all_sectors = st.session_state.market_df['Sector'].unique().tolist()
        selected = st.multiselect("분석할 섹터를 선택하세요", all_sectors)
        if selected:
            target_info = st.session_state.market_df[st.session_state.market_df['Sector'].isin(selected)]

    if st.button("주가 데이터 다운로드 실행", type="secondary"):
        if target_info.empty:
            st.warning("분석할 대상을 먼저 확정해주세요.")
        else:
            start_date = (datetime.now() - timedelta(days=lookback)).strftime('%Y-%m-%d')
            with st.spinner("고속 병렬 다운로드 중..."):
                st.session_state.price_df = fetch_prices_parallel(target_info['Code'].tolist(), start_date)
                st.session_state.target_info = target_info
            st.success(f"{len(st.session_state.price_df.columns)}개 종목 데이터 준비 완료")

st.divider()

# --- Step 3 ---
st.header("Step 3. 페어 분석 및 시각화")
if st.session_state.price_df is not None:
    c1, c2, c3 = st.columns(3)
    p_thresh = c1.number_input("Max P-value", 0.01, 0.2, 0.10)
    z_thresh = c2.number_input("Z-Score 기준", 1.0, 4.0, 2.0)
    corr_min = c3.slider("최소 상관계수", 0.5, 0.95, 0.8)
    
    if st.button("분석 실행", type="primary"):
        with st.spinner("페어 분석 중..."):
            st.session_state.results = analyze_pairs(st.session_state.price_df, st.session_state.target_info, p_thresh, z_thresh, corr_min)
            
    if 'results' in st.session_state and not st.session_state.results.empty:
        res = st.session_state.results
        tab1, tab2 = st.tabs(["실시간 진입 시그널", "전체 Watchlist"])
        
        def draw_chart(pair):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
            p1 = (st.session_state.price_df[pair['Code1']] / st.session_state.price_df[pair['Code1']].iloc[0] - 1) * 100
            p2 = (st.session_state.price_df[pair['Code2']] / st.session_state.price_df[pair['Code2']].iloc[0] - 1) * 100
            ax1.plot(p1, color='#00ffcc', label=pair['Stock1'])
            ax1.plot(p2, color='#ff00ff', label=pair['Stock2'])
            ax1.set_title(f"Cumulative Returns: {pair['Stock1']} vs {pair['Stock2']}"); ax1.legend(); ax1.grid(True, alpha=0.3)
            
            z_series = (pair['Spread'] - pair['Spread'].mean()) / pair['Spread'].std()
            ax2.plot(z_series, color='#ffff00'); ax2.axhline(z_thresh, color='red', ls='--'); ax2.axhline(-z_thresh, color='red', ls='--')
            ax2.fill_between(z_series.index, z_thresh, z_series, where=(z_series>=z_thresh), color='red', alpha=0.3)
            ax2.fill_between(z_series.index, -z_thresh, z_series, where=(z_series<=-z_thresh), color='red', alpha=0.3)
            ax2.set_title(f"Z-Score Spread (Current: {pair['Current_Z']:.2f})"); ax2.grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig)

        with tab1:
            sig = res[abs(res['Current_Z']) >= z_thresh]
            if not sig.empty:
                st.dataframe(sig[['Sector', 'Stock1', 'Stock2', 'Correlation', 'Current_Z', 'P_value']], use_container_width=True, hide_index=True)
                sel = st.selectbox("차트 선택 (Signal)", sig.index, format_func=lambda x: f"{sig.loc[x, 'Stock1']} - {sig.loc[x, 'Stock2']}")
                draw_chart(sig.loc[sel])
            else: st.info("신호 없음")
        with tab2:
            st.dataframe(res[['Sector', 'Stock1', 'Stock2', 'Correlation', 'Current_Z', 'P_value']], use_container_width=True, hide_index=True)
            sel_w = st.selectbox("차트 선택 (Watchlist)", res.index, format_func=lambda x: f"{res.loc[x, 'Stock1']} - {res.loc[x, 'Stock2']}")
            draw_chart(res.loc[sel_w])
