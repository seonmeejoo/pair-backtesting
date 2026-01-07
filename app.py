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
# 0. 환경 및 스타일 설정 (Bloomberg Style)
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
# 1. 데이터 수집 및 가공 함수
# ==========================================
@st.cache_data(ttl=3600*12)
def fetch_all_market_data():
    """네이버 업종별 전체 종목 수집 및 시가총액 매핑"""
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
        progress_bar = st.progress(0)
        for idx, (name, url) in enumerate(sector_links):
            progress_bar.progress((idx + 1) / len(sector_links))
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
                        'Sector': name, 
                        'Name': a.text.strip(), 
                        'Code': a['href'].split('code=')[-1],
                        'Price': tds[1].text.strip()
                    })
            time.sleep(0.01)
        progress_bar.empty()
        
        df_naver = pd.DataFrame(all_stocks).drop_duplicates('Code')
        df_krx = fdr.StockListing('KRX')[['Code', 'Marcap']]
        df = pd.merge(df_naver, df_krx, on='Code', how='left').fillna(0)
        
        # 시가총액 포맷팅
        def format_m(v):
            v = int(v)
            jo = v // 1000000000000
            uk = (v % 1000000000000) // 100000000
            return f"{jo}조 {uk}억" if jo > 0 else f"{uk}억"
            
        df['Market Cap Value'] = df['Marcap'] # 정렬용 숫자
        df['Market Cap'] = df['Marcap'].apply(format_m)
        return df.sort_values(['Sector', 'Market Cap Value'], ascending=[True, False])
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_prices(codes, start_date):
    """주가 데이터 병렬 수집 (딕셔너리 구조)"""
    data = {}
    p_bar = st.progress(0)
    for i, code in enumerate(codes):
        try:
            df = fdr.DataReader(code, start_date)
            if not df.empty: data[code] = df['Close']
        except: continue
        p_bar.progress((i + 1) / len(codes))
    p_bar.empty()
    return pd.DataFrame(data).dropna()

def analyze_pairs(price_df, stocks_info, p_thresh, z_thresh, corr_limit=0.8):
    """상관계수 선검사 후 공적분 분석 실행"""
    results = []
    # 1. 상관계수 행렬 선계산 (Vectorized)
    corr_matrix = price_df.corr()
    
    sectors = stocks_info['Sector'].unique()
    for sector in sectors:
        sec_stocks = stocks_info[stocks_info['Sector'] == sector]
        codes = [c for c in sec_stocks['Code'] if c in price_df.columns]
        
        if len(codes) < 2: continue
        
        for s1, s2 in combinations(codes, 2):
            # 2. Fast Screening: 상관계수 0.8 미만은 공적분 검사 생략
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
                        'Sector': sector,
                        'Stock1': sec_stocks[sec_stocks['Code']==s1]['Name'].values[0],
                        'Stock2': sec_stocks[sec_stocks['Code']==s2]['Name'].values[0],
                        'Code1': s1, 'Code2': s2,
                        'Correlation': corr_matrix.loc[s1, s2],
                        'P_value': p_val, 'Current_Z': z, 'Spread': spread
                    })
            except: continue
    return pd.DataFrame(results)

# ==========================================
# 2. 메인 UI 및 실행 로직
# ==========================================
st.set_page_config(page_title="Pair Scanner Terminal", layout="wide")

if 'market_df' not in st.session_state: st.session_state.market_df = None
if 'price_df' not in st.session_state: st.session_state.price_df = None

# --- Step 1: 시장 전체 데이터 조회 ---
st.header("1. 시장 데이터 스캔 및 업종별 현황")
if st.button("전체 종목 및 섹터 조회", type="primary"):
    with st.spinner("네이버 금융 데이터 수집 중..."):
        st.session_state.market_df = fetch_all_market_data()

if st.session_state.market_df is not None:
    df = st.session_state.market_df
    st.success(f"스캔 완료: {df['Sector'].nunique()}개 섹터 (기타 제외)")
    
    with st.expander("섹터별 시가총액 TOP 5 리스트 확인"):
        top5_display = df.groupby('Sector').head(5)
        st.dataframe(top5_display[['Sector', 'Name', 'Price', 'Market Cap']], use_container_width=True, hide_index=True)

st.divider()

# --- Step 2: 분석 준비 (Top 30 필터링 및 다운로드) ---
st.header("2. 분석 데이터 준비 (섹터별 상위 30개)")
if st.session_state.market_df is not None:
    all_sectors = st.session_state.market_df['Sector'].unique().tolist()
    selected_sectors = st.multiselect("분석할 섹터 선택", all_sectors, default=all_sectors[:2])
    lookback = st.slider("데이터 조회 기간 (일)", 30, 200, 60)
    
    if st.button("주가 데이터 다운로드"):
        # 섹터별 상위 30개 필터링 (Quality Filter)
        target_info = st.session_state.market_df[st.session_state.market_df['Sector'].isin(selected_sectors)]
        target_info = target_info.groupby('Sector').head(30)
        
        start_date = (datetime.now() - timedelta(days=lookback)).strftime('%Y-%m-%d')
        with st.spinner(f"{len(target_info)}개 종목 주가 다운로드 중..."):
            st.session_state.price_df = fetch_prices(target_info['Code'].tolist(), start_date)
            st.session_state.target_info = target_info
        st.success(f"{len(st.session_state.price_df.columns)}개 종목 데이터 준비 완료")

st.divider()

# --- Step 3: 전략 실행 및 시각화 ---
st.header("3. 페어 분석 및 전략 실행")
if st.session_state.price_df is not None:
    col1, col2, col3 = st.columns(3)
    p_thresh = col1.number_input("Max P-value (공적분 기준)", 0.01, 0.2, 0.10, 0.01)
    z_thresh = col2.number_input("Z-Score Threshold (진입 기준)", 1.0, 4.0, 2.0, 0.1)
    corr_min = col3.slider("최소 상관계수 (Pre-screening)", 0.5, 0.95, 0.8)
    
    if st.button("분석 실행", type="primary"):
        with st.spinner("통계 연산 중..."):
            results = analyze_pairs(st.session_state.price_df, st.session_state.target_info, p_thresh, z_thresh, corr_min)
            st.session_state.results = results
            
    if 'results' in st.session_state and not st.session_state.results.empty:
        res = st.session_state.results
        st.subheader(f"발견된 페어: {len(res)}건")
        
        tab1, tab2 = st.tabs(["🔥 실시간 시그널", "🔍 전체 Watchlist"])
        
        def draw_chart(pair):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
            # 주가 차트 (누적 수익률)
            p1 = (st.session_state.price_df[pair['Code1']] / st.session_state.price_df[pair['Code1']].iloc[0] - 1) * 100
            p2 = (st.session_state.price_df[pair['Code2']] / st.session_state.price_df[pair['Code2']].iloc[0] - 1) * 100
            ax1.plot(p1, color='#00ffcc', label=pair['Stock1'])
            ax1.plot(p2, color='#ff00ff', label=pair['Stock2'])
            ax1.set_title(f"Cumulative Returns: {pair['Stock1']} vs {pair['Stock2']}")
            ax1.legend()
            
            # Z-Score 차트
            z_series = (pair['Spread'] - pair['Spread'].mean()) / pair['Spread'].std()
            ax2.plot(z_score, color='#ffff00', label='Spread Z-Score')
            ax2.axhline(z_thresh, color='red', linestyle='--')
            ax2.axhline(-z_thresh, color='red', linestyle='--')
            ax2.axhline(0, color='gray', alpha=0.5)
            ax2.fill_between(z_series.index, z_thresh, z_series, where=(z_series>=z_thresh), color='red', alpha=0.3)
            ax2.fill_between(z_series.index, -z_thresh, z_series, where=(z_series<=-z_thresh), color='red', alpha=0.3)
            ax2.set_title(f"Z-Score Spread (Current: {pair['Current_Z']:.2f})")
            st.pyplot(fig)

        with tab1:
            sig = res[abs(res['Current_Z']) >= z_thresh]
            if not sig.empty:
                st.dataframe(sig[['Sector', 'Stock1', 'Stock2', 'Correlation', 'Current_Z', 'P_value']], use_container_width=True, hide_index=True)
                sel = st.selectbox("상세 차트 확인 (Signal)", sig.index, format_func=lambda x: f"{sig.loc[x, 'Stock1']} - {sig.loc[x, 'Stock2']}")
                draw_chart(sig.loc[sel])
            else: st.info("현재 진입 기준을 충족하는 페어가 없습니다.")

        with tab2:
            st.dataframe(res[['Sector', 'Stock1', 'Stock2', 'Correlation', 'Current_Z', 'P_value']], use_container_width=True, hide_index=True)
            sel_w = st.selectbox("상세 차트 확인 (Watchlist)", res.index, format_func=lambda x: f"{res.loc[x, 'Stock1']} - {res.loc[x, 'Stock2']}")
            draw_chart(res.loc[sel_w])
    elif 'results' in st.session_state:
        st.warning("분석 결과 유효한 페어가 없습니다. 조건을 완화해보세요.")
else:
    st.info("2단계에서 데이터를 먼저 다운로드해주세요.")
