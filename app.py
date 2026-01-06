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
# 🎨 0. 블룸버그 스타일 & 폰트 설정
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
# 📡 1. 데이터 수집 (전체 종목 & 시가총액 매핑)
# ==========================================
@st.cache_data(ttl=3600*12)
def fetch_all_naver_stocks():
    """
    네이버의 '전체' 업종과 '전체' 종목 코드를 수집합니다.
    """
    base_url = "https://finance.naver.com/sise/sise_group.naver?type=upjong"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        # 1. 업종 리스트 가져오기
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
        
        # 2. 각 업종별 '모든' 종목 긁기
        all_data = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 주의: 전체 업종(약 80개)을 다 돌면 시간이 좀 걸립니다.
        total_sectors = len(sector_links)
        
        for idx, (sec_name, sec_url) in enumerate(sector_links):
            status_text.text(f"📡 전체 데이터 수집 중... [{idx+1}/{total_sectors}] {sec_name}")
            progress_bar.progress((idx + 1) / total_sectors)
            
            res_sec = requests.get(sec_url, headers=headers)
            soup_sec = BeautifulSoup(res_sec.text, 'html.parser')
            sub_table = soup_sec.find('table', {'class': 'type_5'})
            if not sub_table: continue
            
            # 해당 섹터의 모든 종목 루프
            for s_row in sub_table.find_all('tr'):
                s_cols = s_row.find_all('td')
                if len(s_cols) < 2: continue 
                
                name_tag = s_cols[0].find('a')
                if name_tag:
                    stock_name = name_tag.text.strip()
                    stock_code = name_tag['href'].split('code=')[-1]
                    cur_price = s_cols[1].text.strip()
                    
                    all_data.append({
                        'Sector': sec_name,
                        'Name': stock_name,
                        'Code': stock_code,
                        'Price': cur_price
                    })
            # 차단 방지 딜레이
            time.sleep(0.02)
            
        progress_bar.empty()
        status_text.empty()
        
        # 3. 데이터프레임 변환
        df_naver = pd.DataFrame(all_data).drop_duplicates(subset=['Code'])
        
        # 4. 시가총액(Marcap) 정보 매핑 (FDR 사용)
        status_text.text("💰 시가총액 데이터 매핑 및 정렬 중...")
        
        # KRX 전체 리스팅 (시가총액 포함)
        df_krx = fdr.StockListing('KRX')[['Code', 'Marcap']]
        
        # 네이버 데이터 + KRX 시총 데이터 병합
        df_merged = pd.merge(df_naver, df_krx, on='Code', how='left')
        
        # 시가총액 없는 종목(ETF 등)은 0 처리
        df_merged['Marcap'] = df_merged['Marcap'].fillna(0)
        
        # **중요**: 전체 데이터를 시가총액 순서로 미리 정렬해둠
        df_merged = df_merged.sort_values(by=['Sector', 'Marcap'], ascending=[True, False])
        
        # 보기 좋은 포맷팅 컬럼 추가
        def format_marcap(val):
            if val == 0: return "-"
            val = int(val)
            jo = val // 1000000000000
            uk = (val % 1000000000000) // 100000000
            if jo > 0: return f"{jo}조 {uk}억"
            return f"{uk}억"
            
        df_merged['Market Cap'] = df_merged['Marcap'].apply(format_marcap)
        
        status_text.empty()
        return df_merged

    except Exception as e:
        st.error(f"데이터 수집 에러: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_price_history(codes_list, start_date):
    """
    선택된 종목 리스트의 주가 다운로드
    """
    data_dict = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(codes_list)
    for i, code in enumerate(codes_list):
        if i % 5 == 0: # UI 갱신 빈도 조절
            status_text.text(f"📉 주가 데이터 다운로드: {i+1}/{total}")
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
    # 이미 섹터별로 필터링되어 들어오지만 확인
    sectors = stocks_info['Sector'].unique()
    
    for sector in sectors:
        sector_stocks = stocks_info[stocks_info['Sector'] == sector]
        valid_codes = [c for c in sector_stocks['Code'] if c in price_df.columns]
        
        if len(valid_codes) < 2: continue
        
        # 전체 종목(수십개) 간의 조합 (Pairs)
        # 종목이 많으면 연산량이 급증하므로(50개면 1225개 조합), 진행상황 표시가 필요할 수 있음
        stock_combinations = list(combinations(valid_codes, 2))
        
        for s1, s2 in stock_combinations:
            series1 = price_df[s1]
            series2 = price_df[s2]
            
            # 데이터 품질 체크
            if len(series1) < 30 or series1.std() == 0 or series2.std() == 0: continue
            
            # [속도 최적화] 상관계수 먼저 체크 (빠름)
            if series1.corr(series2) < 0.8: continue

            try:
                # 공적분 테스트 (느림)
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
# 🖥️ UI: 블룸버그 스타일 대시보드
# ==========================================
st.set_page_config(page_title="Pair Terminal", layout="wide", page_icon="📊")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
</style>
""", unsafe_allow_html=True)

st.title("📊 Pair Scanner")

if 'all_market_data' not in st.session_state:
    st.session_state.all_market_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# ==========================================
# [STEP 1] 전체 시장 조회 (Data Fetching)
# ==========================================
st.header("1️⃣ Market Data Fetch (Naver x FDR)")
col_btn, col_msg = st.columns([1, 4])

with col_btn:
    if st.button("🔄 전체 데이터 수집 (Click)", type="primary"):
        df = fetch_all_naver_stocks()
        st.session_state.all_market_data = df
        st.session_state.analysis_results = None # 데이터 바뀌면 결과 초기화

with col_msg:
    if st.session_state.all_market_data is not None:
        raw_df = st.session_state.all_market_data
        st.success(f"✅ 수집 완료: 총 {raw_df['Sector'].nunique()}개 섹터, {len(raw_df)}개 종목 (All Pairs Ready)")
    else:
        st.info("버튼을 눌러 전체 시장 데이터를 가져오세요. (약 30초 소요)")

# [Display] 요약 리스트 (Top 5 Display ONLY)
if st.session_state.all_market_data is not None:
    st.markdown("##### 📂 섹터별 대장주 요약 (Top 3 by Market Cap)")
    
    # 데이터는 전체를 가지고 있지만, 보여주는 건 섹터별 Top 3만
    display_df = st.session_state.all_market_data.groupby('Sector').head(3)
    
    with st.expander("리스트 펼쳐보기", expanded=True):
        st.dataframe(
            display_df[['Sector', 'Name', 'Price', 'Market Cap']], 
            use_container_width=True,
            column_config={
                "Sector": "업종명",
                "Name": "종목명",
                "Price": "현재가",
                "Market Cap": "시가총액"
            },
            hide_index=True
        )

# ==========================================
# [STEP 2] 심층 분석 (Deep Dive)
# ==========================================
st.divider()
st.header("2️⃣ Sector Selection & Pair Analysis")

if st.session_state.all_market_data is not None:
    raw_df = st.session_state.all_market_data
    all_sectors = raw_df['Sector'].unique().tolist()
    
    # 2-1. 섹터 선택
    selected_sectors = st.multiselect(
        "분석할 섹터를 선택하세요 (다중 선택 가능):", 
        all_sectors,
        default=all_sectors[:1] if len(all_sectors) > 0 else None
    )
    
    c1, c2, c3 = st.columns(3)
    lookback = c1.slider("조회 기간 (Lookback)", 100, 730, 365)
    z_thresh = c2.number_input("Z-Score Threshold (진입)", 1.5, 4.0, 2.0, 0.1)
    p_thresh = c3.number_input("P-value (유의수준)", 0.01, 0.1, 0.05, 0.01)
    
    # 2-2. 분석 실행 버튼
    if st.button("🚀 선택 섹터 전체 종목 분석 (Full Analysis)", type="primary"):
        if not selected_sectors:
            st.warning("섹터를 먼저 선택해주세요.")
        else:
            # [핵심] 선택된 섹터의 '전체' 종목을 가져옴 (Top 5 아님!)
            target_stocks_info = raw_df[raw_df['Sector'].isin(selected_sectors)]
            
            st.info(f"🧐 선택된 섹터의 전체 종목 {len(target_stocks_info)}개를 분석합니다...")
            
            start_date = (datetime.now() - timedelta(days=lookback)).strftime('%Y-%m-%d')
            
            # 주가 데이터는 분석할 종목들만 다운로드
            codes_to_fetch = target_stocks_info['Code'].tolist()
            price_df = fetch_price_history(codes_to_fetch, start_date)
            
            if price_df.empty:
                st.error("주가 데이터를 가져오지 못했습니다.")
            else:
                with st.spinner("퀀트 엔진 가동 중... 모든 가능한 조합(Pair)을 계산합니다."):
                    results = run_pair_analysis(price_df, target_stocks_info, p_thresh, z_thresh)
                    st.session_state.analysis_results = (results, price_df)

# ==========================================
# [STEP 3] 결과 시각화 (Bloomberg Style)
# ==========================================
if st.session_state.analysis_results is not None:
    results, price_df = st.session_state.analysis_results
    
    if not results.empty:
        signals = results[abs(results['Current_Z']) >= z_thresh].copy()
        signals['Signal'] = np.where(signals['Current_Z'] > 0, "SHORT A / LONG B", "LONG A / SHORT B")
        
        st.divider()
        st.subheader(f"📊 Analysis Result: {len(results)} Pairs Found")
        
        tab1, tab2 = st.tabs(["🔥 TRADING SIGNALS", "👀 WATCHLIST"])
        
        def draw_bloomberg_chart(pair_data, price_df, z_limit):
            s1, s2 = pair_data['Code1'], pair_data['Code2']
            n1, n2 = pair_data['Stock1'], pair_data['Stock2']
            spread = pair_data['Spread_Series']
            
            p1 = price_df[s1] / price_df[s1].iloc[0] * 100
            p2 = price_df[s2] / price_df[s2].iloc[0] * 100
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
            
            # Neon Colors
            ax1.plot(p1, color='#00ffcc', label=n1, linewidth=2) 
            ax1.plot(p2, color='#ff00ff', label=n2, linewidth=2)
            ax1.set_title(f"PRICE ACTION: {n1} vs {n2}", color='#ff9900', fontsize=16, pad=15)
            ax1.legend(facecolor='#1e1e1e', edgecolor='#444444')
            
            z_score = (spread - spread.mean()) / spread.std()
            ax2.plot(z_score, color='#ffff00', label='Z-Score', linewidth=1.5)
            ax2.axhline(z_limit, color='red', linestyle='--', linewidth=1)
            ax2.axhline(-z_limit, color='red', linestyle='--', linewidth=1)
            ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)
            
            ax2.fill_between(z_score.index, z_limit, z_score, where=(z_score >= z_limit), color='red', alpha=0.3)
            ax2.fill_between(z_score.index, -z_limit, z_score, where=(z_score <= -z_limit), color='red', alpha=0.3)
            ax2.set_title(f"SPREAD Z-SCORE (Current: {pair_data['Current_Z']:.2f})", color='#ff9900', fontsize=12)
            
            plt.tight_layout()
            st.pyplot(fig)

        with tab1:
            if signals.empty:
                st.info("진입 시그널이 없습니다.")
            else:
                st.dataframe(signals[['Sector', 'Stock1', 'Stock2', 'Current_Z', 'Signal', 'P_value']], use_container_width=True, hide_index=True)
                sel_sig = st.selectbox("차트 확인:", signals.index, format_func=lambda i: f"{signals.loc[i,'Stock1']} vs {signals.loc[i,'Stock2']}", key='sig_sel')
                draw_bloomberg_chart(signals.loc[sel_sig], price_df, z_thresh)

        with tab2:
            watchlist = results[abs(results['Current_Z']) < z_thresh].sort_values('P_value')
            if watchlist.empty:
                st.info("Watchlist가 비어있습니다.")
            else:
                st.dataframe(watchlist[['Sector', 'Stock1', 'Stock2', 'Current_Z', 'P_value']], use_container_width=True)
                sel_watch = st.selectbox("차트 확인:", watchlist.index, format_func=lambda i: f"{watchlist.loc[i,'Stock1']} vs {watchlist.loc[i,'Stock2']}", key='watch_sel')
                draw_bloomberg_chart(watchlist.loc[sel_watch], price_df, z_thresh)
    else:
        st.warning("분석 결과, 유의미한 페어를 찾지 못했습니다.")
