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
    # 1. 한글 폰트 설정
    font_path = "NanumGothic.ttf"
    if not os.path.exists(font_path):
        url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
        response = requests.get(url)
        with open(font_path, "wb") as f:
            f.write(response.content)
    fm.fontManager.addfont(font_path)
    plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False 

    # 2. 블룸버그 스타일(Dark Theme) 적용
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
        'axes.labelcolor': '#ff9900', # 블룸버그 오렌지
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'lines.linewidth': 1.5
    })

init_settings()

# ==========================================
# 📡 1. 데이터 크롤링 (지표 포함)
# ==========================================
@st.cache_data(ttl=3600*12)
def fetch_sector_overview():
    """
    네이버 증권에서 섹터 목록과, 각 섹터별 Top 종목들의 주요 지표(현재가, 등락률 등)를 긁어옵니다.
    """
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
        
        # 전체 섹터를 다 긁으면 오래 걸리므로, 여기서는 상위 30개 섹터만 빠르게 조회하도록 제한 (조절 가능)
        # 사용자가 원하면 전체 루프를 돌려도 됩니다.
        target_sectors = sector_links[:30] 
        
        for idx, (sec_name, sec_url) in enumerate(target_sectors):
            status_text.text(f"📡 [Step 1] 섹터 데이터 수집 중... {sec_name} ({idx+1}/{len(target_sectors)})")
            progress_bar.progress((idx + 1) / len(target_sectors))
            
            res_sec = requests.get(sec_url, headers=headers)
            soup_sec = BeautifulSoup(res_sec.text, 'html.parser')
            sub_table = soup_sec.find('table', {'class': 'type_5'})
            if not sub_table: continue
            
            # 섹터 내 상위 5개 종목만 추출
            count = 0
            for s_row in sub_table.find_all('tr'):
                s_cols = s_row.find_all('td')
                if len(s_cols) < 5: continue # 데이터 없는 줄 패스
                
                name_tag = s_cols[0].find('a')
                if name_tag:
                    stock_name = name_tag.text.strip()
                    stock_code = name_tag['href'].split('code=')[-1]
                    
                    # 주요 지표 추출 (현재가, 등락률, 시가총액 등)
                    # 네이버 페이지 구조: 0:명, 1:현재가, 2:전일비, 3:등락률 ...
                    cur_price = s_cols[1].text.strip()
                    change_rate = s_cols[3].text.strip()
                    
                    all_data.append({
                        'Sector': sec_name,
                        'Name': stock_name,
                        'Code': stock_code,
                        'Price': cur_price,
                        'Change(%)': change_rate
                    })
                    count += 1
                if count >= 5: break # Top 5만
            time.sleep(0.05)
            
        progress_bar.empty()
        status_text.empty()
        
        return pd.DataFrame(all_data).drop_duplicates(subset=['Code'])
    except Exception as e:
        st.error(f"크롤링 에러: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_price_history(target_stocks_df, start_date):
    """
    선택된 종목들의 주가 데이터 다운로드
    """
    codes = target_stocks_df['Code'].unique().tolist()
    data_dict = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, code in enumerate(codes):
        status_text.text(f"📉 [Step 2] 주가 데이터 다운로드: {i+1}/{len(codes)}")
        progress_bar.progress((i + 1) / len(codes))
        try:
            df = fdr.DataReader(code, start_date)
            if not df.empty:
                data_dict[code] = df['Close']
        except: continue
    
    progress_bar.empty()
    status_text.empty()
    
    price_df = pd.DataFrame(data_dict).dropna()
    return price_df

def run_pair_analysis(price_df, stocks_info, p_thresh, z_thresh):
    """
    통계적 차익거래 분석 실행 (예외처리 포함)
    """
    pairs = []
    sectors = stocks_info['Sector'].unique()
    
    for sector in sectors:
        # 해당 섹터에 속하고 + 주가 데이터가 있는 종목만 필터링
        sector_stocks = stocks_info[stocks_info['Sector'] == sector]
        valid_codes = [c for c in sector_stocks['Code'] if c in price_df.columns]
        
        if len(valid_codes) < 2: continue
        
        for s1, s2 in combinations(valid_codes, 2):
            series1 = price_df[s1]
            series2 = price_df[s2]
            
            # 예외처리: 데이터 부족 or 거래정지
            if len(series1) < 30 or series1.std() == 0 or series2.std() == 0: continue
            
            # 상관계수 0.8 미만 칼차단
            if series1.corr(series2) < 0.8: continue

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
                        'Hedge_Ratio': hedge_ratio,
                        'Spread_Series': spread
                    })
            except: continue
            
    return pd.DataFrame(pairs)

# ==========================================
# 🖥️ UI: 블룸버그 스타일 대시보드
# ==========================================
st.set_page_config(page_title="Bloomberg Quant Terminal", layout="wide", page_icon="📊")

# CSS 커스텀 (다크모드 강제 적용 및 테이블 스타일링)
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .metric-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Bloomberg Quant Pair Scanner")
st.markdown("Top-Down 접근법: **섹터 현황 파악** ➔ **타겟 섹터 선정** ➔ **Pair 발굴**")

# --- Session State 관리 ---
if 'raw_market_data' not in st.session_state:
    st.session_state.raw_market_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# ==========================================
# [STEP 1] 전체 시장 조회
# ==========================================
st.header("1️⃣ Market Overview")
col_btn, col_info = st.columns([1, 4])

with col_btn:
    if st.button("🔄 전체 섹터 및 종목 조회 (Naver)", type="primary"):
        df = fetch_sector_overview()
        st.session_state.raw_market_data = df
        st.session_state.analysis_results = None # 데이터 바뀌면 결과 초기화

with col_info:
    if st.session_state.raw_market_data is not None:
        raw_df = st.session_state.raw_market_data
        n_sectors = raw_df['Sector'].nunique()
        n_stocks = len(raw_df)
        st.success(f"✅ 데이터 수신 완료: {n_sectors}개 섹터, {n_stocks}개 종목 (Top 5 per Sector)")
    else:
        st.info("좌측 버튼을 눌러 최신 시장 데이터를 가져오세요.")

# 데이터가 있을 때만 섹터별 Top 5 미리보기 보여줌
if st.session_state.raw_market_data is not None:
    with st.expander("📂 섹터별 Top 5 종목 및 주요 지표 확인하기 (Click to Expand)", expanded=True):
        st.dataframe(
            st.session_state.raw_market_data, 
            use_container_width=True,
            column_config={
                "Sector": "업종명",
                "Name": "종목명",
                "Price": "현재가",
                "Change(%)": "등락률"
            }
        )

# ==========================================
# [STEP 2] 타겟 섹터 선정 및 분석
# ==========================================
st.divider()
st.header("2️⃣ Deep Dive Analysis")

if st.session_state.raw_market_data is not None:
    raw_df = st.session_state.raw_market_data
    all_sectors = raw_df['Sector'].unique().tolist()
    
    # 2-1. 섹터 선택 (Multi-select)
    selected_sectors = st.multiselect(
        "분석하고 싶은 섹터를 선택하세요 (다중 선택 가능):", 
        all_sectors,
        default=all_sectors[:3] if len(all_sectors) > 3 else all_sectors
    )
    
    # 설정 옵션 (사이드바 대신 메인 상단으로 이동하여 접근성 강화)
    c1, c2, c3 = st.columns(3)
    lookback = c1.slider("조회 기간 (Lookback)", 100, 730, 365)
    z_thresh = c2.number_input("Z-Score Threshold (진입)", 1.5, 4.0, 2.0, 0.1)
    p_thresh = c3.number_input("P-value (유의수준)", 0.01, 0.1, 0.05, 0.01)
    
    if st.button("🚀 선택한 섹터 심층 분석 시작", type="primary"):
        if not selected_sectors:
            st.warning("섹터를 하나 이상 선택해주세요.")
        else:
            # 선택된 섹터의 종목만 추림
            target_stocks_info = raw_df[raw_df['Sector'].isin(selected_sectors)]
            
            start_date = (datetime.now() - timedelta(days=lookback)).strftime('%Y-%m-%d')
            
            # 주가 다운로드
            price_df = fetch_price_history(target_stocks_info, start_date)
            
            if price_df.empty:
                st.error("주가 데이터를 가져오지 못했습니다.")
            else:
                # 분석 실행
                with st.spinner("퀀트 엔진 가동 중... (Cointegration Test & Z-Score Calc)"):
                    results = run_pair_analysis(price_df, target_stocks_info, p_thresh, z_thresh)
                    st.session_state.analysis_results = (results, price_df) # 결과 및 가격데이터 저장

# ==========================================
# [STEP 3] 결과 시각화 (Bloomberg Style)
# ==========================================
if st.session_state.analysis_results is not None:
    results, price_df = st.session_state.analysis_results
    
    if not results.empty:
        # 시그널 분리
        signals = results[abs(results['Current_Z']) >= z_thresh].copy()
        signals['Signal'] = np.where(signals['Current_Z'] > 0, "SHORT A / LONG B", "LONG A / SHORT B")
        
        st.divider()
        st.subheader(f"📊 Analysis Result: {len(results)} Pairs Found")
        
        tab1, tab2 = st.tabs(["🔥 TRADING SIGNALS", "👀 WATCHLIST"])
        
        # --- [TAB 1] Signals ---
        with tab1:
            if signals.empty:
                st.info("현재 진입 기준(Threshold)을 만족하는 시그널이 없습니다.")
            else:
                # 테이블 표시
                st.dataframe(
                    signals[['Sector', 'Stock1', 'Stock2', 'Current_Z', 'Signal', 'P_value']], 
                    use_container_width=True,
                    hide_index=True
                )
                
                # 차트 선택
                st.markdown("### 📈 Interactive Chart")
                sel_sig = st.selectbox(
                    "차트를 확인할 페어를 선택하세요:", 
                    signals.index, 
                    format_func=lambda i: f"[{signals.loc[i,'Sector']}] {signals.loc[i,'Stock1']} vs {signals.loc[i,'Stock2']} (Z: {signals.loc[i,'Current_Z']:.2f})"
                )
                
                # 블룸버그 스타일 차트 그리기 함수
                def draw_bloomberg_chart(pair_data, price_df, z_limit):
                    s1, s2 = pair_data['Code1'], pair_data['Code2']
                    n1, n2 = pair_data['Stock1'], pair_data['Stock2']
                    spread = pair_data['Spread_Series']
                    
                    # 정규화
                    p1 = price_df[s1] / price_df[s1].iloc[0] * 100
                    p2 = price_df[s2] / price_df[s2].iloc[0] * 100
                    
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
                    
                    # 상단: 주가 흐름
                    ax1.plot(p1, color='#00ffcc', label=n1, linewidth=2) # 네온 민트
                    ax1.plot(p2, color='#ff00ff', label=n2, linewidth=2) # 네온 마젠타
                    ax1.set_title(f"PRICE ACTION: {n1} vs {n2}", color='#ff9900', fontsize=16, pad=15)
                    ax1.legend(facecolor='#1e1e1e', edgecolor='#444444')
                    
                    # 하단: Spread Z-Score
                    z_score = (spread - spread.mean()) / spread.std()
                    ax2.plot(z_score, color='#ffff00', label='Z-Score', linewidth=1.5) # 네온 옐로우
                    ax2.axhline(z_limit, color='red', linestyle='--', linewidth=1)
                    ax2.axhline(-z_limit, color='red', linestyle='--', linewidth=1)
                    ax2.axhline(0, color='gray', linestyle='-', alpha=0.5)
                    
                    # 영역 채우기 (진입 구간)
                    ax2.fill_between(z_score.index, z_limit, z_score, where=(z_score >= z_limit), color='red', alpha=0.3)
                    ax2.fill_between(z_score.index, -z_limit, z_score, where=(z_score <= -z_limit), color='red', alpha=0.3)
                    
                    ax2.set_title(f"SPREAD Z-SCORE (Current: {pair_data['Current_Z']:.2f})", color='#ff9900', fontsize=12)
                    
                    plt.tight_layout()
                    st.pyplot(fig)

                draw_bloomberg_chart(signals.loc[sel_sig], price_df, z_thresh)

        # --- [TAB 2] Watchlist ---
        with tab2:
            watchlist = results[abs(results['Current_Z']) < z_thresh].sort_values('P_value')
            if watchlist.empty:
                st.info("Watchlist가 비어있습니다.")
            else:
                st.dataframe(watchlist[['Sector', 'Stock1', 'Stock2', 'Current_Z', 'P_value']], use_container_width=True)
                
                st.markdown("### 📈 Interactive Chart (Watchlist)")
                sel_watch = st.selectbox(
                    "대기 종목 차트 확인:", 
                    watchlist.index, 
                    format_func=lambda i: f"[{watchlist.loc[i,'Sector']}] {watchlist.loc[i,'Stock1']} vs {watchlist.loc[i,'Stock2']}",
                    key='watch_sel'
                )
                draw_bloomberg_chart(watchlist.loc[sel_watch], price_df, z_thresh)
    else:
        st.warning("조건에 맞는 페어를 찾지 못했습니다. P-value를 높이거나 섹터를 더 추가해보세요.")
