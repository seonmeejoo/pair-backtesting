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
# 🛠️ [필수] 한글 폰트 자동 설정
# ==========================================
def init_font():
    font_path = "NanumGothic.ttf"
    if not os.path.exists(font_path):
        url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
        response = requests.get(url)
        with open(font_path, "wb") as f:
            f.write(response.content)
            
    fm.fontManager.addfont(font_path)
    plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False 

init_font()

# ==========================================
# 📡 데이터 크롤링 및 분석 함수
# ==========================================
@st.cache_data(ttl=3600*12)
def get_naver_sectors(limit_sectors=None):
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
        
        if limit_sectors:
            sector_links = sector_links[:limit_sectors]

        all_stocks = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (sec_name, sec_url) in enumerate(sector_links):
            status_text.text(f"⏳ 크롤링 중: {sec_name} ({idx+1}/{len(sector_links)})")
            progress_bar.progress((idx + 1) / len(sector_links))
            
            res_sec = requests.get(sec_url, headers=headers)
            soup_sec = BeautifulSoup(res_sec.text, 'html.parser')
            sub_table = soup_sec.find('table', {'class': 'type_5'})
            if not sub_table: continue
            
            for s_row in sub_table.find_all('tr'):
                s_cols = s_row.find_all('td')
                if len(s_cols) < 2: continue
                name_tag = s_cols[0].find('a')
                if name_tag:
                    all_stocks.append({
                        'Sector': sec_name,
                        'Name': name_tag.text.strip(),
                        'Code': name_tag['href'].split('code=')[-1]
                    })
            time.sleep(0.05)
            
        progress_bar.empty()
        status_text.empty()
        
        return pd.DataFrame(all_stocks).drop_duplicates(subset=['Code'])
    except Exception as e:
        st.error(f"크롤링 에러: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_prices(stock_list, start_date, top_n=5):
    target_df = stock_list.groupby('Sector').head(top_n)
    codes = target_df['Code'].tolist()
    data_dict = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, code in enumerate(codes):
        status_text.text(f"📉 주가 다운로드: {i+1}/{len(codes)}")
        progress_bar.progress((i + 1) / len(codes))
        try:
            df = fdr.DataReader(code, start_date)
            if not df.empty:
                data_dict[code] = df['Close']
        except: continue
        
    progress_bar.empty()
    status_text.empty()
    
    price_df = pd.DataFrame(data_dict).dropna()
    valid_stocks = target_df[target_df['Code'].isin(price_df.columns)]
    
    return price_df, valid_stocks

def analyze_pairs(price_df, valid_stocks, p_val_thresh, z_score_thresh):
    pairs = []
    sectors = valid_stocks['Sector'].unique()
    
    for sector in sectors:
        sector_codes = valid_stocks[valid_stocks['Sector'] == sector]['Code'].tolist()
        if len(sector_codes) < 2: continue
        
        for s1, s2 in combinations(sector_codes, 2):
            series1 = price_df[s1]
            series2 = price_df[s2]
            
            if series1.corr(series2) < 0.8: continue

            score, p_value, _ = coint(series1, series2)
            if p_value < p_val_thresh:
                name1 = valid_stocks[valid_stocks['Code'] == s1]['Name'].values[0]
                name2 = valid_stocks[valid_stocks['Code'] == s2]['Name'].values[0]
                
                x = sm.add_constant(series2)
                model = sm.OLS(series1, x).fit()
                spread = series1 - (model.params[1] * series2)
                z_score = (spread.iloc[-1] - spread.mean()) / spread.std()
                
                pairs.append({
                    'Sector': sector, 'Stock1': name1, 'Stock2': name2,
                    'P_value': p_value, 'Current_Z': z_score,
                    'Code1': s1, 'Code2': s2, 'Spread_Series': spread
                })
    return pd.DataFrame(pairs)

# ==========================================
# 🖥️ Streamlit UI 디자인
# ==========================================
st.set_page_config(page_title="Pairs Trading Scanner", layout="wide", page_icon="📈")

st.title("📈 Sector-based Pair Trading Scanner")
st.markdown("""
네이버 금융의 업종 데이터를 분석하여, **높은 상관관계(Correlation)**를 가진 종목 중 
**일시적 가격 괴리(Spread)**가 발생한 **유망 Pair**를 실시간으로 발굴합니다.
""")

# --- 세션 상태 초기화 (데이터 저장소) ---
if 'price_df' not in st.session_state:
    st.session_state.price_df = None
if 'valid_stocks' not in st.session_state:
    st.session_state.valid_stocks = None
if 'data_downloaded' not in st.session_state:
    st.session_state.data_downloaded = False

with st.sidebar:
    st.header("1️⃣ 데이터 준비 (Data Fetch)")
    st.caption("먼저 데이터를 다운로드 받으세요. (시간 소요)")
    limit_sectors = st.slider("업종 개수 (속도 조절)", 5, 50, 10)
    lookback = st.slider("조회 기간 (일)", 100, 730, 365)
    
    # [Step 1] 데이터 다운로드 버튼
    if st.button("📥 데이터 가져오기", type="primary"):
        stocks_df = get_naver_sectors(limit_sectors)
        st.success(f"{len(stocks_df)}개 종목 목록 확보")
        
        start_date = (datetime.now() - timedelta(days=lookback)).strftime('%Y-%m-%d')
        p_df, v_stocks = fetch_prices(stocks_df, start_date)
        
        # 세션에 저장!
        st.session_state.price_df = p_df
        st.session_state.valid_stocks = v_stocks
        st.session_state.data_downloaded = True
        st.rerun() # 화면 새로고침

    st.divider()
    
    st.header("2️⃣ 전략 분석 (Analysis)")
    st.caption("다운로드된 데이터로 조건을 변경하며 분석하세요.")
    z_thresh = st.number_input("Z-Score 기준", 1.5, 4.0, 2.0, 0.1)
    p_thresh = st.number_input("P-value 기준", 0.01, 0.1, 0.05, 0.01)

# --- 메인 로직 ---

if st.session_state.data_downloaded:
    # 데이터가 준비되었을 때만 실행
    st.success(f"✅ 데이터 준비 완료! (보유 종목 수: {len(st.session_state.price_df.columns)}개)")
    
    # [Step 2] 분석은 버튼 없이 조건만 바꾸면 자동 실행 (또는 버튼 추가 가능)
    # 여기서는 자동으로 매번 빠르게 계산
    with st.spinner("⚡ 조건에 맞춰 분석 중..."):
        results = analyze_pairs(st.session_state.price_df, st.session_state.valid_stocks, p_thresh, z_thresh)
    
    if not results.empty:
        signals = results[abs(results['Current_Z']) >= z_thresh].copy()
        signals['Action'] = np.where(signals['Current_Z'] > 0, 
                                     "Short A / Long B", 
                                     "Long A / Short B")
        
        watchlist = results[abs(results['Current_Z']) < z_thresh].sort_values('P_value')
        
        col1, col2 = st.columns(2)
        col1.metric("발견된 총 페어", f"{len(results)}개")
        col2.metric("🚀 진입 추천 시그널", f"{len(signals)}개", delta_color="inverse")
        
        tab1, tab2 = st.tabs(["🔥 진입 시그널 (Action)", "👀 관심 종목 (Watchlist)"])
        
        # --- Tab 1 ---
        with tab1:
            if not signals.empty:
                st.dataframe(signals[['Sector', 'Stock1', 'Stock2', 'Current_Z', 'Action', 'P_value']], use_container_width=True)
                
                st.subheader("📊 상세 차트 분석")
                sel_idx = st.selectbox("차트를 볼 페어를 선택하세요", signals.index, 
                                       format_func=lambda x: f"{signals.loc[x, 'Stock1']} vs {signals.loc[x, 'Stock2']}",
                                       key="sig_select")
                
                pair = signals.loc[sel_idx]
                s1, s2 = pair['Code1'], pair['Code2']
                spread = pair['Spread_Series']
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                
                p1 = st.session_state.price_df[s1] / st.session_state.price_df[s1].iloc[0] * 100
                p2 = st.session_state.price_df[s2] / st.session_state.price_df[s2].iloc[0] * 100
                
                ax1.plot(p1, label=pair['Stock1'], color='blue')
                ax1.plot(p2, label=pair['Stock2'], color='orange')
                ax1.set_title(f"Price Trend: {pair['Stock1']} vs {pair['Stock2']} ({pair['Sector']})")
                ax1.legend(); ax1.grid(True, alpha=0.3)
                
                z_score_series = (spread - spread.mean()) / spread.std()
                ax2.plot(z_score_series, color='green', label='Spread Z-Score')
                ax2.axhline(z_thresh, c='r', ls='--'); ax2.axhline(-z_thresh, c='r', ls='--'); ax2.axhline(0, c='k', alpha=0.5)
                ax2.fill_between(z_score_series.index, z_thresh, z_score_series, where=(z_score_series >= z_thresh), color='red', alpha=0.3)
                ax2.fill_between(z_score_series.index, -z_thresh, z_score_series, where=(z_score_series <= -z_thresh), color='red', alpha=0.3)
                ax2.set_title(f"Spread Z-Score (Current: {pair['Current_Z']})")
                ax2.legend(); ax2.grid(True, alpha=0.3)
                st.pyplot(fig)
            else:
                st.info("설정된 Z-Score 기준을 만족하는 종목이 없습니다. 좌측 사이드바에서 기준을 낮춰보세요.")

        # --- Tab 2 ---
        with tab2:
            st.dataframe(watchlist[['Sector', 'Stock1', 'Stock2', 'Current_Z', 'P_value']], use_container_width=True)
            
            if not watchlist.empty:
                st.divider()
                st.subheader("📊 상세 차트 분석 (Watchlist)")
                w_idx = st.selectbox("차트를 볼 페어를 선택하세요", watchlist.index, 
                                     format_func=lambda x: f"{watchlist.loc[x, 'Stock1']} vs {watchlist.loc[x, 'Stock2']}",
                                     key="watch_select")
                
                w_pair = watchlist.loc[w_idx]
                ws1, ws2 = w_pair['Code1'], w_pair['Code2']
                w_spread = w_pair['Spread_Series']
                
                fig_w, (wax1, wax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                
                wp1 = st.session_state.price_df[ws1] / st.session_state.price_df[ws1].iloc[0] * 100
                wp2 = st.session_state.price_df[ws2] / st.session_state.price_df[ws2].iloc[0] * 100
                
                wax1.plot(wp1, label=w_pair['Stock1'], color='blue')
                wax1.plot(wp2, label=w_pair['Stock2'], color='orange')
                wax1.set_title(f"Price Trend: {w_pair['Stock1']} vs {w_pair['Stock2']} ({w_pair['Sector']})")
                wax1.legend(); wax1.grid(True, alpha=0.3)
                
                w_z_series = (w_spread - w_spread.mean()) / w_spread.std()
                wax2.plot(w_z_series, color='green', label='Spread Z-Score')
                wax2.axhline(z_thresh, c='r', ls='--'); wax2.axhline(-z_thresh, c='r', ls='--'); wax2.axhline(0, c='k', alpha=0.5)
                wax2.set_title(f"Spread Z-Score (Current: {w_pair['Current_Z']})")
                wax2.legend(); wax2.grid(True, alpha=0.3)
                st.pyplot(fig_w)
            else:
                st.info("조건에 맞는 Watchlist가 없습니다.")
    else:
        st.warning("조건에 맞는 페어를 찾지 못했습니다.")

else:
    # 데이터가 아직 없을 때 보여줄 화면
    st.info("👈 왼쪽 사이드바에서 **'데이터 가져오기'** 버튼을 먼저 눌러주세요.")
