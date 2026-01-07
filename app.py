import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
from datetime import datetime, timedelta

# --- 페이지 설정 ---
st.set_page_config(layout="wide", page_title="Pair Trading Analyst")

# --- 스타일 정의 (블룸버그 다크 테마) ---
BLOOMBERG_THEME = {
    'bgcolor': '#1e1e1e',
    'paper_bgcolor': '#121212',
    'font_color': '#e0e0e0',
    'grid_color': '#444444',
    'line_colors': ['#ff9f1c', '#2ec4b6']  # 오렌지, 청록
}

# --- 함수 정의 ---

@st.cache_data
def get_stock_list():
    """KRX 전체 종목 조회 (데이터 타입 오류 방지 강화)"""
    try:
        df = fdr.StockListing('KRX')
    except Exception:
        # KRX 조회 실패 시 KOSPI/KOSDAQ 각각 조회 후 병합
        try:
            df_kospi = fdr.StockListing('KOSPI')
            df_kosdaq = fdr.StockListing('KOSDAQ')
            df = pd.concat([df_kospi, df_kosdaq])
        except Exception:
             # 최악의 경우 빈 데이터프레임 반환 (앱이 죽는 것 방지)
            return pd.DataFrame(columns=['Code', 'Name', 'Sector', 'Marcap', 'Close', 'ChgesRatio'])
    
    # 1. 컬럼명 통일 (Symbol -> Code)
    if 'Symbol' in df.columns:
        df = df.rename(columns={'Symbol': 'Code'})
        
    # 2. 필수 컬럼 존재 여부 확인 및 생성
    required_cols = ['Code', 'Name', 'Sector', 'Marcap', 'Close', 'ChgesRatio']
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan

    # 3. 데이터 타입 강제 변환 (이 부분이 핵심 해결책)
    # Sector 컬럼의 결측치(NaN)를 'Unknown'으로 채우고, 강제로 문자열(str)로 변환합니다.
    df['Sector'] = df['Sector'].fillna('Unknown').astype(str)
    
    # 4. 필터링
    # 이제 모든 값이 문자열이므로 .str 접근자가 에러를 내지 않습니다.
    # 'nan', 'null' 등의 문자열도 걸러냅니다.
    df = df[~df['Sector'].isin(['Unknown', 'nan', 'NaN'])] 
    df = df[~df['Sector'].str.contains('기타', na=False)]
    
    # 필요한 컬럼만 리턴
    return df[required_cols]

def get_top_stocks_per_sector(df, top_n=30):
    """섹터별 시가총액 상위 N개 필터링"""
    return df.sort_values(['Sector', 'Marcap'], ascending=[True, False]).groupby('Sector').head(top_n)

def fetch_price_data_parallel(codes, days=365):
    """병렬 주가 데이터 수집"""
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    price_data = {}
    
    def fetch(code):
        try:
            df = fdr.DataReader(code, start_date)
            return code, df['Close']
        except:
            return code, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch, code) for code in codes]
        for future in concurrent.futures.as_completed(futures):
            code, series = future.result()
            if series is not None:
                price_data[code] = series
    
    df_prices = pd.DataFrame(price_data)
    df_prices = df_prices.fillna(method='ffill').dropna(axis=1)
    return df_prices

def calculate_pairs(price_df, ticker_map, min_corr=0.8, p_val_thresh=0.05):
    """상관계수 선검사 -> 공적분 검사 -> Z-Score 산출"""
    pairs = []
    corr_matrix = price_df.corr()
    cols = corr_matrix.columns
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            stock_a = cols[i]
            stock_b = cols[j]
            corr = corr_matrix.iloc[i, j]
            
            if corr > min_corr:
                series_a = price_df[stock_a]
                series_b = price_df[stock_b]
                
                score, pvalue, _ = coint(np.log(series_a), np.log(series_b))
                
                if pvalue < p_val_thresh:
                    x = sm.add_constant(np.log(series_b))
                    y = np.log(series_a)
                    model = sm.OLS(y, x).fit()
                    spread = y - model.predict(x)
                    z_score = (spread - spread.mean()) / spread.std()
                    
                    pairs.append({
                        'Display': f"{ticker_map[stock_a]} - {ticker_map[stock_b]}",
                        'Stock A': ticker_map[stock_a],
                        'Stock B': ticker_map[stock_b],
                        'Correlation': corr,
                        'P-Value': pvalue,
                        'Current Z-Score': z_score.iloc[-1],
                        'Code A': stock_a,
                        'Code B': stock_b,
                        'Model': model
                    })
    
    return pd.DataFrame(pairs)

def plot_bloomberg_style(price_df, pair_info):
    """차트 시각화"""
    stock_a_code = pair_info['Code A']
    stock_b_code = pair_info['Code B']
    
    series_a = np.log(price_df[stock_a_code])
    series_b = np.log(price_df[stock_b_code])
    
    x = sm.add_constant(series_b)
    model = pair_info['Model']
    spread = series_a - model.predict(x)
    z_score = (spread - spread.mean()) / spread.std()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.6, 0.4],
                        subplot_titles=("Price Performance (Log Normalized)", "Spread Z-Score"))

    norm_a = (series_a - series_a.iloc[0]) 
    norm_b = (series_b - series_b.iloc[0])
    
    fig.add_trace(go.Scatter(x=series_a.index, y=norm_a, mode='lines', name=pair_info['Stock A'], line=dict(color=BLOOMBERG_THEME['line_colors'][0], width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=series_b.index, y=norm_b, mode='lines', name=pair_info['Stock B'], line=dict(color=BLOOMBERG_THEME['line_colors'][1], width=1.5)), row=1, col=1)

    fig.add_trace(go.Scatter(x=z_score.index, y=z_score, mode='lines', name='Z-Score', line=dict(color='#ffffff', width=1)), row=2, col=1)
    
    fig.add_hline(y=2.0, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=-2.0, line_dash="dot", line_color="green", row=2, col=1)
    fig.add_hline(y=0, line_color="gray", row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BLOOMBERG_THEME['paper_bgcolor'],
        plot_bgcolor=BLOOMBERG_THEME['bgcolor'],
        font=dict(color=BLOOMBERG_THEME['font_color']),
        height=600,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor=BLOOMBERG_THEME['grid_color'])
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor=BLOOMBERG_THEME['grid_color'])

    return fig

# --- 메인 로직 ---

st.title("Pair Trading Analysis")
st.markdown("Automated Cointegration Scanner & Z-Score Analysis")

# Session State
if 'market_data' not in st.session_state:
    st.session_state['market_data'] = None
if 'sector_list' not in st.session_state:
    st.session_state['sector_list'] = []

# --- 섹션 1: 데이터 로드 ---
st.subheader("1. Market Data Retrieval")

if st.button("전체 종목 데이터 불러오기", type="primary"):
    with st.spinner("KRX 데이터 조회 중..."):
        df_market = get_stock_list()
        st.session_state['market_data'] = df_market
        st.session_state['sector_list'] = df_market['Sector'].unique().tolist()
    st.success(f"데이터 로드 완료. 총 {len(df_market)}개 종목.")

if st.session_state['market_data'] is not None:
    df_market = st.session_state['market_data']
    
    # 섹터별 Top 5 (Clean Table)
    st.markdown("**Sector Top 5 (by Market Cap)**")
    top5_df = df_market.sort_values(['Sector', 'Marcap'], ascending=[True, False]).groupby('Sector').head(5)
    
    st.dataframe(
        top5_df[['Sector', 'Name', 'Close', 'ChgesRatio', 'Marcap']],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Marcap": st.column_config.NumberColumn("시가총액", format="%d 억"),
            "Close": st.column_config.NumberColumn("현재가", format="%d 원"),
            "ChgesRatio": st.column_config.NumberColumn("등락률", format="%.2f %%")
        }
    )

    st.divider()

    # --- 섹션 2: 분석 ---
    st.subheader("2. Pair Strategy Analysis")
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        selected_sector = st.selectbox("섹터 선택", st.session_state['sector_list'])
        run_analysis = st.button("분석 실행", type="secondary")
        
    if run_analysis and selected_sector:
        st.write(f"**Target:** {selected_sector} 섹터 시가총액 상위 30개 종목")
        
        # 필터링
        sector_stocks = df_market[df_market['Sector'] == selected_sector]
        top30_stocks = get_top_stocks_per_sector(sector_stocks, top_n=30)
        target_codes = top30_stocks['Code'].tolist()
        ticker_map = dict(zip(top30_stocks['Code'], top30_stocks['Name']))
        
        # 데이터 다운로드
        with st.spinner("주가 데이터 다운로드 및 연산 중 (Parallel Fetching)..."):
            price_df = fetch_price_data_parallel(target_codes)
            pair_results = calculate_pairs(price_df, ticker_map)
            
        if not pair_results.empty:
            st.success(f"Cointegration 검증 완료: {len(pair_results)}개 페어 발견")
            
            pair_results = pair_results.sort_values('P-Value')
            
            # UI: List & Chart
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.markdown("**Pairs List** (P-value < 0.05)")
                selected_pair_idx = st.radio(
                    "Select a pair to visualize:", 
                    pair_results.index, 
                    format_func=lambda x: f"{pair_results.loc[x, 'Display']} (Z: {pair_results.loc[x, 'Current Z-Score']:.2f})",
                    label_visibility="collapsed"
                )
            
            with c2:
                if selected_pair_idx is not None:
                    row = pair_results.loc[selected_pair_idx]
                    fig = plot_bloomberg_style(price_df, row)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 통계 지표 (Clean Format)
                    st.markdown("#### Statistics")
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    stat_col1.metric("Correlation", f"{row['Correlation']:.4f}")
                    stat_col2.metric("P-Value", f"{row['P-Value']:.5f}")
                    stat_col3.metric("Z-Score", f"{row['Current Z-Score']:.2f}")
        else:
            st.warning("조건을 만족하는 페어가 없습니다. (Correlation > 0.8, P-value < 0.05)")
