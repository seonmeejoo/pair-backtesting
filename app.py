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
st.set_page_config(layout="wide", page_title="Pair Trading Scanner", page_icon="📈")

# --- 스타일 정의 (블룸버그 스타일 테마) ---
BLOOMBERG_THEME = {
    'bgcolor': '#1e1e1e',
    'paper_bgcolor': '#121212',
    'font_color': '#e0e0e0',
    'grid_color': '#444444',
    'line_colors': ['#ff9f1c', '#2ec4b6']  # 오렌지, 청록 (가독성 높은 대비)
}

# --- 함수 정의 ---

@st.cache_data
def get_stock_list():
    """KRX 전체 종목을 가져와서 섹터별로 정리합니다."""
    df = fdr.StockListing('KRX')
    
    # 필요한 컬럼만 선택 및 정리
    df = df[['Code', 'Name', 'Sector', 'Marcap', 'Close', 'ChgesRatio']]
    df = df.dropna(subset=['Sector']) # 섹터 없는 것 제거
    df = df[~df['Sector'].str.contains('기타', na=False)] # "기타" 섹터 제외 (요청사항)
    
    return df

def get_top_stocks_per_sector(df, top_n=30):
    """각 섹터별 시가총액 상위 N개만 필터링합니다."""
    return df.sort_values(['Sector', 'Marcap'], ascending=[True, False]).groupby('Sector').head(top_n)

def fetch_price_data_parallel(codes, days=365):
    """병렬 처리로 주가 데이터를 빠르게 다운로드합니다."""
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    price_data = {}
    
    def fetch(code):
        try:
            df = fdr.DataReader(code, start_date)
            return code, df['Close']
        except:
            return code, None

    # ThreadPoolExecutor를 사용한 병렬 다운로드
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch, code) for code in codes]
        for future in concurrent.futures.as_completed(futures):
            code, series = future.result()
            if series is not None:
                price_data[code] = series
    
    # DataFrame으로 변환 및 결측치 처리 (ffill)
    df_prices = pd.DataFrame(price_data)
    df_prices = df_prices.fillna(method='ffill').dropna(axis=1) # 데이터가 너무 없는 종목은 제외
    return df_prices

def calculate_pairs(price_df, ticker_map, min_corr=0.8, p_val_thresh=0.05):
    """
    1. 상관계수 > 0.8 (Fast Screening)
    2. 공적분 검사 (Cointegration Test)
    3. Z-score 계산
    """
    pairs = []
    
    # 1. 벡터화된 상관계수 계산 (고속)
    corr_matrix = price_df.corr()
    
    # 상부 삼각행렬만 사용하여 중복 제거 및 자기 자신 제외
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            stock_a = cols[i]
            stock_b = cols[j]
            corr = corr_matrix.iloc[i, j]
            
            # 2. 상관계수 필터링
            if corr > min_corr:
                # 3. 공적분 검사 (Cointegration) - 무거운 작업이므로 여기서 수행
                series_a = price_df[stock_a]
                series_b = price_df[stock_b]
                
                # 로그 가격 사용 (일반적인 페어 트레이딩 관행)
                score, pvalue, _ = coint(np.log(series_a), np.log(series_b))
                
                if pvalue < p_val_thresh:
                    # Z-Score 계산을 위한 Spread 생성 (OLS 회귀)
                    # Y = beta * X + alpha
                    x = sm.add_constant(np.log(series_b))
                    y = np.log(series_a)
                    model = sm.OLS(y, x).fit()
                    spread = y - model.predict(x)
                    z_score = (spread - spread.mean()) / spread.std()
                    
                    pairs.append({
                        'Stock A': f"{ticker_map[stock_a]} ({stock_a})",
                        'Stock B': f"{ticker_map[stock_b]} ({stock_b})",
                        'Correlation': corr,
                        'P-Value': pvalue,
                        'Current Z-Score': z_score.iloc[-1], # 가장 최근 Z-score
                        'Code A': stock_a,
                        'Code B': stock_b,
                        'Model': model
                    })
    
    return pd.DataFrame(pairs)

def plot_bloomberg_style(price_df, pair_info):
    """쌔끈한 블룸버그 스타일 차트"""
    stock_a_code = pair_info['Code A']
    stock_b_code = pair_info['Code B']
    
    series_a = np.log(price_df[stock_a_code])
    series_b = np.log(price_df[stock_b_code])
    
    # Spread 재계산
    x = sm.add_constant(series_b)
    model = pair_info['Model']
    spread = series_a - model.predict(x)
    z_score = (spread - spread.mean()) / spread.std()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.6, 0.4],
                        subplot_titles=("Normalized Price Performance (Log)", "Spread Z-Score"))

    # 상단: 주가 비교 (정규화하여 시작점 맞춤)
    norm_a = (series_a - series_a.iloc[0]) 
    norm_b = (series_b - series_b.iloc[0])
    
    fig.add_trace(go.Scatter(x=series_a.index, y=norm_a, mode='lines', name=pair_info['Stock A'], line=dict(color=BLOOMBERG_THEME['line_colors'][0], width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=series_b.index, y=norm_b, mode='lines', name=pair_info['Stock B'], line=dict(color=BLOOMBERG_THEME['line_colors'][1], width=1.5)), row=1, col=1)

    # 하단: Z-Score
    fig.add_trace(go.Scatter(x=z_score.index, y=z_score, mode='lines', name='Z-Score', line=dict(color='#ffffff', width=1)), row=2, col=1)
    
    # Z-Score 밴드 (Entry/Exit signals)
    fig.add_hline(y=2.0, line_dash="dot", line_color="red", row=2, col=1, annotation_text="Short Spread")
    fig.add_hline(y=-2.0, line_dash="dot", line_color="green", row=2, col=1, annotation_text="Long Spread")
    fig.add_hline(y=0, line_color="gray", row=2, col=1)

    # 레이아웃 커스터마이징 (Dark Theme)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BLOOMBERG_THEME['paper_bgcolor'],
        plot_bgcolor=BLOOMBERG_THEME['bgcolor'],
        font=dict(color=BLOOMBERG_THEME['font_color']),
        height=700,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor=BLOOMBERG_THEME['grid_color'])
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor=BLOOMBERG_THEME['grid_color'])

    return fig

# --- 메인 로직 ---

st.title("🐍 Quant Pair Trading Scanner")
st.markdown("Top 30 Market Cap | High Correlation | Cointegration | Bloomberg Style Viz")

# Session State 초기화
if 'market_data' not in st.session_state:
    st.session_state['market_data'] = None
if 'sector_list' not in st.session_state:
    st.session_state['sector_list'] = []

# --- STEP 1: 데이터 로드 ---
st.header("Step 1. Market Data Overview")

if st.button("🔄 네이버(KRX) 전체 데이터 조회", type="primary"):
    with st.spinner("데이터를 긁어오는 중입니다..."):
        df_market = get_stock_list()
        st.session_state['market_data'] = df_market
        st.session_state['sector_list'] = df_market['Sector'].unique().tolist()
    st.success("데이터 로드 완료!")

if st.session_state['market_data'] is not None:
    df_market = st.session_state['market_data']
    
    # 섹터별 TOP 5 보여주기
    st.subheader("📊 Sector Top 5 Leaders (By Market Cap)")
    
    top5_df = df_market.sort_values(['Sector', 'Marcap'], ascending=[True, False]).groupby('Sector').head(5)
    
    # 깔끔한 테이블 디스플레이
    st.dataframe(
        top5_df[['Sector', 'Name', 'Code', 'Close', 'ChgesRatio', 'Marcap']],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Marcap": st.column_config.NumberColumn("시가총액", format="%d 억"),
            "Close": st.column_config.NumberColumn("현재가", format="%d 원"),
            "ChgesRatio": st.column_config.NumberColumn("등락률", format="%.2f %%")
        }
    )

    st.divider()

    # --- STEP 2: 페어 분석 ---
    st.header("Step 2. Pair Analysis (Cointegration)")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_sector = st.selectbox("분석할 섹터를 선택하세요", st.session_state['sector_list'])
        run_analysis = st.button("🚀 페어링 분석 실행", type="secondary")
        
    if run_analysis and selected_sector:
        st.info(f"[{selected_sector}] 섹터 분석 시작...")
        
        # 1. 섹터 필터링 & Top 30 선정 (Quality Filter)
        sector_stocks = df_market[df_market['Sector'] == selected_sector]
        top30_stocks = get_top_stocks_per_sector(sector_stocks, top_n=30)
        target_codes = top30_stocks['Code'].tolist()
        ticker_map = dict(zip(top30_stocks['Code'], top30_stocks['Name']))
        
        st.write(f"👉 시가총액 상위 {len(target_codes)}개 종목 대상으로 분석합니다.")
        
        # 2. 병렬 데이터 다운로드 (Parallel Fetching)
        with st.spinner("과거 데이터 병렬 다운로드 중... (Bloomberg Terminal Speed 흉내내는 중)"):
            price_df = fetch_price_data_parallel(target_codes)
        
        # 3. 상관계수 & 공적분 검사
        with st.spinner("상관계수 필터링 (>0.8) 및 공적분(Cointegration) 계산 중..."):
            pair_results = calculate_pairs(price_df, ticker_map)
            
        if not pair_results.empty:
            st.success(f"총 {len(pair_results)}개의 유의미한 페어를 발견했습니다!")
            
            # 결과 테이블 정렬 (P-value 낮은 순, 즉 통계적으로 가장 유의미한 순)
            pair_results = pair_results.sort_values('P-Value')
            
            # 페어 선택 UI
            st.subheader("Discoveries")
            
            # 왼쪽: 리스트 / 오른쪽: 차트
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.caption("Cointegrated Pairs (P-value < 0.05)")
                selected_pair_idx = st.radio(
                    "결과 리스트", 
                    pair_results.index, 
                    format_func=lambda x: f"{pair_results.loc[x, 'Stock A']} - {pair_results.loc[x, 'Stock B']} (Z: {pair_results.loc[x, 'Current Z-Score']:.2f})"
                )
            
            with c2:
                if selected_pair_idx is not None:
                    row = pair_results.loc[selected_pair_idx]
                    fig = plot_bloomberg_style(price_df, row)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown(f"""
                    **Pair Stats:**
                    - **Correlation:** {row['Correlation']:.4f}
                    - **Cointegration P-Value:** {row['P-Value']:.5f} (낮을수록 좋음)
                    - **Current Z-Score:** {row['Current Z-Score']:.2f} (2.0 이상이면 벌어짐 -> 평균 회귀 기대)
                    """)
        else:
            st.warning("조건(Corr > 0.8, P-value < 0.05)을 만족하는 페어가 없습니다. 섹터를 변경해보세요.")
