import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import coint
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import warnings
import time
import random

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. 페이지 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="Pair Trading System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card { background-color: #f0f2f6; border-radius: 10px; padding: 15px; margin: 10px 0; }
    .buy-signal { color: #2E7D32; font-weight: bold; }
    .sell-signal { color: #C62828; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("Pair Trading System Dashboard")

# ---------------------------------------------------------
# 2. 사이드바 (전략 및 유니버스 선택)
# ---------------------------------------------------------
with st.sidebar:
    st.header("1. Target Universe")
    
    # [핵심] 유니버스 선택 (1번 요청 vs 3번 요청)
    universe_mode = st.radio(
        "분석 대상 그룹 선택",
        ["🦁 주식선물 가능 종목 (Hedge)", "🐳 Top 100 대규모 탐색 (Long Only)"]
    )
    
    st.info(
        "🦁 **선물 모드**: 개별 주식 선물이 상장된 우량주 위주. (공매도/선물매도 가능)\n\n"
        "🐳 **대규모 모드**: 시총 상위 100개. '저평가 매수' 기회 포착용."
    )
    
    st.divider()
    st.header("2. Analysis Mode")
    app_mode = st.radio("분석 모드", ["📡 실시간 분석 (Live)", "🔙 과거 백테스팅 (Backtest)"])

    st.divider()
    st.header("3. Parameters")
    total_capital = st.number_input("총 투자금 (KRW)", value=10000000, step=1000000, format="%d")
    window_size = st.slider("Rolling Window", 20, 120, 60)
    z_threshold = st.slider("Z-Score Threshold", 1.5, 3.0, 2.0)
    
    # 대규모 모드일 때는 속도를 위해 P-value 기준을 조금 엄격하게 잡는 게 좋음
    default_p = 0.05 if universe_mode.startswith("🐳") else 0.10
    p_cutoff = st.slider("Max P-value", 0.01, 0.20, default_p)

    st.divider()
    
    if app_mode.startswith("🔙"):
        st.header("📅 Backtest Period")
        c1, c2 = st.columns(2)
        start_input = c1.date_input("Start", datetime(2025, 1, 1))
        end_input = c2.date_input("End", datetime(2025, 12, 31))
        run_label = "RUN BACKTEST"
    else:
        end_input = datetime.now()
        start_input = end_input - timedelta(days=365)
        run_label = "RUN LIVE ANALYSIS"

    run_btn = st.button(run_label, type="primary", use_container_width=True)

# ---------------------------------------------------------
# 3. 데이터 로딩 (유니버스별 리스트 분리)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_stock_data(universe_type, start_date, end_date):
    
    # 1번 요청: 선물 있는 종목 (유동성 풍부, 공매도/선물매도 가능)
    tickers_futures = {
        '005930.KS': '삼성전자', '000660.KS': 'SK하이닉스', '005380.KS': '현대차', '000270.KS': '기아',
        '005490.KS': 'POSCO홀딩스', '006400.KS': '삼성SDI', '051910.KS': 'LG화학', '035420.KS': 'NAVER',
        '035720.KS': '카카오', '105560.KS': 'KB금융', '055550.KS': '신한지주', '086790.KS': '하나금융지주',
        '000810.KS': '삼성화재', '032830.KS': '삼성생명', '015760.KS': '한국전력', '034020.KS': '두산에너빌리티',
        '012330.KS': '현대모비스', '009540.KS': 'HD한국조선해양', '010140.KS': '삼성중공업', '042660.KS': '한화오션',
        '011200.KS': 'HMM', '003490.KS': '대한항공', '030200.KS': 'KT', '017670.KS': 'SK텔레콤',
        '009150.KS': '삼성전기', '011070.KS': 'LG이노텍', '018260.KS': '삼성SDS', '259960.KS': '크래프톤',
        '036570.KS': '엔씨소프트', '251270.KS': '넷마블', '090430.KS': '아모레퍼시픽', '097950.KS': 'CJ제일제당',
        '010130.KS': '고려아연', '004020.KS': '현대제철', '010950.KS': 'S-Oil', '096770.KS': 'SK이노베이션',
        '323410.KS': '카카오뱅크', '377300.KS': '카카오페이', '034730.KS': 'SK', '003550.KS': 'LG',
        '247540.KQ': '에코프로비엠', '086520.KQ': '에코프로', '028300.KQ': 'HLB', '293490.KQ': '카카오게임즈',
        '066970.KQ': '엘앤에프', '035900.KQ': 'JYP Ent.', '041510.KQ': '에스엠', '263750.KQ': '펄어비스'
    }

    # 3번 요청: 최대한 많은 종목 (Top 100 + 알짜 코스닥)
    tickers_massive = tickers_futures.copy() # 선물 종목 포함하고 추가
    additional = {
        '373220.KS': 'LG에너지솔루션', '207940.KS': '삼성바이오로직스', '068270.KS': '셀트리온', 
        '000100.KS': '유한양행', '128940.KS': '한미약품', '196170.KQ': '알테오젠', '214150.KQ': '클래시스',
        '145020.KQ': '휴젤', '042700.KS': '한미반도체', '403870.KQ': 'HPSP', '071050.KS': '한국금융지주',
        '024110.KS': '기업은행', '316140.KS': '우리금융지주', '000120.KS': 'CJ대한통운', '028670.KS': '팬오션',
        '010120.KS': 'LS ELECTRIC', '267250.KS': 'HD현대일렉트릭', '012450.KS': '한화에어로스페이스',
        '047810.KS': '한국항공우주', '079550.KS': 'LIG넥스원', '021240.KS': '코웨이', '033780.KS': 'KT&G',
        '004370.KS': '농심', '007310.KS': '오뚜기', '271560.KS': '오리온', '280360.KS': '롯데웰푸드',
        '005940.KS': 'NH투자증권', '016360.KS': '삼성증권', '039490.KS': '키움증권', '001450.KS': '현대해상',
        '000150.KS': '두산', '278280.KQ': '천보', '365550.KS': '성일하이텍', '137400.KQ': '피엔티'
    }
    
    if universe_type.startswith("🐳"):
        manual_tickers = {**tickers_massive, **additional} # 딕셔너리 병합
    else:
        manual_tickers = tickers_futures

    # 데이터 다운로드 (Chunking)
    fetch_start = (pd.to_datetime(start_date) - timedelta(days=150)).strftime('%Y-%m-%d')
    fetch_end = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    
    tickers_list = list(manual_tickers.keys())
    all_data_list = []
    
    st_msg = st.status(f"Fetching {len(tickers_list)} stocks ({fetch_start} ~)...", expanded=True)
    
    chunk_size = 5
    for i in range(0, len(tickers_list), chunk_size):
        chunk = tickers_list[i:i + chunk_size]
        try:
            st_msg.write(f"📥 Batch {i//chunk_size + 1}...")
            df_chunk = yf.download(chunk, start=fetch_start, end=fetch_end, progress=False)['Close']
            if isinstance(df_chunk, pd.Series): df_chunk = df_chunk.to_frame(name=chunk[0])
            all_data_list.append(df_chunk)
            time.sleep(random.uniform(0.1, 0.4)) # 약간 더 빠르게
        except: continue
        
    st_msg.update(label="Download Complete!", state="complete", expanded=False)
    
    if all_data_list:
        df_final = pd.concat(all_data_list, axis=1)
        df_final = df_final.rename(columns=manual_tickers)
        return df_final.ffill().dropna(axis=1)
    return pd.DataFrame()

# ---------------------------------------------------------
# 4. 분석 엔진
# ---------------------------------------------------------
def run_analysis(df_prices, window, threshold, p_cutoff, mode, start, end):
    pairs = []
    cols = df_prices.columns
    
    if mode.startswith("🔙"):
        mask = (df_prices.index >= pd.to_datetime(start)) & (df_prices.index <= pd.to_datetime(end))
        df_anl = df_prices.loc[mask]
    else:
        df_anl = df_prices
        
    if len(df_anl) < window: return pd.DataFrame()

    prog_bar = st.progress(0)
    # 효율성: 상관계수 먼저 체크하여 연산량 줄이기
    
    total_checks = len(cols) * (len(cols) - 1) // 2
    checked = 0
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            sa, sb = cols[i], cols[j]
            
            # [최적화] 상관계수가 0.5 미만이면 공적분 검사 안함 (속도 향상)
            if df_anl[sa].corr(df_anl[sb]) < 0.5:
                checked += 1
                continue
                
            try:
                score, pval, _ = coint(df_anl[sa], df_anl[sb])
                if pval < p_cutoff:
                    # Metrics
                    log_a, log_b = np.log(df_prices[sa]), np.log(df_prices[sb])
                    spread = log_a - log_b
                    
                    mean = spread.rolling(window).mean()
                    std = spread.rolling(window).std()
                    z_all = (spread - mean) / std
                    
                    z_period = z_all.loc[df_anl.index]
                    
                    # Backtest Returns
                    pos = np.where(z_period < -threshold, 1, np.where(z_period > threshold, -1, 0))
                    ret_a, ret_b = df_anl[sa].pct_change().fillna(0), df_anl[sb].pct_change().fillna(0)
                    spr_ret = (ret_a - ret_b) * pd.Series(pos).shift(1).fillna(0).values
                    cum_ret = (1 + spr_ret).cumprod() - 1
                    
                    # Status
                    curr_z = z_all.iloc[-1]
                    status = "Watch"
                    if curr_z < -threshold: status = "Buy A" # Short B implied
                    elif curr_z > threshold: status = "Buy B" # Short A implied
                    
                    pairs.append({
                        'Stock A': sa, 'Stock B': sb,
                        'P-value': pval, 'Z-Score': curr_z,
                        'Status': status, 'Final_Ret': cum_ret[-1],
                        'Spread': spread, 'Mean': mean, 'Std': std,
                        'Cum_Ret_Series': cum_ret, 'Analysis_Dates': df_anl.index,
                        'Price A': df_anl[sa].iloc[-1], 'Price B': df_anl[sb].iloc[-1]
                    })
            except: pass
            
            checked += 1
            if checked % 50 == 0: prog_bar.progress(min(checked / total_checks, 1.0))
            
    prog_bar.empty()
    return pd.DataFrame(pairs)

# ---------------------------------------------------------
# 5. 차트 그리기
# ---------------------------------------------------------
def plot_chart(row, df_prices, threshold, mode):
    sa, sb = row['Stock A'], row['Stock B']
    dates = row['Analysis_Dates']
    
    rows = 3 if mode.startswith("🔙") else 2
    titles = (f"Price: {sa} vs {sb}", "Cumulative Return", "Z-Score") if rows == 3 else (f"Price: {sa} vs {sb}", "Z-Score")
    height = 700 if rows == 3 else 500
    
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=titles)
    
    # 1. Price
    pa, pb = df_prices[sa].loc[dates], df_prices[sb].loc[dates]
    fig.add_trace(go.Scatter(x=dates, y=(pa/pa.iloc[0])*100, name=sa, line=dict(color='#1f77b4')), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=(pb/pb.iloc[0])*100, name=sb, line=dict(color='#ff7f0e')), row=1, col=1)
    
    # 2. Return (Backtest only)
    if rows == 3:
        fig.add_trace(go.Scatter(x=dates, y=row['Cum_Ret_Series']*100, name='Profit %', line=dict(color='green'), fill='tozeroy'), row=2, col=1)
    
    # 3. Z-Score
    z_row = rows
    z_vals = ((row['Spread'] - row['Mean']) / row['Std']).loc[dates]
    fig.add_trace(go.Scatter(x=dates, y=z_vals, name='Z-Score', line=dict(color='#9467bd')), row=z_row, col=1)
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", row=z_row, col=1)
    fig.add_hline(y=-threshold, line_dash="dash", line_color="blue", row=z_row, col=1)
    
    fig.update_layout(height=height, hovermode="x unified", margin=dict(t=30, b=20, l=20, r=20))
    return fig

# ---------------------------------------------------------
# 6. 메인 실행
# ---------------------------------------------------------
if run_btn:
    df_prices = load_stock_data(universe_mode, start_input, end_input)
    
    if df_prices.empty:
        st.error("Data Load Failed.")
    else:
        results = run_analysis(df_prices, window_size, z_threshold, p_cutoff, app_mode, start_input, end_input)
        
        if results.empty:
            st.warning("조건에 맞는 페어가 없습니다.")
        else:
            # ------------------------------------------------
            # 모드별 표시 로직
            # ------------------------------------------------
            if app_mode.startswith("🔙"):
                # Backtest Result
                best = results.loc[results['Final_Ret'].idxmax()]
                st.success(f"🏆 Best Pair: {best['Stock A']} - {best['Stock B']} ({best['Final_Ret']*100:.1f}%)")
                
                for idx, row in results.sort_values('Final_Ret', ascending=False).head(5).iterrows():
                    with st.expander(f" 수익률 {row['Final_Ret']*100:.2f}% | {row['Stock A']} vs {row['Stock B']}", expanded=(idx==0)):
                        st.plotly_chart(plot_chart(row, df_prices, z_threshold, app_mode), use_container_width=True)
            else:
                # Live Result
                actives = results[results['Status'] != 'Watch']
                st.metric("Active Signals", f"{len(actives)}", f"Total Analyzed: {len(results)}")
                
                if not actives.empty:
                    # 유니버스 모드에 따라 메시지 다르게
                    is_futures = universe_mode.startswith("🦁")
                    
                    for idx, row in actives.sort_values(by='Z-Score', key=abs, ascending=False).iterrows():
                        # 수량 계산
                        alloc = total_capital / 2
                        qa = int(alloc / row['Price A'])
                        qb = int(alloc / row['Price B'])
                        
                        sa, sb = row['Stock A'], row['Stock B']
                        z = row['Z-Score']
                        
                        # 메시지 생성
                        if row['Status'] == "Buy A":
                            # Z < -2 : A가 저평가
                            if is_futures:
                                msg = f"🔵 **현물 매수** {sa} ({qa:,}주)  |  🔴 **선물 매도** {sb} ({qb:,}주)"
                                title_clr = "green"
                            else:
                                # 대규모 모드 (Long Only)
                                msg = f"💡 **강력 매수 기회**: {sa} ({qa:,}주) \n\n (참고: {sb} 대비 저평가됨)"
                                title_clr = "green"
                        else:
                            # Z > 2 : A가 고평가 (B가 저평가)
                            if is_futures:
                                msg = f"🔴 **선물 매도** {sa} ({qa:,}주)  |  🔵 **현물 매수** {sb} ({qb:,}주)"
                                title_clr = "red"
                            else:
                                # 대규모 모드 (Long Only)
                                msg = f"💡 **강력 매수 기회**: {sb} ({qb:,}주) \n\n (참고: {sa} 대비 저평가됨)"
                                title_clr = "green" # Buy 관점에서 표시
                        
                        with st.expander(f":{title_clr}[Signal] {sa} vs {sb} (Z: {z:.2f})", expanded=True):
                            st.info(msg)
                            st.plotly_chart(plot_chart(row, df_prices, z_threshold, app_mode), use_container_width=True)
                else:
                    st.info("현재 진입 신호가 없습니다.")
                    st.dataframe(results[['Stock A', 'Stock B', 'Z-Score', 'P-value']].sort_values('P-value'))
else:
    st.info("👈 설정 후 실행 버튼을 눌러주세요.")
