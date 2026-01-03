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
st.set_page_config(page_title="Pro Quant Ultimate", page_icon="💎", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .metric-card { background-color: #f0f2f6; border-radius: 10px; padding: 15px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

st.title("💎 Pro Quant Ultimate: Robust Version")

# ---------------------------------------------------------
# 2. 사이드바
# ---------------------------------------------------------
with st.sidebar:
    st.header("1. Target Universe")
    universe_mode = st.radio("분석 대상", ["🦁 주식선물 가능 종목 (Hedge)", "🐳 Top 100 대규모 탐색 (Long Only)"])
    
    st.divider()
    st.header("2. Mode")
    app_mode = st.radio("실행 모드", ["📡 실시간 분석 (Live)", "🔙 포트폴리오 백테스트"])

    st.divider()
    st.header("3. Parameters")
    total_capital = st.number_input("총 투자금 (KRW)", value=10000000, step=1000000, format="%d")
    window_size = st.slider("Rolling Window", 20, 120, 60)
    z_threshold = st.slider("Z-Score Threshold", 1.5, 3.0, 2.0)
    
    # 🚨 P-value 기본값을 0.10으로 완화 (결과가 잘 나오도록)
    p_cutoff = st.slider("Max P-value", 0.01, 0.20, 0.10, help="값이 클수록 조건을 완화하여 더 많은 페어를 찾습니다.")

    st.divider()
    
    if app_mode.startswith("🔙"):
        st.header("📅 Backtest Period")
        # 기본값: 작년 1년치
        c1, c2 = st.columns(2)
        start_input = c1.date_input("Start", datetime(2023, 1, 1))
        end_input = c2.date_input("End", datetime(2023, 12, 31))
        run_label = "RUN PORTFOLIO BACKTEST"
    else:
        end_input = datetime.now()
        start_input = end_input - timedelta(days=365)
        run_label = "RUN LIVE ANALYSIS"

    run_btn = st.button(run_label, type="primary", use_container_width=True)

# ---------------------------------------------------------
# 3. 데이터 로딩 (개선된 버전: 데이터 보존율 높임)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_stock_data(universe_type, start_date, end_date):
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

    tickers_massive = tickers_futures.copy()
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
        manual_tickers = {**tickers_massive, **additional}
    else:
        manual_tickers = tickers_futures

    # [핵심 변경] 데이터 부족 방지를 위해 1년치 여유분을 더 가져옵니다.
    # 백테스트 기간이 짧아도, 공적분 계산은 긴 데이터로 해야 정확합니다.
    fetch_start = (pd.to_datetime(start_date) - timedelta(days=365)).strftime('%Y-%m-%d')
    fetch_end = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    
    tickers_list = list(manual_tickers.keys())
    all_data_list = []
    
    st_msg = st.status(f"Fetching {len(tickers_list)} stocks...", expanded=True)
    
    chunk_size = 5
    for i in range(0, len(tickers_list), chunk_size):
        chunk = tickers_list[i:i + chunk_size]
        try:
            st_msg.write(f"📥 Batch {i//chunk_size + 1}...")
            df_chunk = yf.download(chunk, start=fetch_start, end=fetch_end, progress=False)['Close']
            if isinstance(df_chunk, pd.Series): df_chunk = df_chunk.to_frame(name=chunk[0])
            all_data_list.append(df_chunk)
            time.sleep(random.uniform(0.1, 0.4))
        except: continue
        
    st_msg.update(label="Download Complete!", state="complete", expanded=False)
    
    if all_data_list:
        df_final = pd.concat(all_data_list, axis=1)
        df_final = df_final.rename(columns=manual_tickers)
        
        # [핵심 변경] dropna 조건을 완화합니다 (데이터가 일부 없어도 살림)
        # 1. 일단 앞뒤 빈값 채우기
        df_final = df_final.ffill().bfill()
        # 2. 그래도 비어있는 컬럼만 삭제 (특정 종목만 삭제되고 나머지는 유지)
        df_final = df_final.dropna(axis=1, how='any')
        
        return df_final
    return pd.DataFrame()

# ---------------------------------------------------------
# 4. 분석 엔진 (Logic 수정: Training vs Testing 분리)
# ---------------------------------------------------------
def run_analysis(df_prices, window, threshold, p_cutoff, mode, start, end):
    pairs = []
    cols = df_prices.columns
    
    # [핵심 로직 변경]
    # 공적분(관계성)은 '전체 기간'으로 확인하고,
    # 수익률 계산만 '설정된 기간'으로 수행합니다.
    # 이렇게 해야 기간을 짧게 잡아도 페어가 나옵니다.
    
    if len(df_prices) < window: return pd.DataFrame()

    prog_bar = st.progress(0)
    total_checks = len(cols) * (len(cols) - 1) // 2
    checked = 0
    
    # 백테스팅 타겟 기간 마스크
    target_mask = (df_prices.index >= pd.to_datetime(start)) & (df_prices.index <= pd.to_datetime(end))
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            sa, sb = cols[i], cols[j]
            
            if df_prices[sa].corr(df_prices[sb]) < 0.5:
                checked += 1
                continue
                
            try:
                # 1. 관계성 검증 (전체 데이터 사용 -> 페어 발견 확률 Up)
                score, pval, _ = coint(df_prices[sa], df_prices[sb])
                
                if pval < p_cutoff:
                    # 2. 지표 계산
                    log_a, log_b = np.log(df_prices[sa]), np.log(df_prices[sb])
                    spread = log_a - log_b
                    
                    mean = spread.rolling(window).mean()
                    std = spread.rolling(window).std()
                    z_all = (spread - mean) / std
                    
                    # 3. 백테스팅 (사용자가 지정한 기간만 잘라서 계산)
                    z_target = z_all.loc[target_mask]
                    
                    if z_target.empty: continue # 기간 내 데이터 없음
                    
                    pos = np.where(z_target < -threshold, 1, np.where(z_target > threshold, -1, 0))
                    
                    ret_a = df_prices[sa].loc[target_mask].pct_change().fillna(0)
                    ret_b = df_prices[sb].loc[target_mask].pct_change().fillna(0)
                    
                    spr_ret = (ret_a - ret_b) * pd.Series(pos).shift(1).fillna(0).values
                    cum_ret = (1 + spr_ret).cumprod() - 1
                    
                    curr_z = z_all.iloc[-1]
                    status = "Watch"
                    if curr_z < -threshold: status = "Buy A"
                    elif curr_z > threshold: status = "Buy B"
                    
                    pairs.append({
                        'Stock A': sa, 'Stock B': sb,
                        'P-value': pval, 'Z-Score': curr_z,
                        'Status': status, 'Final_Ret': cum_ret[-1],
                        'Spread': spread, 'Mean': mean, 'Std': std,
                        'Cum_Ret_Series': cum_ret, 
                        'Daily_Ret_Series': spr_ret,
                        'Analysis_Dates': z_target.index,
                        'Price A': df_prices[sa].iloc[-1], 'Price B': df_prices[sb].iloc[-1]
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
    titles = (f"Price: {sa} vs {sb}", "Cumulative Return", "Z-Score") if rows==3 else (f"Price: {sa} vs {sb}", "Z-Score")
    
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=titles)
    
    pa, pb = df_prices[sa].loc[dates], df_prices[sb].loc[dates]
    fig.add_trace(go.Scatter(x=dates, y=(pa/pa.iloc[0])*100, name=sa, line=dict(color='#1f77b4')), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=(pb/pb.iloc[0])*100, name=sb, line=dict(color='#ff7f0e')), row=1, col=1)
    
    if rows == 3:
        fig.add_trace(go.Scatter(x=dates, y=row['Cum_Ret_Series']*100, name='Profit %', line=dict(color='green'), fill='tozeroy'), row=2, col=1)
    
    z_row = rows
    z_vals = ((row['Spread'] - row['Mean']) / row['Std']).loc[dates]
    fig.add_trace(go.Scatter(x=dates, y=z_vals, name='Z-Score', line=dict(color='#9467bd')), row=z_row, col=1)
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", row=z_row, col=1)
    fig.add_hline(y=-threshold, line_dash="dash", line_color="blue", row=z_row, col=1)
    
    fig.update_layout(height=700 if rows==3 else 500, hovermode="x unified", margin=dict(t=30, b=20, l=20, r=20))
    return fig

# ---------------------------------------------------------
# 6. 메인 실행
# ---------------------------------------------------------
if run_btn:
    # 1. 데이터 확인
    with st.spinner("데이터 준비 중..."):
        df_prices = load_stock_data(universe_mode, start_input, end_input)

    if df_prices.empty or len(df_prices.columns) < 2:
        st.error("🚨 데이터를 불러오지 못했습니다. (종목 수 부족)")
        st.info("Tip: 잠시 후 다시 시도하거나, 앱을 재부팅(Reboot) 해주세요.")
    else:
        st.success(f"✅ 데이터 로드 완료: {len(df_prices.columns)}개 종목 (기간: {df_prices.index[0].date()} ~ {df_prices.index[-1].date()})")
        
        # 2. 분석 실행
        results = run_analysis(df_prices, window_size, z_threshold, p_cutoff, app_mode, start_input, end_input)
        
        if results.empty:
            st.warning("⚠️ 조건에 맞는 페어를 찾지 못했습니다.")
            st.info("👉 **해결 방법:** 사이드바에서 'Max P-value'를 0.15~0.20으로 높여보세요.")
        else:
            if app_mode.startswith("🔙"):
                st.markdown(f"### 📊 Portfolio Backtest Result")
                st.info(f"시스템이 찾아낸 **총 {len(results)}개의 페어**로 포트폴리오를 구성했습니다.")

                # 포트폴리오 계산
                all_returns_df = pd.DataFrame(index=pd.date_range(start=start_input, end=end_input))
                for idx, row in results.iterrows():
                    series = row['Daily_Ret_Series']
                    series.index = pd.to_datetime(series.index) # 인덱스 통일
                    # 중복 인덱스 제거 및 리인덱싱
                    series = series[~series.index.duplicated(keep='first')]
                    series = series.reindex(all_returns_df.index).fillna(0)
                    all_returns_df[f"{row['Stock A']}-{row['Stock B']}"] = series

                portfolio_daily_ret = all_returns_df.mean(axis=1).fillna(0)
                portfolio_cum_ret = (1 + portfolio_daily_ret).cumprod() - 1
                
                # MDD
                wealth = (1 + portfolio_daily_ret).cumprod()
                peak = wealth.expanding(min_periods=1).max()
                dd = (wealth - peak) / peak
                mdd = dd.min()

                k1, k2, k3 = st.columns(3)
                k1.metric("💰 Portfolio Return", f"{portfolio_cum_ret.iloc[-1]*100:.2f}%")
                k2.metric("📉 MDD", f"{mdd*100:.2f}%")
                k3.metric("🧩 Pairs", f"{len(results)} ea")
                
                # 차트
                fig_port = go.Figure()
                fig_port.add_trace(go.Scatter(x=portfolio_cum_ret.index, y=portfolio_cum_ret*100, mode='lines', name='Portfolio', line=dict(color='#00C805', width=3), fill='tozeroy'))
                fig_port.add_hline(y=0, line_color="gray")
                fig_port.update_layout(title="<b>Portfolio Equity Curve</b>", height=500, hovermode="x unified")
                st.plotly_chart(fig_port, use_container_width=True)
                
                st.divider()
                st.subheader("🔍 Individual Pair Performance (Top 5)")
                for idx, row in results.sort_values('Final_Ret', ascending=False).head(5).iterrows():
                    with st.expander(f" 수익률 {row['Final_Ret']*100:.2f}% | {row['Stock A']} vs {row['Stock B']}", expanded=False):
                        st.plotly_chart(plot_chart(row, df_prices, z_threshold, app_mode), use_container_width=True)

            else:
                # Live Mode
                actives = results[results['Status'] != 'Watch']
                st.metric("Active Signals", f"{len(actives)}", f"Total Analyzed: {len(results)}")
                
                if not actives.empty:
                    for idx, row in actives.sort_values(by='Z-Score', key=abs, ascending=False).iterrows():
                        alloc = total_capital / 2
                        qa = int(alloc / row['Price A'])
                        qb = int(alloc / row['Price B'])
                        sa, sb = row['Stock A'], row['Stock B']
                        
                        msg = f"🔵 Buy {sa} ({qa:,}주) | 🔴 Sell {sb} ({qb:,}주)" if row['Status']=="Buy A" else f"🔴 Sell {sa} ({qa:,}주) | 🔵 Buy {sb} ({qb:,}주)"
                        clr = "green" if row['Status'].startswith("Buy") else "red"
                        
                        with st.expander(f":{clr}[Signal] {sa} vs {sb} (Z: {row['Z-Score']:.2f})", expanded=True):
                            st.info(msg)
                            st.plotly_chart(plot_chart(row, df_prices, z_threshold, app_mode), use_container_width=True)
                else:
                    st.info("현재 진입 신호가 없습니다.")
                    st.dataframe(results[['Stock A', 'Stock B', 'Z-Score', 'P-value']].sort_values('P-value'))
else:
    st.info("👈 설정 후 실행 버튼을 눌러주세요.")
