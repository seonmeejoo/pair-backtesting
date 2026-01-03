import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import coint
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import warnings
import time
import random

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. Clean & Minimalist UI Settings
# ---------------------------------------------------------
st.set_page_config(
    page_title="Pair Trading Scanner",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 모던하고 깔끔한 CSS
st.markdown("""
<style>
    .stApp {
        background-color: #1A1C24;
        color: #E0E0E0;
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #111317;
        border-right: 1px solid #2B2D35;
    }
    div[data-testid="metric-container"] {
        background-color: #252830;
        border: 1px solid #363945;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    div.stButton > button {
        background-color: #3B82F6;
        color: white;
        border: none;
        border-radius: 6px;
        height: 3em;
        font-weight: 600;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        background-color: #2563EB;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    h1, h2, h3 {
        color: #F3F4F6 !important;
        font-weight: 700 !important;
    }
    .streamlit-expanderHeader {
        background-color: #252830;
        border-radius: 4px;
        color: #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)

st.title("Pair Trading Scanner")
st.markdown("데이터 기반의 롱숏(Long-Short) 기회 포착 및 백테스팅")

# ---------------------------------------------------------
# 2. 직관적인 사이드바
# ---------------------------------------------------------
with st.sidebar:
    st.header("설정 (Settings)")
    
    universe_mode = st.selectbox(
        "분석 대상 그룹",
        ["KOSPI 200 (선물/헷지)", "시가총액 상위 100 (Long Only)"]
    )
    
    st.divider()
    
    app_mode = st.radio("실행 모드", ["실시간 분석 (Live)", "백테스트 (Backtest)"])

    st.divider()
    
    total_capital = st.number_input("투자 원금 (KRW)", value=10000000, step=1000000, format="%d")
    
    with st.expander("⚙️ 고급 설정 (민감도 조절)"):
        st.caption("익숙하지 않다면 기본값을 추천합니다.")
        window_size = st.slider("분석 기간 (Window)", 20, 120, 60)
        entry_z = st.slider("진입 기준 (Z-Score)", 1.5, 3.0, 2.0)
        exit_z = st.slider("익절 기준 (Z-Score)", 0.0, 1.0, 0.0)
        stop_loss_z = st.slider("손절 기준 (Z-Score)", 3.0, 6.0, 4.0)
        
        default_p = 0.05 if "상위 100" in universe_mode else 0.10
        p_cutoff = st.slider("연관성 기준 (P-value)", 0.01, 0.20, default_p)

    st.divider()
    
    if app_mode == "백테스트 (Backtest)":
        st.subheader("검증 기간")
        c1, c2 = st.columns(2)
        start_input = c1.date_input("시작일", datetime(2025, 1, 1))
        end_input = c2.date_input("종료일", datetime(2025, 12, 31))
        run_label = "백테스트 실행"
    else:
        end_input = datetime.now()
        start_input = end_input - timedelta(days=365)
        run_label = "분석 시작"

    run_btn = st.button(run_label, type="primary", use_container_width=True)

# ---------------------------------------------------------
# 3. 데이터 로딩
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_stock_data(universe_type, start_date, end_date):
    tickers_futures = {
        '005930.KS': '삼성전자', '000660.KS': 'SK하이닉스', '005380.KS': '현대차', 
        '000270.KS': '기아', '005490.KS': 'POSCO홀딩스', '006400.KS': '삼성SDI', 
        '051910.KS': 'LG화학', '035420.KS': 'NAVER', '035720.KS': '카카오', 
        '105560.KS': 'KB금융', '055550.KS': '신한지주', '086790.KS': '하나금융지주',
        '000810.KS': '삼성화재', '032830.KS': '삼성생명', '015760.KS': '한국전력', 
        '012330.KS': '현대모비스', '009540.KS': 'HD한국조선해양', '042660.KS': '한화오션', 
        '011200.KS': 'HMM', '003490.KS': '대한항공', '030200.KS': 'KT', '017670.KS': 'SK텔레콤',
        '009150.KS': '삼성전기', '011070.KS': 'LG이노텍', '018260.KS': '삼성SDS', 
        '259960.KS': '크래프톤', '036570.KS': '엔씨소프트', '251270.KS': '넷마블', 
        '090430.KS': '아모레퍼시픽', '097950.KS': 'CJ제일제당', '010130.KS': '고려아연', 
        '010950.KS': 'S-Oil', '096770.KS': 'SK이노베이션', '323410.KS': '카카오뱅크', 
        '377300.KS': '카카오페이', '034730.KS': 'SK', '003550.KS': 'LG',
        '247540.KQ': '에코프로비엠', '086520.KQ': '에코프로', '028300.KQ': 'HLB'
    }

    tickers_massive = tickers_futures.copy()
    additional = {
        '373220.KS': 'LG에너지솔루션', '207940.KS': '삼성바이오로직스', '068270.KS': '셀트리온', 
        '000100.KS': '유한양행', '128940.KS': '한미약품', '196170.KQ': '알테오젠', 
        '214150.KQ': '클래시스', '145020.KQ': '휴젤', '042700.KS': '한미반도체', 
        '403870.KQ': 'HPSP', '071050.KS': '한국금융지주', '024110.KS': '기업은행', 
        '316140.KS': '우리금융지주', '000120.KS': 'CJ대한통운', '028670.KS': '팬오션',
        '010120.KS': 'LS ELECTRIC', '267250.KS': 'HD현대일렉트릭', '012450.KS': '한화에어로스페이스',
        '047810.KS': '한국항공우주', '079550.KS': 'LIG넥스원', '021240.KS': '코웨이', 
        '033780.KS': 'KT&G', '004370.KS': '농심', '007310.KS': '오뚜기', 
        '271560.KS': '오리온', '280360.KS': '롯데웰푸드', '005940.KS': 'NH투자증권', 
        '016360.KS': '삼성증권', '039490.KS': '키움증권', '001450.KS': '현대해상',
        '000150.KS': '두산', '278280.KQ': '천보', '365550.KS': '성일하이텍'
    }
    
    manual_tickers = {**tickers_massive, **additional} if "상위 100" in universe_type else tickers_futures

    fetch_start = (pd.to_datetime(start_date) - timedelta(days=365)).strftime('%Y-%m-%d')
    fetch_end = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    
    tickers_list = list(manual_tickers.keys())
    all_data_list = []
    
    status_placeholder = st.empty()
    status_placeholder.info(f"데이터 불러오는 중 ({len(tickers_list)} 종목)...")
    
    chunk_size = 5
    for i in range(0, len(tickers_list), chunk_size):
        chunk = tickers_list[i:i + chunk_size]
        try:
            df_chunk = yf.download(chunk, start=fetch_start, end=fetch_end, progress=False)['Close']
            if isinstance(df_chunk, pd.Series): df_chunk = df_chunk.to_frame(name=chunk[0])
            all_data_list.append(df_chunk)
            time.sleep(random.uniform(0.05, 0.15))
        except: continue
        
    status_placeholder.empty()
    
    if all_data_list:
        df_final = pd.concat(all_data_list, axis=1)
        df_final = df_final.rename(columns=manual_tickers)
        df_final = df_final.ffill().bfill().dropna(axis=1, how='any')
        return df_final, manual_tickers
    
    return pd.DataFrame(), manual_tickers

# ---------------------------------------------------------
# 4. 분석 로직
# ---------------------------------------------------------
def run_analysis(df_prices, window, entry_thresh, exit_thresh, stop_loss, p_cutoff, mode, start, end):
    pairs = []
    cols = df_prices.columns
    
    if len(df_prices) < window: return pd.DataFrame()

    target_mask = (df_prices.index >= pd.to_datetime(start)) & (df_prices.index <= pd.to_datetime(end))
    
    prog_bar = st.progress(0, text="시장 데이터를 스캔하고 있습니다...")
    checked = 0
    total_checks = len(cols) * (len(cols) - 1) // 2
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            sa, sb = cols[i], cols[j]
            
            corr_val = df_prices[sa].corr(df_prices[sb])
            if corr_val < 0.6: 
                checked += 1
                continue
                
            try:
                score, pval, _ = coint(df_prices[sa], df_prices[sb])
                
                if pval < p_cutoff:
                    log_a, log_b = np.log(df_prices[sa]), np.log(df_prices[sb])
                    spread = log_a - log_b
                    
                    mean = spread.rolling(window).mean()
                    std = spread.rolling(window).std()
                    z_all = (spread - mean) / std
                    z_target = z_all.loc[target_mask]
                    
                    if z_target.empty: continue
                    
                    # Backtest Simulation
                    positions = np.zeros(len(z_target))
                    current_pos = 0 
                    
                    for k in range(len(z_target)):
                        z_val = z_target.iloc[k]
                        if current_pos == 0:
                            if z_val < -entry_thresh: current_pos = 1 
                            elif z_val > entry_thresh: current_pos = -1
                        elif current_pos == 1:
                            if z_val >= -exit_thresh: current_pos = 0 
                            elif z_val < -stop_loss: current_pos = 0  
                        elif current_pos == -1:
                            if z_val <= exit_thresh: current_pos = 0 
                            elif z_val > stop_loss: current_pos = 0   
                        positions[k] = current_pos

                    ret_a = df_prices[sa].loc[target_mask].pct_change().fillna(0)
                    ret_b = df_prices[sb].loc[target_mask].pct_change().fillna(0)
                    spr_ret = (ret_a - ret_b) * pd.Series(positions, index=z_target.index).shift(1).fillna(0).values
                    cum_ret = (1 + spr_ret).cumprod() - 1
                    
                    total_trades = np.abs(np.diff(positions)).sum() / 2
                    sharpe = np.mean(spr_ret) / (np.std(spr_ret) + 1e-9) * np.sqrt(252)
                    
                    curr_z = z_all.iloc[-1]
                    status = "Watch"
                    if curr_z < -entry_thresh: status = "Buy A"
                    elif curr_z > entry_thresh: status = "Buy B"
                    
                    pairs.append({
                        'Stock A': sa, 'Stock B': sb,
                        'P-value': pval, 'Z-Score': curr_z,
                        'Corr': corr_val, 
                        'Status': status, 'Final_Ret': cum_ret[-1],
                        'Sharpe': sharpe, 'Trades': total_trades,
                        'Spread': spread, 'Mean': mean, 'Std': std,
                        'Cum_Ret_Series': cum_ret, 
                        'Daily_Ret_Series': pd.Series(spr_ret, index=z_target.index),
                        'Analysis_Dates': z_target.index,
                        'Price A': df_prices[sa].iloc[-1], 'Price B': df_prices[sb].iloc[-1]
                    })
            except: pass
            
            checked += 1
            if checked % 20 == 0: prog_bar.progress(min(checked / total_checks, 1.0))
            
    prog_bar.empty()
    return pd.DataFrame(pairs)

# ---------------------------------------------------------
# 5. 시각화 (타이틀 추가)
# ---------------------------------------------------------
def plot_chart(row, df_prices, entry, exit, stop, mode):
    sa, sb = row['Stock A'], row['Stock B']
    dates = row['Analysis_Dates']
    
    # [NEW] 한글 종목명 형식의 타이틀 생성
    title_text = f"{sa} vs {sb} 상세 분석"
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25])
    
    # 1. Price
    pa, pb = df_prices[sa].loc[dates], df_prices[sb].loc[dates]
    fig.add_trace(go.Scatter(x=dates, y=(pa/pa.iloc[0]-1)*100, name=sa, line=dict(color='#3B82F6', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=(pb/pb.iloc[0]-1)*100, name=sb, line=dict(color='#F59E0B', width=1.5)), row=1, col=1)
    
    # 2. Z-Score
    z_vals = ((row['Spread'] - row['Mean']) / row['Std']).loc[dates]
    fig.add_trace(go.Scatter(x=dates, y=z_vals, name='Z-Score', line=dict(color='#9CA3AF', width=1)), row=2, col=1)
    fig.add_hline(y=entry, line_dash="dot", line_color="#10B981", row=2, col=1)
    fig.add_hline(y=-entry, line_dash="dot", line_color="#10B981", row=2, col=1)
    fig.add_hline(y=stop, line_color="#EF4444", row=2, col=1)
    fig.add_hline(y=-stop, line_color="#EF4444", row=2, col=1)

    # 3. PnL
    if "백테스트" in mode:
        cum = row['Cum_Ret_Series'] * 100
        fig.add_trace(go.Scatter(x=dates, y=cum, name='수익률 %', line=dict(color='#10B981', width=1.5), fill='tozeroy'), row=3, col=1)

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=20, color="white")), # [NEW] 타이틀 추가
        height=600, hovermode="x unified", template="plotly_dark",
        margin=dict(l=10, r=10, t=50, b=10), showlegend=True,
        plot_bgcolor='#1A1C24', paper_bgcolor='#1A1C24',
        font=dict(family="Pretendard, sans-serif")
    )
    return fig

# [NEW] 히트맵 차트 함수 추가
def plot_heatmap(results):
    if results.empty: return None
    top_pairs = results.sort_values(by='Z-Score', key=abs, ascending=False).head(10)
    data = []
    for idx, row in top_pairs.iterrows():
        data.append({'Pair': f"{row['Stock A']}/{row['Stock B']}", 'Z-Score': row['Z-Score']})
    
    df_heat = pd.DataFrame(data)
    fig = go.Figure(data=go.Heatmap(
        z=df_heat['Z-Score'], x=df_heat['Pair'], y=['괴리율 강도'],
        colorscale='Blues', zmid=0
    ))
    fig.update_layout(
        title=dict(text="상위 페어 괴리율 히트맵", font=dict(size=16, color="white")),
        height=300, template="plotly_dark", 
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# ---------------------------------------------------------
# 6. 메인 실행
# ---------------------------------------------------------
if run_btn:
    with st.spinner("시장 데이터 분석 중..."):
        df_prices, ticker_map = load_stock_data(universe_mode, start_input, end_input)
        name_to_code = {v: k for k, v in ticker_map.items()}

    if df_prices.empty or len(df_prices.columns) < 2:
        st.error("데이터 로드에 실패했습니다. (종목 수 부족)")
    else:
        results = run_analysis(df_prices, window_size, entry_z, exit_z, stop_loss_z, p_cutoff, app_mode, start_input, end_input)
        
        def fmt(name):
            full_code = name_to_code.get(name, 'Unknown')
            clean_code = full_code.split('.')[0]
            return f"{name} ({clean_code})"

        if results.empty:
            st.warning("조건에 맞는 페어가 발견되지 않았습니다.")
            st.caption("Tip: '설정 더보기'에서 P-value를 조금 높여보세요.")
        else:
            if app_mode == "백테스트 (Backtest)":
                st.subheader("📊 포트폴리오 성과 리포트")
                
                all_ret = pd.DataFrame(index=pd.date_range(start=start_input, end=end_input))
                for _, row in results.iterrows():
                    s = row['Daily_Ret_Series']
                    s.index = pd.to_datetime(s.index)
                    s = s[~s.index.duplicated(keep='first')].reindex(all_ret.index).fillna(0)
                    all_ret[f"{row['Stock A']}-{row['Stock B']}"] = s
                
                port_daily = all_ret.mean(axis=1).fillna(0)
                port_cum = (1 + port_daily).cumprod() - 1
                
                total_ret = port_cum.iloc[-1]
                mdd = ((1 + port_daily).cumprod() / (1 + port_daily).cumprod().expanding().max() - 1).min()

                c1, c2, c3 = st.columns(3)
                c1.metric("총 수익률", f"{total_ret*100:.2f}%")
                c2.metric("최대 낙폭 (MDD)", f"{mdd*100:.2f}%")
                c3.metric("매매 페어 수", f"{len(results)}개")
                
                fig_eq = go.Figure()
                fig_eq.add_trace(go.Scatter(x=port_cum.index, y=port_cum*100, mode='lines', name='내 계좌', line=dict(color='#10B981', width=2)))
                fig_eq.update_layout(title="포트폴리오 누적 수익률", template="plotly_dark", height=350, plot_bgcolor='#1A1C24', paper_bgcolor='#1A1C24')
                st.plotly_chart(fig_eq, use_container_width=True)

                st.plotly_chart(plot_heatmap(results), use_container_width=True)

                st.markdown("---")
                st.subheader("🏆 Best Performers (Top 5)")
                for idx, row in results.sort_values('Final_Ret', ascending=False).head(5).iterrows():
                    with st.expander(f"🟢 {fmt(row['Stock A'])} / {fmt(row['Stock B'])} (수익률: {row['Final_Ret']*100:.1f}%)"):
                        st.plotly_chart(plot_chart(row, df_prices, entry_z, exit_z, stop_loss_z, app_mode), use_container_width=True)
                
                st.subheader("Bad Guys (Worst)")
                for idx, row in results.sort_values('Final_Ret', ascending=True).head(3).iterrows():
                    with st.expander(f"🔴 {fmt(row['Stock A'])} / {fmt(row['Stock B'])} (손실: {row['Final_Ret']*100:.1f}%)"):
                        st.plotly_chart(plot_chart(row, df_prices, entry_z, exit_z, stop_loss_z, app_mode), use_container_width=True)

            else:
                actives = results[results['Status'] != 'Watch']
                col1, col2 = st.columns([3, 1])
                with col1: st.subheader("실시간 시장 스캐너")
                with col2: st.metric("매매 신호", f"{len(actives)}건")
                
                tab1, tab2 = st.tabs(["🔥 매매 신호 (Signals)", "👀 전체 관찰 (Watchlist)"])
                
                with tab1:
                    if not actives.empty:
                        for idx, row in actives.sort_values(by='Z-Score', key=abs, ascending=False).iterrows():
                            alloc = total_capital / 2
                            qa = int(alloc / row['Price A'])
                            qb = int(alloc / row['Price B'])
                            sa_fmt, sb_fmt = fmt(row['Stock A']), fmt(row['Stock B'])
                            
                            is_long_a = row['Status'] == "Buy A"
                            
                            with st.expander(f"{sa_fmt} vs {sb_fmt} (괴리율: {row['Z-Score']:.2f}σ)", expanded=True):
                                c1, c2 = st.columns(2)
                                if is_long_a:
                                    c1.success(f"매수: {sa_fmt} ({qa:,}주)")
                                    c2.error(f"매도: {sb_fmt} ({qb:,}주)")
                                else:
                                    c1.error(f"매도: {sa_fmt} ({qa:,}주)")
                                    c2.success(f"매수: {sb_fmt} ({qb:,}주)")
                                st.plotly_chart(plot_chart(row, df_prices, entry_z, exit_z, stop_loss_z, app_mode), use_container_width=True)
                    else:
                        st.info("현재 진입 조건(Z-Score)을 만족하는 종목이 없습니다.")
                
                with tab2:
                    st.plotly_chart(plot_heatmap(results), use_container_width=True)
                    df_disp = results[['Stock A', 'Stock B', 'Z-Score', 'P-value', 'Corr']].copy()
                    df_disp['Stock A'] = df_disp['Stock A'].apply(fmt)
                    df_disp['Stock B'] = df_disp['Stock B'].apply(fmt)
                    df_disp.columns = ['종목 A', '종목 B', '괴리율(Z)', '유의확률(P)', '상관계수']
                    st.dataframe(df_disp.sort_values('괴리율(Z)', key=abs, ascending=False), use_container_width=True)
else:
    st.info("👈 사이드바에서 설정을 확인하고 [분석 시작] 버튼을 눌러주세요.")
