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
# 1. UI 및 테마 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="Pair Trading Scanner",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #1A1C24; color: #E0E0E0; font-family: 'Pretendard', sans-serif; }
    section[data-testid="stSidebar"] { background-color: #111317; border-right: 1px solid #2B2D35; }
    div[data-testid="metric-container"] { background-color: #252830; border: 1px solid #363945; border-radius: 8px; padding: 15px; }
    div.stButton > button { background-color: #3B82F6; color: white; border: none; border-radius: 6px; height: 3em; font-weight: 600; }
    h1, h2, h3 { color: #F3F4F6 !important; font-weight: 700 !important; }
    
    /* 태그 뱃지 스타일 */
    .tag-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        font-weight: 600;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

DEFAULTS = {
    "window_size": 60,
    "entry_z": 2.0,
    "exit_z": 0.0,
    "stop_loss_z": 4.0,
    "p_cutoff": 0.05
}

# ---------------------------------------------------------
# 2. 그룹핑 정의 (Logic Engine) - 확장판 (V11.5)
# ---------------------------------------------------------
RELATIONSHIP_MAP = [
    # 1. 👨‍👦 지주사 vs 자회사 (Holding Discounts)
    ({'SK', 'SK하이닉스'}, '👨‍👦 SK그룹(지주-반도체)'),
    ({'SK', 'SK이노베이션'}, '👨‍👦 SK그룹(지주-에너지)'),
    ({'SK', 'SK텔레콤'}, '👨‍👦 SK그룹(지주-통신)'),
    ({'LG', 'LG전자'}, '👨‍👦 LG그룹(지주-가전)'),
    ({'LG', 'LG화학'}, '👨‍👦 LG그룹(지주-화학)'),
    ({'POSCO홀딩스', 'POSCO퓨처엠'}, '👨‍👦 포스코(지주-소재)'),
    ({'CJ', 'CJ제일제당'}, '👨‍👦 CJ그룹(지주-식품)'),
    ({'LS', 'LS ELECTRIC'}, '👨‍👦 LS그룹(지주-전력)'),
    ({'삼성물산', '삼성전자'}, '👨‍👦 삼성(사실상 지주)'),
    ({'삼성물산', '삼성생명'}, '👨‍👦 삼성(지배구조)'),
    ({'한화', '한화에어로스페이스'}, '👨‍👦 한화(지주-방산)'),
    ({'한화', '한화솔루션'}, '👨‍👦 한화(지주-태양광)'),
    ({'HD현대', 'HD한국조선해양'}, '👨‍👦 HD현대(지주-조선)'),

    # 2. ⚡ 우선주 vs 본주 (괴리율 차익)
    ({'삼성전자', '삼성전자우'}, '⚡ 본주-우선주'),
    ({'현대차', '현대차2우B'}, '⚡ 본주-우선주'),
    ({'현대차', '현대차우'}, '⚡ 본주-우선주'),
    ({'LG화학', 'LG화학우'}, '⚡ 본주-우선주'),
    ({'LG전자', 'LG전자우'}, '⚡ 본주-우선주'),
    ({'삼성SDI', '삼성SDI우'}, '⚡ 본주-우선주'),
    ({'아모레퍼시픽', '아모레퍼시픽우'}, '⚡ 본주-우선주'),

    # 3. ⚔️ 업종 내 경쟁사 (Rivals)
    # 반도체/IT
    ({'삼성전자', 'SK하이닉스'}, '⚔️ 반도체 투톱'),
    ({'NAVER', '카카오'}, '⚔️ 빅테크 경쟁'),
    ({'삼성SDS', 'SK C&C'}, '⚔️ IT서비스'), # SK C&C는 비상장이므로 SK로 대체 고려 가능하나 여기선 제외
    
    # 자동차/배터리
    ({'현대차', '기아'}, '⚔️ 완성차 형제'),
    ({'현대모비스', '현대위아'}, '⚔️ 자동차 부품'),
    ({'LG에너지솔루션', '삼성SDI'}, '⚔️ 배터리 셀'),
    ({'삼성SDI', 'SK이노베이션'}, '⚔️ 배터리 셀'),
    ({'에코프로비엠', '엘앤에프'}, '⚔️ 양극재(코스닥)'),
    ({'POSCO퓨처엠', '에코프로비엠'}, '⚔️ 양극재(소재)'),

    # 중공업/소재
    ({'HD현대중공업', '삼성중공업'}, '⚔️ 조선 빅3'),
    ({'한화오션', '삼성중공업'}, '⚔️ 조선 빅3'),
    ({'HD현대중공업', '한화오션'}, '⚔️ 조선 빅3'),
    ({'POSCO홀딩스', '현대제철'}, '⚔️ 철강 경쟁'),
    ({'고려아연', '영풍'}, '⚔️ 비철금속(경영권)'),
    ({'S-Oil', 'GS'}, '⚔️ 정유(GS칼텍스)'), 
    
    # 소비재/유통
    ({'아모레퍼시픽', 'LG생활건강'}, '⚔️ 화장품 투톱'),
    ({'이마트', '롯데쇼핑'}, '⚔️ 유통 공룡'),
    ({'하이트진로', '롯데칠성'}, '⚔️ 주류 경쟁'),
    ({'대한항공', '아시아나항공'}, '⚔️ 항공(인수이슈)'),
    ({'하나투어', '모두투어'}, '⚔️ 여행사'),

    # 금융/통신
    ({'KB금융', '신한지주'}, '⚔️ 금융지주 1,2위'),
    ({'하나금융지주', '우리금융지주'}, '⚔️ 금융지주 3,4위'),
    ({'삼성화재', 'DB손해보험'}, '⚔️ 손해보험'),
    ({'미래에셋증권', '한국금융지주'}, '⚔️ 증권사'),
    ({'SK텔레콤', 'KT'}, '⚔️ 통신 1,2위'),
    ({'KT', 'LG유플러스'}, '⚔️ 통신 2,3위'),

    # 게임/엔터
    ({'크래프톤', '엔씨소프트'}, '⚔️ 게임 대장주'),
    ({'넷마블', '엔씨소프트'}, '⚔️ 게임 경쟁'),
    ({'하이브', '에스엠'}, '⚔️ 엔터 대장주'),
    ({'JYP Ent.', '와이지엔터테인먼트'}, '⚔️ 엔터 경쟁'),

    # 4. 🔗 밸류체인 (Supply Chain)
    ({'SK하이닉스', '한미반도체'}, '🔗 HBM 연합'),
    ({'삼성전자', '삼성전기'}, '🔗 IT부품 공급'),
    ({'LG전자', 'LG이노텍'}, '🔗 카메라모듈'),
    ({'현대차', '현대모비스'}, '🔗 완성차-모듈'),
    ({'현대차', '현대글로비스'}, '🔗 완성차-물류'),
    ({'한화에어로스페이스', 'LIG넥스원'}, '🔗 K-방산 수출'),
    ({'한화에어로스페이스', '현대로템'}, '🔗 K-방산 수출')
]

def get_pair_tag(stock_a, stock_b):
    current_set = {stock_a, stock_b}
    for pair_set, tag_name in RELATIONSHIP_MAP:
        if current_set == pair_set:
            return tag_name
    return "📊 통계적 발견" # 리스트에 없지만 통계적으로 잡힌 경우

# ---------------------------------------------------------
# 3. 사이드바
# ---------------------------------------------------------
with st.sidebar:
    st.header("설정 (Settings)")
    if st.button("🔄 설정 초기화"):
        for key, value in DEFAULTS.items():
            st.session_state[key] = value
        st.rerun()

    st.divider()
    universe_mode = st.selectbox("분석 대상 그룹", ["KOSPI 200 (선물/헷지)", "시가총액 상위 100 (Long Only)"])
    app_mode = st.radio("실행 모드", ["실시간 분석 (Live)", "백테스트 (Backtest)"])
    st.divider()
    total_capital = st.number_input("투자 원금 (KRW)", value=10000000, step=1000000, format="%d")
    
    with st.expander("⚙️ 전략 파라미터", expanded=True):
        for key in DEFAULTS:
            if key not in st.session_state: st.session_state[key] = DEFAULTS[key]
        window_size = st.slider("분석 기간 (Window)", 20, 120, key="window_size")
        entry_z = st.slider("진입 기준 (Z-Score)", 1.0, 4.0, key="entry_z")
        exit_z = st.slider("익절 기준 (Z-Score)", -1.0, 1.0, key="exit_z")
        stop_loss_z = st.slider("손절 기준 (Z-Score)", 3.0, 8.0, key="stop_loss_z")
        p_cutoff = st.slider("연관성 기준 (P-value)", 0.01, 0.30, key="p_cutoff")

    st.divider()
    if app_mode == "백테스트 (Backtest)":
        st.subheader("검증 기간")
        c1, c2 = st.columns(2)
        start_input = c1.date_input("시작일", datetime(2025, 1, 1))
        end_input = c2.date_input("종료일", datetime(2025, 12, 31))
        run_label = "백테스트 실행"
    else:
        end_input = datetime.now(); start_input = end_input - timedelta(days=365)
        run_label = "실시간 분석 시작"

    run_btn = st.button(run_label, type="primary", use_container_width=True)

# ---------------------------------------------------------
# 4. 데이터 로딩 (종목 추가됨)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data(universe_type, start_date, end_date):
    # 기본 선물 종목
    tickers_futures = {
        '005930.KS': '삼성전자', '000660.KS': 'SK하이닉스', '005380.KS': '현대차', '000270.KS': '기아',
        '005490.KS': 'POSCO홀딩스', '006400.KS': '삼성SDI', '051910.KS': 'LG화학', '035420.KS': 'NAVER',
        '035720.KS': '카카오', '105560.KS': 'KB금융', '055550.KS': '신한지주', '086790.KS': '하나금융지주',
        '000810.KS': '삼성화재', '032830.KS': '삼성생명', '015760.KS': '한국전력', '034020.KS': 'SK', # SK지주사
        '003550.KS': 'LG', '009150.KS': '삼성전기', '011070.KS': 'LG이노텍', '034220.KS': 'LG디스플레이',
        '012330.KS': '현대모비스', '009540.KS': 'HD한국조선해양', '042660.KS': '한화오션', '010140.KS': '삼성중공업',
        '373220.KS': 'LG에너지솔루션', '247540.KQ': '에코프로비엠', '086520.KQ': '에코프로', '042700.KS': '한미반도체', # 한미반도체 추가
        '005935.KS': '삼성전자우', '005387.KS': '현대차2우B', '051915.KS': 'LG화학우' # 우선주 추가
    }
    
    # 추가 종목들 (생략 없이 주요 종목 포함)
    additional = {
        '207940.KS': '삼성바이오로직스', '068270.KS': '셀트리온', '000100.KS': '유한양행', '128940.KS': '한미약품',
        '316140.KS': '우리금융지주', '000120.KS': 'CJ대한통운', '028670.KS': '팬오션', '010120.KS': 'LS ELECTRIC',
        '021240.KS': '코웨이', '033780.KS': 'KT&G', '004370.KS': '농심', '007310.KS': '오뚜기',
        '097950.KS': 'CJ제일제당', '001040.KS': 'CJ', '003670.KS': 'POSCO퓨처엠', '006260.KS': 'LS'
    }
    
    manual_tickers = {**tickers_futures, **additional} if "상위 100" in universe_type else tickers_futures
    
    fetch_start = (pd.to_datetime(start_date) - timedelta(days=365)).strftime('%Y-%m-%d')
    fetch_end = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    
    try:
        # 지수(^KS11) 포함
        df_all = yf.download(list(manual_tickers.keys()) + ['^KS11'], start=fetch_start, end=fetch_end, progress=False)['Close']
        kospi = df_all['^KS11'].rename('KOSPI')
        stocks = df_all.drop(columns=['^KS11']).rename(columns=manual_tickers)
        stocks = stocks.ffill().bfill().dropna(axis=1, how='any')
        return stocks, kospi, manual_tickers
    except: return pd.DataFrame(), pd.Series(), {}

# ---------------------------------------------------------
# 5. 분석 엔진 (태깅 로직 포함)
# ---------------------------------------------------------
def run_analysis(df_prices, window, entry, exit, stop, p_val, start, end):
    pairs = []
    cols = df_prices.columns
    target_mask = (df_prices.index >= pd.to_datetime(start)) & (df_prices.index <= pd.to_datetime(end))
    prog_bar = st.progress(0, text="종목 간의 통계적 관계를 계산하고 있습니다...")
    checked = 0; total = len(cols) * (len(cols) - 1) // 2
    
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            sa, sb = cols[i], cols[j]
            corr = df_prices[sa].corr(df_prices[sb])
            if corr < 0.6: checked += 1; continue
            try:
                score, pval, _ = coint(df_prices[sa], df_prices[sb])
                if pval < p_val:
                    log_a, log_b = np.log(df_prices[sa]), np.log(df_prices[sb])
                    spread = log_a - log_b
                    mean, std = spread.rolling(window).mean(), spread.rolling(window).std()
                    z_all = (spread - mean) / std; z_target = z_all.loc[target_mask]
                    if z_target.empty: continue
                    
                    positions = np.zeros(len(z_target)); curr_pos = 0
                    for k in range(len(z_target)):
                        z = z_target.iloc[k]
                        if curr_pos == 0:
                            if z < -entry: curr_pos = 1
                            elif z > entry: curr_pos = -1
                        elif curr_pos == 1:
                            if z >= -exit or z < -stop: curr_pos = 0
                        elif curr_pos == -1:
                            if z <= exit or z > stop: curr_pos = 0
                        positions[k] = curr_pos
                    
                    ret_a, ret_b = df_prices[sa].loc[target_mask].pct_change().fillna(0), df_prices[sb].loc[target_mask].pct_change().fillna(0)
                    spr_ret = (ret_a - ret_b) * pd.Series(positions, index=z_target.index).shift(1).fillna(0).values
                    
                    # [NEW] 태그 가져오기
                    tag = get_pair_tag(sa, sb)
                    
                    pairs.append({
                        'Stock A': sa, 'Stock B': sb, 'Tag': tag, # 태그 추가
                        'Z-Score': z_all.iloc[-1], 'Corr': corr, 'P-value': pval,
                        'Final_Ret': (1 + spr_ret).prod() - 1, 'Daily_Ret_Series': pd.Series(spr_ret, index=z_target.index),
                        'Spread': spread, 'Mean': mean, 'Std': std, 'Analysis_Dates': z_target.index,
                        'Price A': df_prices[sa].iloc[-1], 'Price B': df_prices[sb].iloc[-1]
                    })
            except: pass
            checked += 1
            if checked % 50 == 0: prog_bar.progress(min(checked/total, 1.0))
    prog_bar.empty()
    return pd.DataFrame(pairs)

# ---------------------------------------------------------
# 6. 시각화 함수 (태그 표시 추가)
# ---------------------------------------------------------
def plot_pair_analysis(row, df_prices, entry):
    sa, sb = row['Stock A'], row['Stock B']
    dates = row['Analysis_Dates']
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25])
    pa, pb = df_prices[sa].loc[dates], df_prices[sb].loc[dates]
    fig.add_trace(go.Scatter(x=dates, y=(pa/pa.iloc[0])*100, name=sa, line=dict(color='#3B82F6', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=(pb/pb.iloc[0])*100, name=sb, line=dict(color='#F59E0B', width=1.5)), row=1, col=1)
    z_vals = ((row['Spread'] - row['Mean']) / row['Std']).loc[dates]
    fig.add_trace(go.Scatter(x=dates, y=z_vals, name='Z-Score', line=dict(color='#9CA3AF', width=1)), row=2, col=1)
    
    sell_sig = z_vals[z_vals > entry]; buy_sig = z_vals[z_vals < -entry]
    fig.add_trace(go.Scatter(x=sell_sig.index, y=sell_sig, mode='markers', marker=dict(color='#EF4444', size=5), name='매도', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=buy_sig.index, y=buy_sig, mode='markers', marker=dict(color='#3B82F6', size=5), name='매수', showlegend=False), row=2, col=1)
    
    fig.add_hline(y=entry, line_dash="dash", line_color="#EF4444", row=2, col=1)
    fig.add_hline(y=-entry, line_dash="dash", line_color="#3B82F6", row=2, col=1)
    fig.add_hrect(y0=-entry, y1=entry, fillcolor="gray", opacity=0.1, line_width=0, row=2, col=1)
    
    cum = (1 + row['Daily_Ret_Series']).cumprod() * 100 - 100
    fig.add_trace(go.Scatter(x=dates, y=cum, name='수익률 %', line=dict(color='#10B981', width=1.5), fill='tozeroy'), row=3, col=1)
    
    # 제목에 태그 포함
    title_text = f"<b>[{row['Tag']}] {sa} vs {sb}</b>"
    fig.update_layout(title=title_text, height=600, template="plotly_dark", plot_bgcolor='#1A1C24', paper_bgcolor='#1A1C24', margin=dict(t=50, b=10))
    return fig

# ---------------------------------------------------------
# 7. 메인 실행
# ---------------------------------------------------------
if run_btn:
    with st.spinner("시장 데이터 스캔 및 그룹핑 분석 중..."):
        df_prices, df_kospi, ticker_map = load_data(universe_mode, start_input, end_input)
    if df_prices.empty: st.error("데이터 로드 실패")
    else:
        results = run_analysis(df_prices, window_size, entry_z, exit_z, stop_loss_z, p_cutoff, start_input, end_input)
        def fmt(name):
            code = {v: k for k, v in ticker_map.items()}.get(name, '').split('.')[0]
            return f"{name} ({code})"
        
        if results.empty: st.warning("조건에 부합하는 페어가 없습니다.")
        elif app_mode == "백테스트 (Backtest)":
            k_period = df_kospi.loc[start_input:end_input]; k_ret = (k_period / k_period.iloc[0]) - 1
            all_ret = pd.DataFrame(index=k_period.index)
            for _, row in results.iterrows(): all_ret[f"{row['Stock A']}-{row['Stock B']}"] = row['Daily_Ret_Series'].reindex(all_ret.index).fillna(0)
            p_daily = all_ret.mean(axis=1); p_cum = (1 + p_daily).cumprod() - 1
            
            st.subheader("📊 전략 vs 시장(KOSPI) 성과 리포트")
            c1, c2, c3 = st.columns(3)
            s_final, k_final = p_cum.iloc[-1]*100, k_ret.iloc[-1]*100
            c1.metric("내 전략 수익률", f"{s_final:.2f}%", f"{s_final-k_final:.2f}% vs 시장")
            c2.metric("KOSPI 지수 수익률", f"{k_final:.2f}%"); c3.metric("Alpha (초과수익)", f"{s_final-k_final:.2f}%p")
            
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(x=p_cum.index, y=p_cum*100, name='내 전략', line=dict(color='#10B981', width=3)))
            fig_comp.add_trace(go.Scatter(x=k_ret.index, y=k_ret*100, name='시장 지수(KOSPI)', line=dict(color='#9CA3AF', width=2, dash='dot')))
            fig_comp.update_layout(title="누적 수익률 비교 차트", template="plotly_dark", height=400, plot_bgcolor='#1A1C24', paper_bgcolor='#1A1C24')
            st.plotly_chart(fig_comp, use_container_width=True)

            st.divider()
            col_t, col_w = st.columns(2)
            with col_t:
                st.subheader("🏆 베스트 퍼포머 (Top 5)")
                for _, row in results.sort_values('Final_Ret', ascending=False).head(5).iterrows():
                    with st.expander(f"{row['Tag']} | {fmt(row['Stock A'])} / {fmt(row['Stock B'])} ({row['Final_Ret']*100:.1f}%)"):
                        st.plotly_chart(plot_pair_analysis(row, df_prices, entry_z), use_container_width=True)
            with col_w:
                st.subheader("💀 워스트 퍼포머 (Worst 5)")
                for _, row in results.sort_values('Final_Ret', ascending=True).head(5).iterrows():
                    with st.expander(f"{row['Tag']} | {fmt(row['Stock A'])} / {fmt(row['Stock B'])} ({row['Final_Ret']*100:.1f}%)"):
                        st.plotly_chart(plot_pair_analysis(row, df_prices, entry_z), use_container_width=True)
        else:
            st.subheader("🔥 실시간 시장 매매 신호")
            actives = results[results['Z-Score'].abs() >= entry_z]
            col1, col2 = st.columns([3, 1]); col1.markdown(f"**{len(results)}개**의 유효 페어를 감시 중입니다."); col2.metric("진입 신호", f"{len(actives)}건")
            tab1, tab2 = st.tabs(["⚡ 진입 신호 (Signals)", "📡 전체 감시 리스트 (Watchlist)"])
            with tab1:
                if not actives.empty:
                    for _, row in actives.sort_values(by='Z-Score', key=abs, ascending=False).iterrows():
                        with st.expander(f"🎯 [{row['Tag']}] {fmt(row['Stock A'])} / {fmt(row['Stock B'])} (Z: {row['Z-Score']:.2f})", expanded=True):
                            st.plotly_chart(plot_pair_analysis(row, df_prices, entry_z), use_container_width=True)
                else: st.info("현재 진입 조건을 만족하는 종목이 없습니다.")
            with tab2:
                df_v = results[['Tag', 'Stock A', 'Stock B', 'Z-Score', 'Corr', 'Price A', 'Price B']].copy()
                df_v['Stock A'] = df_v['Stock A'].apply(fmt); df_v['Stock B'] = df_v['Stock B'].apply(fmt)
                st.dataframe(df_v.sort_values('Z-Score', key=abs, ascending=False), use_container_width=True)
else: st.info("👈 설정을 확인하고 분석 시작 버튼을 눌러주세요.")
