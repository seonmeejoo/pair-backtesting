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

st.markdown("""
<style>
    .stApp { background-color: #1A1C24; color: #E0E0E0; font-family: 'Pretendard', sans-serif; }
    section[data-testid="stSidebar"] { background-color: #111317; border-right: 1px solid #2B2D35; }
    div[data-testid="metric-container"] { background-color: #252830; border: 1px solid #363945; border-radius: 8px; padding: 15px; }
    div.stButton > button { background-color: #374151; color: white; border: 1px solid #4B5563; border-radius: 4px; font-size: 0.8rem; }
    div.stButton > button:hover { background-color: #4B5563; }
    h1, h2, h3 { color: #F3F4F6 !important; font-weight: 700 !important; }
    .tag-badge { background-color: #3B82F6; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; font-weight: 500; margin-right: 6px; }
</style>
""", unsafe_allow_html=True)

# 기본값 정의 (단순화됨)
DEFAULTS = {
    "window_size": 60,
    "z_threshold": 2.0, # entry_z 등을 이것 하나로 통합
    "p_cutoff": 0.05
}

# ---------------------------------------------------------
# 2. Logic Engine (태깅 시스템)
# ---------------------------------------------------------
RELATIONSHIP_MAP = [
    # 1. Parent-Child
    ({'SK', 'SK하이닉스'}, 'SK Group (Parent-Child)'),
    ({'SK', 'SK이노베이션'}, 'SK Group (Parent-Child)'),
    ({'SK', 'SK텔레콤'}, 'SK Group (Parent-Child)'),
    ({'LG', 'LG전자'}, 'LG Group (Parent-Child)'),
    ({'LG', 'LG화학'}, 'LG Group (Parent-Child)'),
    ({'POSCO홀딩스', 'POSCO퓨처엠'}, 'POSCO (Parent-Child)'),
    ({'CJ', 'CJ제일제당'}, 'CJ Group (Parent-Child)'),
    ({'LS', 'LS ELECTRIC'}, 'LS Group (Parent-Child)'),
    ({'삼성물산', '삼성전자'}, 'Samsung (Governance)'),
    ({'삼성물산', '삼성생명'}, 'Samsung (Governance)'),
    ({'한화', '한화에어로스페이스'}, 'Hanwha (Parent-Child)'),
    ({'한화', '한화솔루션'}, 'Hanwha (Parent-Child)'),
    ({'HD현대', 'HD한국조선해양'}, 'HD Hyundai (Parent-Child)'),

    # 2. Preferred-Common
    ({'삼성전자', '삼성전자우'}, 'Common-Preferred'),
    ({'현대차', '현대차2우B'}, 'Common-Preferred'),
    ({'현대차', '현대차우'}, 'Common-Preferred'),
    ({'LG화학', 'LG화학우'}, 'Common-Preferred'),
    ({'LG전자', 'LG전자우'}, 'Common-Preferred'),
    ({'삼성SDI', '삼성SDI우'}, 'Common-Preferred'),
    ({'아모레퍼시픽', '아모레퍼시픽우'}, 'Common-Preferred'),

    # 3. Rivals
    ({'삼성전자', 'SK하이닉스'}, 'Semicon Rivals'),
    ({'NAVER', '카카오'}, 'Tech Rivals'),
    ({'현대차', '기아'}, 'Auto Rivals'),
    ({'현대모비스', '현대위아'}, 'Auto Parts Rivals'),
    ({'LG에너지솔루션', '삼성SDI'}, 'Battery Rivals'),
    ({'삼성SDI', 'SK이노베이션'}, 'Battery Rivals'),
    ({'에코프로비엠', '엘앤에프'}, 'Cathode Rivals'),
    ({'POSCO퓨처엠', '에코프로비엠'}, 'Cathode Rivals'),
    ({'HD현대중공업', '삼성중공업'}, 'Shipbuilding Rivals'),
    ({'한화오션', '삼성중공업'}, 'Shipbuilding Rivals'),
    ({'HD현대중공업', '한화오션'}, 'Shipbuilding Rivals'),
    ({'POSCO홀딩스', '현대제철'}, 'Steel Rivals'),
    ({'고려아연', '영풍'}, 'Metal Rivals'),
    ({'S-Oil', 'GS'}, 'Oil Rivals'), 
    ({'아모레퍼시픽', 'LG생활건강'}, 'Cosmetic Rivals'),
    ({'이마트', '롯데쇼핑'}, 'Retail Rivals'),
    ({'하이트진로', '롯데칠성'}, 'Beverage Rivals'),
    ({'대한항공', '아시아나항공'}, 'Airline Rivals'),
    ({'KB금융', '신한지주'}, 'Bank Rivals'),
    ({'하나금융지주', '우리금융지주'}, 'Bank Rivals'),
    ({'삼성화재', 'DB손해보험'}, 'Insurance Rivals'),
    ({'미래에셋증권', '한국금융지주'}, 'Securities Rivals'),
    ({'SK텔레콤', 'KT'}, 'Telco Rivals'),
    ({'KT', 'LG유플러스'}, 'Telco Rivals'),
    ({'크래프톤', '엔씨소프트'}, 'Game Rivals'),
    ({'넷마블', '엔씨소프트'}, 'Game Rivals'),
    ({'하이브', '에스엠'}, 'Ent. Rivals'),
    ({'JYP Ent.', '와이지엔터테인먼트'}, 'Ent. Rivals'),

    # 4. Supply Chain
    ({'SK하이닉스', '한미반도체'}, 'Value Chain (HBM)'),
    ({'삼성전자', '삼성전기'}, 'Value Chain (IT)'),
    ({'LG전자', 'LG이노텍'}, 'Value Chain (IT)'),
    ({'현대차', '현대모비스'}, 'Value Chain (Auto)'),
    ({'현대차', '현대글로비스'}, 'Value Chain (Logistics)'),
    ({'한화에어로스페이스', 'LIG넥스원'}, 'Defense Peers'),
    ({'한화에어로스페이스', '현대로템'}, 'Defense Peers')
]

def get_pair_tag(stock_a, stock_b):
    current_set = {stock_a, stock_b}
    for pair_set, tag_name in RELATIONSHIP_MAP:
        if current_set == pair_set:
            return tag_name
    return "Random" 

# ---------------------------------------------------------
# 3. Sidebar (Simplified)
# ---------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    
    universe_mode = st.selectbox("Target Universe", ["KOSPI 200 (Futures/Hedge)", "Top 100 (Long Only)"])
    app_mode = st.radio("Mode", ["Live Analysis", "Backtest"])
    
    st.divider()
    
    total_capital = st.number_input("Capital (KRW)", value=10000000, step=1000000, format="%d")
    
    with st.expander("Parameters", expanded=True):
        # Session State Init
        for key in DEFAULTS:
            if key not in st.session_state:
                st.session_state[key] = DEFAULTS[key]

        window_size = st.slider("Window Size", 20, 120, key="window_size")
        
        # Z-Score 하나만 사용
        z_threshold = st.slider("Z-Score Threshold", 1.0, 4.0, key="z_threshold", help="Entry at Threshold, Exit at 0.")
        
        p_cutoff = st.slider("Max P-value", 0.01, 0.30, key="p_cutoff")
        
        st.write("") 
        if st.button("Reset Parameters"):
            for key, value in DEFAULTS.items():
                st.session_state[key] = value
            st.rerun()

    st.divider()
    
    if app_mode == "Backtest":
        st.subheader("Period")
        c1, c2 = st.columns(2)
        start_input = c1.date_input("Start", datetime(2025, 1, 1))
        end_input = c2.date_input("End", datetime(2025, 12, 31))
        run_label = "Run Backtest"
    else:
        end_input = datetime.now()
        start_input = end_input - timedelta(days=365)
        run_label = "Run Analysis"

    run_btn = st.button(run_label, type="primary", use_container_width=True)

# ---------------------------------------------------------
# 4. Data Loading (Extended List)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data(universe_type, start_date, end_date):
    tickers_core = {
        '005930.KS': '삼성전자', '000660.KS': 'SK하이닉스', '005380.KS': '현대차', '000270.KS': '기아',
        '005490.KS': 'POSCO홀딩스', '006400.KS': '삼성SDI', '051910.KS': 'LG화학', '035420.KS': 'NAVER',
        '035720.KS': '카카오', '105560.KS': 'KB금융', '055550.KS': '신한지주', '086790.KS': '하나금융지주',
        '000810.KS': '삼성화재', '005830.KS': 'DB손해보험', '032830.KS': '삼성생명', '015760.KS': '한국전력',
        '034020.KS': '두산에너빌리티', '012330.KS': '현대모비스', '009540.KS': 'HD한국조선해양', '042660.KS': '한화오션',
        '010140.KS': '삼성중공업', '329180.KS': 'HD현대중공업', '011200.KS': 'HMM', '003490.KS': '대한항공',
        '030200.KS': 'KT', '017670.KS': 'SK텔레콤', '032640.KS': 'LG유플러스', '009150.KS': '삼성전기',
        '011070.KS': 'LG이노텍', '018260.KS': '삼성SDS', '259960.KS': '크래프톤', '036570.KS': '엔씨소프트',
        '251270.KS': '넷마블', '090430.KS': '아모레퍼시픽', '051900.KS': 'LG생활건강', '097950.KS': 'CJ제일제당',
        '010130.KS': '고려아연', '004020.KS': '현대제철', '010950.KS': 'S-Oil', '096770.KS': 'SK이노베이션',
        '323410.KS': '카카오뱅크', '377300.KS': '카카오페이', '034730.KS': 'SK', '003550.KS': 'LG',
        '028260.KS': '삼성물산', '000880.KS': '한화', '267260.KS': 'HD현대', '001040.KS': 'CJ'
    }
    tickers_growth = {
        '247540.KQ': '에코프로비엠', '086520.KQ': '에코프로', '066970.KQ': '엘앤에프', '028300.KQ': 'HLB',
        '293490.KQ': '카카오게임즈', '035900.KQ': 'JYP Ent.', '041510.KQ': '에스엠', '122870.KQ': '와이지엔터테인먼트',
        '352820.KS': '하이브', '042700.KS': '한미반도체', '028300.KQ': 'HLB'
    }
    tickers_pref = {
        '005935.KS': '삼성전자우', '005387.KS': '현대차2우B', '005385.KS': '현대차우',
        '051915.KS': 'LG화학우', '066575.KS': 'LG전자우', '006405.KS': '삼성SDI우',
        '090435.KS': '아모레퍼시픽우'
    }
    tickers_value = {
        '373220.KS': 'LG에너지솔루션', '207940.KS': '삼성바이오로직스', '068270.KS': '셀트리온',
        '000100.KS': '유한양행', '128940.KS': '한미약품', '316140.KS': '우리금융지주',
        '000120.KS': 'CJ대한통운', '028670.KS': '팬오션', '010120.KS': 'LS ELECTRIC',
        '021240.KS': '코웨이', '033780.KS': 'KT&G', '004370.KS': '농심', '007310.KS': '오뚜기',
        '003670.KS': 'POSCO퓨처엠', '006260.KS': 'LS', '012450.KS': '한화에어로스페이스',
        '047810.KS': '한국항공우주', '079550.KS': 'LIG넥스원', '064350.KS': '현대로템',
        '086280.KS': '현대글로비스', '011210.KS': '현대위아', '139480.KS': '이마트', '023530.KS': '롯데쇼핑',
        '000080.KS': '하이트진로', '005300.KS': '롯데칠성', '007890.KS': '한국금융지주', '006800.KS': '미래에셋증권',
        '039490.KS': '키움증권', '034220.KS': 'LG디스플레이', '066570.KS': 'LG전자', '000150.KS': '두산'
    }
    
    full_tickers = {**tickers_core, **tickers_growth, **tickers_pref, **tickers_value}
    manual_tickers = full_tickers if "Top 100" in universe_type else {**tickers_core, **tickers_growth}

    fetch_start = (pd.to_datetime(start_date) - timedelta(days=365)).strftime('%Y-%m-%d')
    fetch_end = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    
    try:
        df_all = yf.download(list(manual_tickers.keys()) + ['^KS11'], start=fetch_start, end=fetch_end, progress=False)['Close']
        kospi = df_all['^KS11'].rename('KOSPI')
        stocks = df_all.drop(columns=['^KS11']).rename(columns=manual_tickers)
        stocks = stocks.ffill().bfill().dropna(axis=1, how='any')
        return stocks, kospi, manual_tickers
    except:
        return pd.DataFrame(), pd.Series(), {}

# ---------------------------------------------------------
# 5. Analysis Engine (Fix: Correct Parameter Mapping)
# ---------------------------------------------------------
# [중요 수정] 이제 threshold 변수 하나만 받아서 내부에서 처리합니다.
def run_analysis(df_prices, window, threshold, p_val, start, end):
    pairs = []
    cols = df_prices.columns
    target_mask = (df_prices.index >= pd.to_datetime(start)) & (df_prices.index <= pd.to_datetime(end))
    
    prog_bar = st.progress(0, text="Scanning Market Data...")
    checked = 0; total = len(cols) * (len(cols) - 1) // 2
    
    # Implicit Settings based on single threshold
    entry_z = threshold
    exit_z = 0.0        # Mean reversion
    stop_loss_z = 4.0   # Hard stop
    
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
                    z_all = (spread - mean) / std
                    z_target = z_all.loc[target_mask]
                    if z_target.empty: continue
                    
                    # Backtest Logic
                    positions = np.zeros(len(z_target)); curr_pos = 0
                    for k in range(len(z_target)):
                        z = z_target.iloc[k]
                        if curr_pos == 0:
                            if z < -entry_z: curr_pos = 1 
                            elif z > entry_z: curr_pos = -1
                        elif curr_pos == 1:
                            if z >= exit_z or z < -stop_loss_z: curr_pos = 0
                        elif curr_pos == -1:
                            if z <= exit_z or z > stop_loss_z: curr_pos = 0
                        positions[k] = curr_pos
                    
                    ret_a, ret_b = df_prices[sa].loc[target_mask].pct_change().fillna(0), df_prices[sb].loc[target_mask].pct_change().fillna(0)
                    spr_ret = (ret_a - ret_b) * pd.Series(positions, index=z_target.index).shift(1).fillna(0).values
                    tag = get_pair_tag(sa, sb)
                    
                    pairs.append({
                        'Stock A': sa, 'Stock B': sb, 'Tag': tag,
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
# 6. Visualization
# ---------------------------------------------------------
def plot_pair_analysis(row, df_prices, threshold):
    sa, sb = row['Stock A'], row['Stock B']
    dates = row['Analysis_Dates']
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25])
    
    pa, pb = df_prices[sa].loc[dates], df_prices[sb].loc[dates]
    fig.add_trace(go.Scatter(x=dates, y=(pa/pa.iloc[0])*100, name=sa, line=dict(color='#3B82F6', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=(pb/pb.iloc[0])*100, name=sb, line=dict(color='#F59E0B', width=1.5)), row=1, col=1)
    
    z_vals = ((row['Spread'] - row['Mean']) / row['Std']).loc[dates]
    fig.add_trace(go.Scatter(x=dates, y=z_vals, name='Z-Score', line=dict(color='#9CA3AF', width=1)), row=2, col=1)
    
    # Markers
    sell_sig = z_vals[z_vals > threshold]; buy_sig = z_vals[z_vals < -threshold]
    fig.add_trace(go.Scatter(x=sell_sig.index, y=sell_sig, mode='markers', marker=dict(color='#EF4444', size=5), name='Sell', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=buy_sig.index, y=buy_sig, mode='markers', marker=dict(color='#3B82F6', size=5), name='Buy', showlegend=False), row=2, col=1)
    
    fig.add_hline(y=threshold, line_dash="dash", line_color="#EF4444", row=2, col=1)
    fig.add_hline(y=-threshold, line_dash="dash", line_color="#3B82F6", row=2, col=1)
    fig.add_hrect(y0=-threshold, y1=threshold, fillcolor="gray", opacity=0.1, line_width=0, row=2, col=1)
    
    cum = (1 + row['Daily_Ret_Series']).cumprod() * 100 - 100
    fig.add_trace(go.Scatter(x=dates, y=cum, name='Return %', line=dict(color='#10B981', width=1.5), fill='tozeroy'), row=3, col=1)
    
    title_text = f"<b>[{row['Tag']}] {sa} vs {sb}</b>"
    fig.update_layout(title=title_text, height=600, template="plotly_dark", plot_bgcolor='#1A1C24', paper_bgcolor='#1A1C24', margin=dict(t=50, b=10))
    return fig

def plot_scatter(results):
    if results.empty: return None
    fig = px.scatter(
        results, x='Corr', y=results['Z-Score'].abs(), color='P-value',
        hover_data=['Stock A', 'Stock B', 'Tag'],
        title='Opportunity Map', labels={'Corr': 'Correlation', 'y': 'Abs Z-Score'},
        color_continuous_scale='Blues_r', template='plotly_dark'
    )
    fig.add_shape(type="rect", x0=0.8, y0=2.0, x1=1.0, y1=results['Z-Score'].abs().max() + 0.5,
        line=dict(color="#10B981", width=1, dash="dot"), fillcolor="#10B981", opacity=0.1)
    fig.update_layout(height=400, plot_bgcolor='#1A1C24', paper_bgcolor='#1A1C24')
    return fig

# ---------------------------------------------------------
# 7. Main Execution
# ---------------------------------------------------------
if run_btn:
    with st.spinner("Processing Market Data..."):
        df_prices, df_kospi, ticker_map = load_data(universe_mode, start_input, end_input)
    if df_prices.empty: st.error("Data Load Failed")
    else:
        # [FIX] Pass single threshold parameter
        results = run_analysis(df_prices, window_size, z_threshold, p_cutoff, start_input, end_input)
        
        def fmt(name):
            code = {v: k for k, v in ticker_map.items()}.get(name, '').split('.')[0]
            return f"{name} ({code})"
        
        if results.empty: st.warning("No pairs found matching criteria.")
        elif app_mode == "Backtest":
            k_period = df_kospi.loc[start_input:end_input]; k_ret = (k_period / k_period.iloc[0]) - 1
            all_ret = pd.DataFrame(index=k_period.index)
            for _, row in results.iterrows(): all_ret[f"{row['Stock A']}-{row['Stock B']}"] = row['Daily_Ret_Series'].reindex(all_ret.index).fillna(0)
            p_daily = all_ret.mean(axis=1); p_cum = (1 + p_daily).cumprod() - 1
            
            st.subheader("Performance Report (vs KOSPI)")
            c1, c2, c3 = st.columns(3)
            s_final, k_final = p_cum.iloc[-1]*100, k_ret.iloc[-1]*100
            c1.metric("Strategy Return", f"{s_final:.2f}%", f"{s_final-k_final:.2f}% vs Market")
            c2.metric("KOSPI Return", f"{k_final:.2f}%"); c3.metric("Alpha", f"{s_final-k_final:.2f}%p")
            
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(x=p_cum.index, y=p_cum*100, name='Strategy', line=dict(color='#10B981', width=3)))
            fig_comp.add_trace(go.Scatter(x=k_ret.index, y=k_ret*100, name='KOSPI', line=dict(color='#9CA3AF', width=2, dash='dot')))
            fig_comp.update_layout(title="Cumulative Return Comparison", template="plotly_dark", height=400, plot_bgcolor='#1A1C24', paper_bgcolor='#1A1C24')
            st.plotly_chart(fig_comp, use_container_width=True)
            
            st.plotly_chart(plot_scatter(results), use_container_width=True)

            st.divider()
            col_t, col_w = st.columns(2)
            with col_t:
                st.subheader("Top Performers")
                for _, row in results.sort_values('Final_Ret', ascending=False).head(5).iterrows():
                    with st.expander(f"{row['Tag']} | {fmt(row['Stock A'])} / {fmt(row['Stock B'])} ({row['Final_Ret']*100:.1f}%)"):
                        st.plotly_chart(plot_pair_analysis(row, df_prices, z_threshold), use_container_width=True)
            with col_w:
                st.subheader("Worst Performers")
                for _, row in results.sort_values('Final_Ret', ascending=True).head(5).iterrows():
                    with st.expander(f"{row['Tag']} | {fmt(row['Stock A'])} / {fmt(row['Stock B'])} ({row['Final_Ret']*100:.1f}%)"):
                        st.plotly_chart(plot_pair_analysis(row, df_prices, z_threshold), use_container_width=True)
        else:
            st.subheader("Live Trading Signals")
            actives = results[results['Z-Score'].abs() >= z_threshold]
            col1, col2 = st.columns([3, 1]); col1.markdown(f"**{len(results)}** pairs monitored."); col2.metric("Active Signals", f"{len(actives)}")
            tab1, tab2 = st.tabs(["Action Required", "Watchlist"])
            with tab1:
                if not actives.empty:
                    for _, row in actives.sort_values(by='Z-Score', key=abs, ascending=False).iterrows():
                        with st.expander(f"🎯 [{row['Tag']}] {fmt(row['Stock A'])} / {fmt(row['Stock B'])} (Z: {row['Z-Score']:.2f})", expanded=True):
                            st.plotly_chart(plot_pair_analysis(row, df_prices, z_threshold), use_container_width=True)
                else: st.info("No signals matching current threshold.")
            with tab2:
                st.plotly_chart(plot_scatter(results), use_container_width=True)
                df_v = results[['Tag', 'Stock A', 'Stock B', 'Z-Score', 'Corr', 'Price A', 'Price B']].copy()
                df_v['Stock A'] = df_v['Stock A'].apply(fmt); df_v['Stock B'] = df_v['Stock B'].apply(fmt)
                st.dataframe(df_v.sort_values('Z-Score', key=abs, ascending=False), use_container_width=True)
else: st.info("Ready. Configure settings and click Run.")
