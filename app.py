import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import coint
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import FinanceDataReader as fdr
import warnings
import re # 정규표현식용

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. UI Settings
# ---------------------------------------------------------
st.set_page_config(page_title="Pair Trading Scanner", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #1A1C24; color: #E0E0E0; font-family: 'Pretendard', sans-serif; }
    section[data-testid="stSidebar"] { background-color: #111317; border-right: 1px solid #2B2D35; }
    div[data-testid="metric-container"] { background-color: #252830; border: 1px solid #363945; border-radius: 4px; padding: 15px; }
    div.stButton > button { background-color: #374151; color: white; border: 1px solid #4B5563; border-radius: 4px; font-size: 0.8rem; }
    div.stButton > button:hover { background-color: #4B5563; }
    h1, h2, h3 { color: #F3F4F6 !important; font-weight: 700 !important; }
    .tag-badge { background-color: #3B82F6; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem; font-weight: 500; margin-right: 6px; }
</style>
""", unsafe_allow_html=True)

DEFAULTS = { "window_size": 60, "z_threshold": 2.0, "p_cutoff": 0.05 }

# ---------------------------------------------------------
# 2. Logic Engine (Advanced Auto-Grouping)
# ---------------------------------------------------------

# 1. 수동 정의 (자동화가 불가능한 밸류체인/특수관계)
MANUAL_MAP = [
    # 🔗 밸류체인 (Supply Chain) - 수동 유지 필수
    ({'SK하이닉스', '한미반도체'}, '🔗 HBM Value Chain'),
    ({'삼성전자', '삼성전기'}, '🔗 IT Parts Chain'),
    ({'LG전자', 'LG이노텍'}, '🔗 Camera Module Chain'),
    ({'현대차', '현대모비스'}, '🔗 Auto Module Chain'),
    ({'한화에어로스페이스', 'LIG넥스원'}, '🔗 Defense Chain'),
    ({'POSCO홀딩스', 'POSCO퓨처엠'}, '🔗 Battery Material Chain'), # 지주사이자 밸류체인 성격
    
    # 특수 지주사 (이름에 '홀딩스'가 안 들어가는 경우)
    ({'SK', 'SK하이닉스'}, '👨‍👦 SK Group (Parent)'),
    ({'LG', 'LG전자'}, '👨‍👦 LG Group (Parent)'),
    ({'CJ', 'CJ제일제당'}, '👨‍👦 CJ Group (Parent)'),
    ({'LS', 'LS ELECTRIC'}, '👨‍👦 LS Group (Parent)'),
    ({'삼성물산', '삼성전자'}, '👨‍👦 Samsung (Governance)'),
    ({'한화', '한화에어로스페이스'}, '👨‍👦 Hanwha (Parent)'),
    ({'HD현대', 'HD한국조선해양'}, '👨‍👦 HD Hyundai (Parent)')
]

# 2. 지능형 자동 그룹핑 함수
@st.cache_data(ttl=86400)
def fetch_smart_pairs():
    try:
        # KRX 전체 종목 로딩
        df_krx = fdr.StockListing('KRX')
        df_krx = df_krx[df_krx['Marcap'] > 3000_0000_0000] # 시총 3천억 이상만 (잡주 제외)
        
        auto_pairs = []
        target_tickers = {}
        
        # -------------------------------------------------
        # A. ⚡ 우선주 자동 매칭 (Preferred Stock Logic)
        # -------------------------------------------------
        # 이름이 '우', '우B'로 끝나는 종목 찾기
        pref_stocks = df_krx[df_krx['Name'].str.contains('우$|우B$', regex=True)]
        
        for _, pref in pref_stocks.iterrows():
            # 본주 이름 추론 (맨 뒤 '우' 제거)
            base_name = re.sub(r'우B?$', '', pref['Name'])
            
            # 본주가 리스트에 있는지 확인
            base_stock = df_krx[df_krx['Name'] == base_name]
            
            if not base_stock.empty:
                base_code = base_stock.iloc[0]['Code']
                pref_code = pref['Code']
                
                # 티커 저장
                suffix_b = ".KS" if base_stock.iloc[0]['Market'] == 'KOSPI' else ".KQ"
                suffix_p = ".KS" if pref['Market'] == 'KOSPI' else ".KQ"
                
                target_tickers[base_code + suffix_b] = base_name
                target_tickers[pref_code + suffix_p] = pref['Name']
                
                auto_pairs.append(({base_name, pref['Name']}, '⚡ Common-Preferred'))

        # -------------------------------------------------
        # B. 👨‍👦 지주사 자동 매칭 (Holdings Logic)
        # -------------------------------------------------
        # 이름에 '홀딩스', '지주'가 들어가는 종목
        holdings = df_krx[df_krx['Name'].str.contains('홀딩스|지주')]
        
        for _, hold in holdings.iterrows():
            # 그룹명 추출 (예: "DB하이텍" -> "DB", "BNK금융지주" -> "BNK")
            # 간단히 앞 2~3글자 파싱 or '홀딩스' 앞부분
            group_name = hold['Name'].replace('홀딩스', '').replace('지주', '').replace('금융', '').strip()
            
            if len(group_name) < 2: continue # 이름이 너무 짧으면 패스
            
            # 같은 그룹 이름을 가진 자회사 찾기 (지주사 본인 제외)
            subsidiaries = df_krx[
                (df_krx['Name'].str.startswith(group_name)) & 
                (df_krx['Code'] != hold['Code'])
            ]
            
            if not subsidiaries.empty:
                # 가장 시가총액이 큰 자회사 1개만 선택 (핵심 자회사)
                core_sub = subsidiaries.sort_values('Marcap', ascending=False).iloc[0]
                
                # 티커 저장
                h_suffix = ".KS" if hold['Market'] == 'KOSPI' else ".KQ"
                s_suffix = ".KS" if core_sub['Market'] == 'KOSPI' else ".KQ"
                
                target_tickers[hold['Code'] + h_suffix] = hold['Name']
                target_tickers[core_sub['Code'] + s_suffix] = core_sub['Name']
                
                auto_pairs.append(({hold['Name'], core_sub['Name']}, '👨‍👦 Holdings-Core Sub'))

        # -------------------------------------------------
        # C. ⚔️ 업종 내 경쟁사 (Sector Rivals)
        # -------------------------------------------------
        sectors = df_krx['Sector'].dropna().unique()
        
        for sector in sectors:
            # 섹터별 시총 1, 2위
            sector_stocks = df_krx[df_krx['Sector'] == sector].sort_values('Marcap', ascending=False)
            
            if len(sector_stocks) >= 2:
                top1 = sector_stocks.iloc[0]
                top2 = sector_stocks.iloc[1]
                
                # 시총 1조 이상인 경우만 의미있는 경쟁으로 간주
                if top1['Marcap'] > 1e12:
                    t1_suf = ".KS" if top1['Market'] == 'KOSPI' else ".KQ"
                    t2_suf = ".KS" if top2['Market'] == 'KOSPI' else ".KQ"
                    
                    target_tickers[top1['Code'] + t1_suf] = top1['Name']
                    target_tickers[top2['Code'] + t2_suf] = top2['Name']
                    
                    # 태그명에 섹터 이름 포함
                    tag = f"⚔️ Rival ({sector})"
                    auto_pairs.append(({top1['Name'], top2['Name']}, tag))

        return auto_pairs, target_tickers
        
    except Exception as e:
        print(f"Error: {e}")
        return [], {}

# ---------------------------------------------------------
# 3. Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    universe_mode = st.selectbox("Target Universe", ["Auto-Detect (Smart)", "Manual Core List"])
    app_mode = st.radio("Mode", ["Live Analysis", "Backtest"])
    
    st.divider()
    total_capital = st.number_input("Capital (KRW)", value=10000000, step=1000000, format="%d")
    
    with st.expander("Parameters", expanded=True):
        for key in DEFAULTS:
            if key not in st.session_state: st.session_state[key] = DEFAULTS[key]
        window_size = st.slider("Window Size", 20, 120, key="window_size")
        z_threshold = st.slider("Z-Score Threshold", 1.0, 4.0, key="z_threshold")
        p_cutoff = st.slider("Max P-value", 0.01, 0.30, key="p_cutoff")
        
        st.write("") 
        if st.button("Reset Parameters"):
            for key, value in DEFAULTS.items(): st.session_state[key] = value
            st.rerun()

    st.divider()
    if app_mode == "Backtest":
        c1, c2 = st.columns(2)
        start_input = c1.date_input("Start", datetime(2025, 1, 1))
        end_input = c2.date_input("End", datetime(2025, 12, 31))
        run_label = "Run Backtest"
    else:
        end_input = datetime.now(); start_input = end_input - timedelta(days=365)
        run_label = "Run Analysis"

    run_btn = st.button(run_label, type="primary", use_container_width=True)

# ---------------------------------------------------------
# 4. Data Loading
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def load_data(mode, start_date, end_date):
    # 1. 자동 감지 실행
    auto_pairs, auto_tickers = fetch_smart_pairs()
    
    # 2. 수동 정의 리스트 (필수 종목)
    manual_tickers = {
        '005930.KS': '삼성전자', '000660.KS': 'SK하이닉스', '005380.KS': '현대차', '000270.KS': '기아',
        '005490.KS': 'POSCO홀딩스', '006400.KS': '삼성SDI', '051910.KS': 'LG화학', '035420.KS': 'NAVER',
        '035720.KS': '카카오', '105560.KS': 'KB금융', '055550.KS': '신한지주', '034020.KS': 'SK',
        '003550.KS': 'LG', '066570.KS': 'LG전자', '000810.KS': '삼성화재', '032830.KS': '삼성생명',
        '028260.KS': '삼성물산', '000880.KS': '한화', '267260.KS': 'HD현대', '001040.KS': 'CJ',
        '042700.KS': '한미반도체', '009150.KS': '삼성전기', '011070.KS': 'LG이노텍', '012330.KS': '현대모비스',
        '012450.KS': '한화에어로스페이스', '079550.KS': 'LIG넥스원', '003670.KS': 'POSCO퓨처엠'
    }
    
    if "Smart" in mode:
        # 수동 + 자동 병합
        final_tickers = {**manual_tickers, **auto_tickers}
        final_map = MANUAL_MAP + auto_pairs
    else:
        final_tickers = manual_tickers
        final_map = MANUAL_MAP

    fetch_start = (pd.to_datetime(start_date) - timedelta(days=365)).strftime('%Y-%m-%d')
    fetch_end = pd.to_datetime(end_date).strftime('%Y-%m-%d')
    
    try:
        # 속도 제한: 티커가 너무 많으면 상위 100개로 제한
        ticker_keys = list(final_tickers.keys())
        if len(ticker_keys) > 100:
            st.toast(f"Analyzing top 100 pairs out of {len(ticker_keys)} detected.", icon="ℹ️")
            ticker_keys = ticker_keys[:100]
            final_tickers = {k: final_tickers[k] for k in ticker_keys}

        df_all = yf.download(ticker_keys + ['^KS11'], start=fetch_start, end=fetch_end, progress=False)['Close']
        kospi = df_all['^KS11'].rename('KOSPI')
        stocks = df_all.drop(columns=['^KS11']).rename(columns=final_tickers)
        stocks = stocks.ffill().bfill().dropna(axis=1, how='any')
        
        return stocks, kospi, final_tickers, final_map
    except Exception as e:
        return pd.DataFrame(), pd.Series(), {}, []

# ---------------------------------------------------------
# 5. Analysis Engine
# ---------------------------------------------------------
def run_analysis(df_prices, window, threshold, p_val, start, end, relationship_map):
    pairs = []
    cols = df_prices.columns
    target_mask = (df_prices.index >= pd.to_datetime(start)) & (df_prices.index <= pd.to_datetime(end))
    
    prog_bar = st.progress(0, text="Analyzing Correlations...")
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
                            if z < -threshold: curr_pos = 1 
                            elif z > threshold: curr_pos = -1
                        elif curr_pos == 1:
                            if z >= 0 or z < -4.0: curr_pos = 0
                        elif curr_pos == -1:
                            if z <= 0 or z > 4.0: curr_pos = 0
                        positions[k] = curr_pos
                    
                    ret_a, ret_b = df_prices[sa].loc[target_mask].pct_change().fillna(0), df_prices[sb].loc[target_mask].pct_change().fillna(0)
                    spr_ret = (ret_a - ret_b) * pd.Series(positions, index=z_target.index).shift(1).fillna(0).values
                    
                    # [Tagging Logic]
                    tag = "Random"
                    current_set = {sa, sb}
                    for pair_set, tag_name in relationship_map:
                        if current_set == pair_set:
                            tag = tag_name
                            break
                    
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
    
    sell_sig = z_vals[z_vals > threshold]; buy_sig = z_vals[z_vals < -threshold]
    fig.add_trace(go.Scatter(x=sell_sig.index, y=sell_sig, mode='markers', marker=dict(color='#EF4444', size=5), name='Sell', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=buy_sig.index, y=buy_sig, mode='markers', marker=dict(color='#3B82F6', size=5), name='Buy', showlegend=False), row=2, col=1)
    
    fig.add_hline(y=threshold, line_dash="dash", line_color="#EF4444", row=2, col=1)
    fig.add_hline(y=-threshold, line_dash="dash", line_color="#3B82F6", row=2, col=1)
    fig.add_hrect(y0=-threshold, y1=threshold, fillcolor="gray", opacity=0.1, line_width=0, row=2, col=1)
    
    cum = (1 + row['Daily_Ret_Series']).cumprod() * 100 - 100
    fig.add_trace(go.Scatter(x=dates, y=cum, name='Return %', line=dict(color='#10B981', width=1.5), fill='tozeroy'), row=3, col=1)
    
    fig.update_layout(title=f"<b>[{row['Tag']}] {sa} vs {sb}</b>", height=600, template="plotly_dark", plot_bgcolor='#1A1C24', paper_bgcolor='#1A1C24', margin=dict(t=50, b=10))
    return fig

def plot_scatter(results):
    if results.empty: return None
    fig = px.scatter(
        results, x='Corr', y=results['Z-Score'].abs(), color='Tag',
        hover_data=['Stock A', 'Stock B'],
        title='Opportunity Map (by Sector/Logic)', labels={'Corr': 'Correlation', 'y': 'Abs Z-Score'},
        template='plotly_dark'
    )
    fig.add_shape(type="rect", x0=0.8, y0=2.0, x1=1.0, y1=results['Z-Score'].abs().max() + 0.5,
        line=dict(color="#10B981", width=1, dash="dot"), fillcolor="#10B981", opacity=0.1)
    fig.update_layout(height=400, plot_bgcolor='#1A1C24', paper_bgcolor='#1A1C24')
    return fig

# ---------------------------------------------------------
# 7. Main Execution
# ---------------------------------------------------------
if run_btn:
    with st.spinner("KRX Scanning & Smart Grouping..."):
        stocks, kospi, tickers, r_map = load_data(universe_mode, start_input, end_input)
        
    if stocks.empty: st.error("Data Load Failed")
    else:
        results = run_analysis(stocks, window_size, z_threshold, p_cutoff, start_input, end_input, r_map)
        
        def fmt(name):
            code = {v: k for k, v in tickers.items()}.get(name, '').split('.')[0]
            return f"{name} ({code})"
        
        if results.empty: st.warning("No pairs found matching criteria.")
        elif app_mode == "Backtest":
            k_period = kospi.loc[start_input:end_input]; k_ret = (k_period / k_period.iloc[0]) - 1
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
                        st.plotly_chart(plot_pair_analysis(row, stocks, z_threshold), use_container_width=True)
            with col_w:
                st.subheader("Worst Performers")
                for _, row in results.sort_values('Final_Ret', ascending=True).head(5).iterrows():
                    with st.expander(f"{row['Tag']} | {fmt(row['Stock A'])} / {fmt(row['Stock B'])} ({row['Final_Ret']*100:.1f}%)"):
                        st.plotly_chart(plot_pair_analysis(row, stocks, z_threshold), use_container_width=True)
        else:
            st.subheader("Live Trading Signals")
            actives = results[results['Z-Score'].abs() >= z_threshold]
            col1, col2 = st.columns([3, 1]); col1.markdown(f"**{len(results)}** pairs monitored."); col2.metric("Active Signals", f"{len(actives)}")
            tab1, tab2 = st.tabs(["Action Required", "Watchlist"])
            with tab1:
                if not actives.empty:
                    for _, row in actives.sort_values(by='Z-Score', key=abs, ascending=False).iterrows():
                        with st.expander(f"🎯 [{row['Tag']}] {fmt(row['Stock A'])} / {fmt(row['Stock B'])} (Z: {row['Z-Score']:.2f})", expanded=True):
                            st.plotly_chart(plot_pair_analysis(row, stocks, z_threshold), use_container_width=True)
                else: st.info("No signals matching current threshold.")
            with tab2:
                st.plotly_chart(plot_scatter(results), use_container_width=True)
                df_v = results[['Tag', 'Stock A', 'Stock B', 'Z-Score', 'Corr', 'Price A', 'Price B']].copy()
                df_v['Stock A'] = df_v['Stock A'].apply(fmt); df_v['Stock B'] = df_v['Stock B'].apply(fmt)
                st.dataframe(df_v.sort_values('Z-Score', key=abs, ascending=False), use_container_width=True)
else: st.info("Configure settings and click Run.")
