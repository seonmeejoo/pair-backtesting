# 필수 라이브러리: pip install finance-datareader statsmodels matplotlib seaborn beautifulsoup4 requests

import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt
from itertools import combinations
from datetime import datetime, timedelta
import warnings
import time

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic' 
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")

class NaverPairScanner:
    def __init__(self, start_date=None, lookback_days=365):
        self.lookback_days = lookback_days
        self.start_date = start_date if start_date else (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        self.stock_list = None
        self.price_data = {}
        
        # 파라미터
        self.p_value_threshold = 0.05
        self.z_score_threshold = 2.0

    def get_naver_sectors(self, limit_sectors=10):
        """
        네이버 금융 '업종별 시세'에서 섹터와 구성 종목을 크롤링합니다.
        limit_sectors: 테스트 속도를 위해 상위 N개 업종만 긁어옵니다 (None이면 전체)
        """
        print("📡 네이버 증권에서 섹터 정보를 긁어오는 중... (조금 걸려요!)")
        
        base_url = "https://finance.naver.com/sise/sise_group.naver?type=upjong"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        res = requests.get(base_url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # 1. 업종 목록 가져오기
        table = soup.find('table', {'class': 'type_1'})
        rows = table.find_all('tr')
        
        sector_links = []
        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 2: continue
            
            link_tag = cols[0].find('a')
            if link_tag:
                sector_name = link_tag.text.strip()
                link_url = "https://finance.naver.com" + link_tag['href']
                sector_links.append((sector_name, link_url))
        
        # 제한 설정 (속도 위해)
        if limit_sectors:
            print(f"ℹ️ 속도를 위해 상위 {limit_sectors}개 업종만 조회합니다.")
            sector_links = sector_links[:limit_sectors]

        # 2. 각 업종별 구성 종목 크롤링
        all_stocks = []
        
        for idx, (sec_name, sec_url) in enumerate(sector_links):
            print(f"   [{idx+1}/{len(sector_links)}] {sec_name} 읽는 중...")
            
            res_sec = requests.get(sec_url, headers=headers)
            soup_sec = BeautifulSoup(res_sec.text, 'html.parser')
            
            # 종목 테이블 찾기
            sub_table = soup_sec.find('table', {'class': 'type_5'})
            if not sub_table: continue
            
            sub_rows = sub_table.find_all('tr')
            for s_row in sub_rows:
                s_cols = s_row.find_all('td')
                if len(s_cols) < 2: continue
                
                # 종목명/코드 찾기
                name_tag = s_cols[0].find('a')
                if name_tag:
                    stock_name = name_tag.text.strip()
                    # href에서 code 추출: /item/main.naver?code=005930
                    stock_code = name_tag['href'].split('code=')[-1]
                    
                    all_stocks.append({
                        'Sector': sec_name,
                        'Name': stock_name,
                        'Code': stock_code
                    })
            
            # 네이버 차단 방지용 딜레이
            time.sleep(0.2)
            
        self.stock_list = pd.DataFrame(all_stocks)
        # 중복 제거 (ETF 등이 섞일 수 있음)
        self.stock_list = self.stock_list.drop_duplicates(subset=['Code'])
        
        print(f"✅ 크롤링 완료! {len(self.stock_list['Sector'].unique())}개 섹터, {len(self.stock_list)}개 종목 확보.")
        return self.stock_list

    def fetch_price_and_filter(self, top_n_per_sector=5):
        """
        크롤링한 종목들의 주가를 받고, 시가총액(또는 임의) 상위 N개만 남겨서 데이터프레임 생성
        (네이버 업종페이지 순서는 보통 등락률 순이므로, 여기서는 단순히 앞순서 N개를 자릅니다)
        """
        print("📉 주가 데이터 다운로드 중...")
        
        # 섹터별로 상위 N개만 추림 (너무 많으면 계산 오래 걸림)
        target_df = self.stock_list.groupby('Sector').head(top_n_per_sector)
        codes = target_df['Code'].tolist()
        
        data_dict = {}
        count = 0
        
        for code in codes:
            try:
                df = fdr.DataReader(code, self.start_date)
                if not df.empty:
                    data_dict[code] = df['Close']
            except:
                continue
            count += 1
            if count % 20 == 0:
                print(f"   ... {count}/{len(codes)} 종목 완료")

        self.price_df = pd.DataFrame(data_dict).dropna()
        
        # 데이터가 받아진 종목만 남기기
        self.valid_stocks = target_df[target_df['Code'].isin(self.price_df.columns)]
        print(f"✅ 데이터 확보 완료: {len(self.price_df.columns)}개 종목")

    def find_pairs(self):
        pairs = []
        sectors = self.valid_stocks['Sector'].unique()
        
        print("🔍 섹터 내 페어 분석 중...")
        
        for sector in sectors:
            # 해당 섹터이면서 데이터가 있는 종목들
            sector_codes = self.valid_stocks[self.valid_stocks['Sector'] == sector]['Code'].tolist()
            
            if len(sector_codes) < 2:
                continue
            
            for s1, s2 in combinations(sector_codes, 2):
                series1 = self.price_df[s1]
                series2 = self.price_df[s2]
                
                # 1차 필터: 상관계수 (계산 속도 높이기 위함)
                if series1.corr(series2) < 0.8:
                    continue

                # 공적분 테스트
                score, p_value, _ = coint(series1, series2)
                
                if p_value < self.p_value_threshold:
                    name1 = self.valid_stocks[self.valid_stocks['Code'] == s1]['Name'].values[0]
                    name2 = self.valid_stocks[self.valid_stocks['Code'] == s2]['Name'].values[0]
                    
                    # 헷지 비율 및 스프레드
                    x = sm.add_constant(series2)
                    model = sm.OLS(series1, x).fit()
                    hedge_ratio = model.params[1]
                    
                    spread = series1 - (hedge_ratio * series2)
                    z_score = (spread.iloc[-1] - spread.mean()) / spread.std()
                    
                    pairs.append({
                        'Sector': sector,
                        'Stock1': name1,
                        'Stock2': name2,
                        'P_value': round(p_value, 5),
                        'Current_Z': round(z_score, 2),
                        'Code1': s1,
                        'Code2': s2,
                        'Spread_Series': spread
                    })
        
        self.results = pd.DataFrame(pairs)
        return self.results

    def get_signals(self):
        if self.results.empty:
            return pd.DataFrame(), pd.DataFrame()
            
        # Z-score 절대값이 기준보다 크면 진입 시그널
        signals = self.results[abs(self.results['Current_Z']) >= self.z_score_threshold].copy()
        signals['Action'] = np.where(signals['Current_Z'] > 0, 
                                     f"Short {signals['Stock1']} / Long {signals['Stock2']}", 
                                     f"Long {signals['Stock1']} / Short {signals['Stock2']}")
        
        watchlist = self.results[abs(self.results['Current_Z']) < self.z_score_threshold].sort_values('P_value')
        return signals, watchlist

    def plot_pair(self, pair_info):
        s1, s2 = pair_info['Code1'], pair_info['Code2']
        name1, name2 = pair_info['Stock1'], pair_info['Stock2']
        spread = pair_info['Spread_Series']
        
        p1 = self.price_df[s1] / self.price_df[s1].iloc[0] * 100
        p2 = self.price_df[s2] / self.price_df[s2].iloc[0] * 100
        z_score_series = (spread - spread.mean()) / spread.std()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        
        ax1.plot(p1, label=name1, color='tab:blue')
        ax1.plot(p2, label=name2, color='tab:orange')
        ax1.set_title(f"Price: {name1} vs {name2} [{pair_info['Sector']}]")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(z_score_series, label='Z-Score', color='green')
        ax2.axhline(2, color='r', linestyle='--'); ax2.axhline(-2, color='r', linestyle='--')
        ax2.axhline(0, color='k', alpha=0.5)
        ax2.set_title(f"Spread Z (Current: {pair_info['Current_Z']})")
        
        plt.tight_layout()
        plt.show()

# ==========================================
# 🚀 실행 (RUN ME)
# ==========================================
# 1. 스캐너 생성
scanner = NaverPairScanner(lookback_days=365)

# 2. 네이버에서 섹터 정보 긁어오기
# (limit_sectors=5 : 속도 위해 상위 5개 업종만 함. 전체 다 하려면 None 입력)
scanner.get_naver_sectors(limit_sectors=10) 

# 3. 주가 받고 분석 (섹터당 5개 종목씩만)
scanner.fetch_price_and_filter(top_n_per_sector=5)
scanner.find_pairs()
signals, watchlist = scanner.get_signals()

print("\n" + "="*50)
if not signals.empty:
    print(f"🔥 진입 추천 페어 ({len(signals)}개):")
    print(signals[['Sector', 'Stock1', 'Stock2', 'Current_Z']].to_string(index=False))
    print("\n📊 첫번째 추천 페어 차트:")
    scanner.plot_pair(signals.iloc[0])
else:
    print("🤷 진입 시그널 없음.")

if not watchlist.empty:
    print(f"\n👀 관심 종목 (Spread 대기중):")
    print(watchlist[['Sector', 'Stock1', 'Stock2', 'Current_Z']].head().to_string(index=False))
