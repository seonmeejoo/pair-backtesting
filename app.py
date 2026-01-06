# 1. 필수 라이브러리 설치 (최초 1회 실행)
# !pip install finance-datareader statsmodels matplotlib seaborn beautifulsoup4 koreanize-matplotlib

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

# 🌟 한글 폰트 자동 설정 (이게 핵심!)
import koreanize_matplotlib 

warnings.filterwarnings("ignore")

class NaverPairScanner:
    def __init__(self, start_date=None, lookback_days=365):
        self.lookback_days = lookback_days
        self.start_date = start_date if start_date else (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        self.stock_list = None
        self.price_data = {}
        self.p_value_threshold = 0.05
        self.z_score_threshold = 2.0

    def get_naver_sectors(self, limit_sectors=10):
        print("📡 네이버 증권에서 섹터 정보를 긁어오는 중...")
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
            for idx, (sec_name, sec_url) in enumerate(sector_links):
                print(f"   [{idx+1}/{len(sector_links)}] {sec_name} 읽는 중...")
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
                time.sleep(0.1) # 차단 방지
                
            self.stock_list = pd.DataFrame(all_stocks).drop_duplicates(subset=['Code'])
            print(f"✅ 크롤링 완료! {len(self.stock_list)}개 종목 확보.")
            return self.stock_list
            
        except Exception as e:
            print(f"❌ 크롤링 에러: {e}")
            return pd.DataFrame()

    def fetch_price_and_filter(self, top_n_per_sector=5):
        print("📉 주가 데이터 다운로드 중...")
        target_df = self.stock_list.groupby('Sector').head(top_n_per_sector)
        codes = target_df['Code'].tolist()
        data_dict = {}
        
        for i, code in enumerate(codes):
            try:
                df = fdr.DataReader(code, self.start_date)
                if not df.empty:
                    data_dict[code] = df['Close']
            except: continue
            if i % 20 == 0: print(f"   ... {i}/{len(codes)} 진행 중")

        self.price_df = pd.DataFrame(data_dict).dropna()
        self.valid_stocks = target_df[target_df['Code'].isin(self.price_df.columns)]
        print(f"✅ 데이터 확보 완료: {len(self.price_df.columns)}개 종목")

    def find_pairs(self):
        pairs = []
        sectors = self.valid_stocks['Sector'].unique()
        print("🔍 섹터 내 페어 분석 중...")
        
        for sector in sectors:
            sector_codes = self.valid_stocks[self.valid_stocks['Sector'] == sector]['Code'].tolist()
            if len(sector_codes) < 2: continue
            
            for s1, s2 in combinations(sector_codes, 2):
                s1_data, s2_data = self.price_df[s1], self.price_df[s2]
                
                # 상관계수 필터 (0.8 미만 스킵)
                if s1_data.corr(s2_data) < 0.8: continue

                score, p_value, _ = coint(s1_data, s2_data)
                if p_value < self.p_value_threshold:
                    name1 = self.valid_stocks[self.valid_stocks['Code'] == s1]['Name'].values[0]
                    name2 = self.valid_stocks[self.valid_stocks['Code'] == s2]['Name'].values[0]
                    
                    x = sm.add_constant(s2_data)
                    model = sm.OLS(s1_data, x).fit()
                    spread = s1_data - (model.params[1] * s2_data)
                    z_score = (spread.iloc[-1] - spread.mean()) / spread.std()
                    
                    pairs.append({
                        'Sector': sector, 'Stock1': name1, 'Stock2': name2,
                        'P_value': round(p_value, 5), 'Current_Z': round(z_score, 2),
                        'Code1': s1, 'Code2': s2, 'Spread_Series': spread
                    })
        
        self.results = pd.DataFrame(pairs)
        return self.results

    def plot_pair(self, pair_info):
        s1, s2 = pair_info['Code1'], pair_info['Code2']
        spread = pair_info['Spread_Series']
        p1 = self.price_df[s1] / self.price_df[s1].iloc[0] * 100
        p2 = self.price_df[s2] / self.price_df[s2].iloc[0] * 100
        z_score = (spread - spread.mean()) / spread.std()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        ax1.plot(p1, label=pair_info['Stock1']); ax1.plot(p2, label=pair_info['Stock2'])
        ax1.set_title(f"가격 추세: {pair_info['Stock1']} vs {pair_info['Stock2']}")
        ax1.legend(); ax1.grid(True, alpha=0.3)
        
        ax2.plot(z_score, color='green', label='Z-Score Spread')
        ax2.axhline(2, c='r', ls='--'); ax2.axhline(-2, c='r', ls='--'); ax2.axhline(0, c='k', alpha=0.5)
        ax2.set_title(f"스프레드 Z-Score (현재: {pair_info['Current_Z']})")
        plt.tight_layout(); plt.show()

# --- 실행 ---
scanner = NaverPairScanner(lookback_days=365)
scanner.get_naver_sectors(limit_sectors=5) # 테스트용 5개
scanner.fetch_price_and_filter(top_n_per_sector=5)
scanner.find_pairs()
results = scanner.results

if not results.empty:
    print(f"\n🔥 총 {len(results)}개 페어 발견!")
    # Z-score 절대값 2.0 이상인 것만 필터링해서 출력
    signals = results[abs(results['Current_Z']) >= 2.0]
    if not signals.empty:
        print(signals[['Sector', 'Stock1', 'Stock2', 'Current_Z']].to_string(index=False))
        scanner.plot_pair(signals.iloc[0])
    else:
        print("현재 진입 시그널(Z > 2.0)은 없습니다.")
else:
    print("발견된 페어가 없습니다.")
