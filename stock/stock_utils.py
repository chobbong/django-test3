import warnings
import yfinance as yf
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
from datetime import date
import os
warnings. filterwarnings('ignore')
matplotlib.use('Agg')

yf.pdr_override()
data = pdr.get_data_yahoo('^KS11')

close = data['Close']


# 비교 기준 구간

start_date = '2002-01-04'
end_date = date.today().strftime('%Y-%m-%d') 

# 주가 그래프 그리기
# close[start_date:end_date].plot();
# plot = close[start_date:end_date].plot(figsize=(12,6), grid=True)
# fig = plot.get_figure()
# fig.savefig("sp4_2.png")

class PatternFinder():

    def __init__(self, period=5):
        self.period = period

    def set_stock (self, code: str):
        self.code = code
        self.data = pdr.get_data_yahoo(code)
        self.close = self.data['Close']
        self.data['Change'] = self.close.pct_change()  # 'Change'를 계산하고 데이터프레임에 추가
        self.change = self.data['Change']
        return self.data

    def search(self, start_date, end_date, threshold=0.98):
        base = self.close[start_date:end_date]
        self.base_norm = (base - base.min()) / (base.max()-base.min())
        self.base = base

        print(base)

        window_size = len(base)
        moving_cnt = len(self.data) - window_size - self.period - 1
        cos_sims = self.__cosine_sims(moving_cnt, window_size)

        self.window_size = window_size
        cos_sims = cos_sims [cos_sims > threshold]
        return cos_sims

    def __cosine_sims (self, moving_cnt, window_size):
        def cosine_similarity(x, y):
            return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))

            # 유사도 저장 딕셔너리
        sim_list = []
        for i in range(moving_cnt):
            target = self.close[i:i+window_size]
            # Normalize
            target_norm = (target - target.min()) / (target.max() - target.min())

            # 코사인 유사도 저장
            cos_similarity = cosine_similarity(self.base_norm, target_norm)
            # 코사인 유사도 <i(인덱스), 시계열데이터 함께 저장
            sim_list.append(cos_similarity)

        return pd.Series(sim_list).sort_values(ascending=False)
    
    def plot_pattern(self, idx, period=5, filename=None):
        
            if filename is None:
                # Django 프로젝트의 루트 디렉토리를 기준으로 상대 경로 설정
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                filename = os.path.join(base_dir, "stock/static/stock/pattern.png")
            
            if period != self.period:
                self.period = period

            top = self.close[idx:idx+self.window_size+period]
            top_norm =  (top - top.min()) / (top.max() -top.min())

            plt.plot(self.base_norm.values, label='base')
            plt.plot(top_norm.values, label='target')
            plt.axvline(x=len(self.base_norm)-1, c='r', linestyle='--')
            plt.axvspan(len(self.base_norm.values)-1, len(top_norm.values)-1, facecolor='yellow', alpha=0.3)
            plt.legend()
            plt.savefig(filename)
            plt.close()  # 이 부분은 그래프 리소스를 정리하기 위해 추가됩니다.

            preds = self.change[idx+self.window_size: idx+self.window_size+period]
            print(preds)
            print(f'pred: {preds.mean()*100} %')

    def stat_prediction(self, result, period=5):
        idx_list = list(result.keys())
        mean_list = []
        for idx in idx_list:
            pred = self.change[idx+self.window_size : idx+self.window_size+period]
            mean_list.append(pred.mean())
        return np.array(mean_list)
  


def AA(ticker_name):
    p = PatternFinder()
    p.set_stock(ticker_name)
    result = p.search('2023-03-01',end_date)
    second_row_index = result.index[1]  # 두 번째 행의 인덱스 값을 가져옵니다.
    p.plot_pattern(second_row_index)
  