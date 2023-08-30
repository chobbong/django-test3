import os
import pandas as pd
import pymysql
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM, Dropout, Dense
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import mean_absolute_error

class RealEstateForecast:
     def __init__(self, model_num, temp_folder="forecast_pngs"):
          self.model_num = model_num
          self.conn = None
          self.cursor = None
          self.connect()
          self.temp_folder = temp_folder
          if not os.path.exists(self.temp_folder):
               os.makedirs(self.temp_folder)

     def connect(self):
          self.conn = pymysql.connect(host="khrpa.com", user='joyunseo77', password='WhgkdbsWhdbstj77', charset='utf8', database="joyunseo77")
          self.cursor = self.conn.cursor()

     def is_connected(self):
          return self.conn and self.conn.open

     def search_estate_data(self, keyword):
          search_query = """
               SELECT 시군구, 시군구번지, 계약년월, 면적당보증금, 면적당매매금, 전세가율, lat, lng, timestep, 전세가율90, 전세가율80, 전세가율70, 전세가율60
               FROM kwy2_data
               WHERE 시군구 LIKE %s OR 시군구번지 LIKE %s
        """
          if not self.is_connected():
               self.connect()

          self.cursor.execute(search_query, (f"%{keyword}%", f"%{keyword}%"))
          results = self.cursor.fetchall()

          # 결과를 pandas DataFrame으로 반환
          columns = ["시군구", "시군구번지", "계약년월", "면적당보증금", "면적당매매금", "전세가율", "lat", "lng", "timestep", "전세가율90", "전세가율80", "전세가율70", "전세가율60"]
          self.cleaned_data = pd.DataFrame(results, columns=columns)
          return self.cleaned_data

     def __del__(self):
               # 연결 종료
               if self.is_connected():
                    self.conn.close()

    
     @staticmethod
     def create_dataset(dataset, look_back=3):
          dataX, dataY = [], []
     # dataset이 pandas Series인 경우만 .values를 사용하여 numpy 배열로 변환
          if isinstance(dataset, pd.Series):
               dataset = dataset.values
          for i in range(len(dataset) - look_back - 1):
               a = dataset[i:(i + look_back)]
               dataX.append(a)
               dataY.append(dataset[i + look_back])
          return np.array(dataX), np.array(dataY)

    # def save_model(self, model, city_name):
    #     model_path = os.path.join(self.temp_folder, f"{city_name}_trained_model.h5")
    #     model.save(model_path)
    #     return model_path

    # def load_saved_model(self, city_name):
    #     model_path = os.path.join(self.temp_folder, f"{city_name}_trained_model.h5")
    #     if os.path.exists(model_path):
    #         model = load_model(model_path)
    #         return model
    #     else:
    #         return None
        
     def train_predict_lstm(self, data_series, epochs=50, batch_size=1):
          # 데이터 스케일링
          scaler = MinMaxScaler(feature_range=(0, 1))
          scaled_data_series = scaler.fit_transform(data_series.values.reshape(-1, 1))
          
          # 데이터를 LSTM에 적합한 형태로 변환
          look_back = 3
          trainX, trainY = self.create_dataset(scaled_data_series, look_back)
          trainX = np.reshape(trainX, (trainX.shape[0], look_back, 1))
     
          # LSTM 모델 구성
          model = Sequential()
          model.add(LSTM(100, input_shape=(look_back, 1), return_sequences=True))
          model.add(Dropout(0.2))
          model.add(LSTM(50, return_sequences=True))
          model.add(Dropout(0.2))
          model.add(LSTM(25))
          model.add(Dropout(0.2))
          model.add(Dense(1))
          model.compile(loss='mae', optimizer='adam')

     
          # LSTM 모델 학습
          model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=0)
     
          # 미래 3개월에 대한 예측
          future_predictions = []
          input_data = trainX[-1]  # 마지막 3개월 데이터로 시작
          for _ in range(3):
               prediction = model.predict(input_data.reshape(1, look_back, 1))
               future_predictions.append(prediction[0, 0])
               input_data = np.roll(input_data, -1)
               input_data[-1] = prediction
          
          # 스케일링된 예측값을 원래의 스케일로 변환
          future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
          
          return future_predictions.flatten()


     @staticmethod
     def convert_to_date_format(date_int):
          return f"{str(date_int)[:4]}-{str(date_int)[4:]}"

     def forecast_for_city(self, city_name):
          city_data = self.cleaned_data[self.cleaned_data['시군구번지'] == city_name]
          
          if city_data.empty:
               raise ValueError(f"{city_name}에 해당하는 데이터가 존재하지 않습니다.")
     
          data_city = city_data[['계약년월', '면적당보증금', '면적당매매금']].groupby('계약년월').mean().reset_index()
          data_city['계약년월'] = data_city['계약년월'].apply(self.convert_to_date_format)
          
          forecasted_sales = self.train_predict_lstm(data_city['면적당매매금'])
          forecasted_lease = self.train_predict_lstm(data_city['면적당보증금'])
     
          results = []
          
          for month in range(3):  # 3개월 예측
               if forecasted_lease[month] > forecasted_sales[month]:
                    results.append({
                         "forecast": f"{month+1}개월 후 {city_name}은(는) 역전세가 의심됩니다.",
                         "보증금": forecasted_lease[month],
                         "매매금": forecasted_sales[month]
                    })
               else:
                    results.append({
                         "forecast": f"{month+1}개월 후 {city_name}은(는) 역전세가 아닙니다.",
                         "보증금": forecasted_lease[month],
                         "매매금": forecasted_sales[month]
                    })
          return results
    
     def evaluate_performance(self, true_values, predicted_values):
          # 성능 평가 메트릭 계산
          mse = mean_squared_error(true_values, predicted_values)
          mae = mean_absolute_error(true_values, predicted_values)
          return mse, mae