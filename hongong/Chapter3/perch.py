import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
#https://gist.github.com/rickiepark/2cd82455e985001542047d7d55d50630

# 농어 데이터
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# # 농어 데이터를 산점도로 만들기
# plt.scatter(perch_length,perch_weight)
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

# 훈련, 테스트 케이스 만들기
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)

# input 값을 2차원 배열로 만들어야하기 때문에 데이터 전처리   reshape(-1 하면 길이가 들어감)
train_input =train_input.reshape(-1, 1)
test_input =test_input.reshape(-1, 1)

# k-최근접 이웃 회귀 모델 훈련시키기
knr = KNeighborsRegressor()
knr.fit(train_input, train_target)

# 결정계수 확인하기
print(knr.score(test_input,test_target)*100)

# 테스트 세트에 대한 예측을 만들기 
test_prediction = knr.predict(test_input)

# 테스트 세트에 대한 평균 절대값 오차를 계산
mae = mean_absolute_error(test_target, test_prediction)
print(mae)

# 훈련 케이스로 결정계수 확인하기 
print(knr.score(train_input, train_target))

# 과소적합 과대적합 문제 해결하기

# 이웃의 개수를 3개로 설정
knr.n_neighbors = 3

# 모델을 다시 훈련
knr.fit(train_input, train_target)
print(knr.score(train_input, train_target))
print(knr.score(test_input, test_target))

