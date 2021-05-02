import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#https://gist.github.com/rickiepark/2cd82455e985001542047d7d55d50630
#농어의 데이터
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

# 훈련세트와 테스트세트 나누기
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)

#2차원 배열로 변경하기
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

# 모델 훈련시키기

knr = KNeighborsRegressor(n_neighbors=3)
knr.fit(train_input, train_target)

print(knr.predict([[50]]))

# # 50cm 농어의 이웃을 구하기

# distances, indexes = knr.kneighbors([[50]])

# # 산점도로 확인하기
# plt.scatter(train_input, train_target)
# plt.scatter(train_input[indexes], train_target[indexes], marker='D')
# plt.scatter(50, 1033, marker="^")
# plt.xlabel("length")
# plt.ylabel("weight")
# plt.show()

# # 100cm 농어의 이웃을 구하기
# distances, indexes = knr.kneighbors([[100]])

# # 산점도로 확인하기
# plt.scatter(train_input, train_target)
# plt.scatter(train_input[indexes], train_target[indexes], marker='D')
# plt.scatter(100, 1033, marker="^")
# plt.xlabel("length")
# plt.ylabel("weight")
# plt.show()

#######################################################
# 위에를 실행하면 알겠지만 k-최근접 이웃은 학습한 데이터에 따라 값이 바낌.
#######################################################

#########################################################
##################선형회귀###############################
#########################################################

lr = LinearRegression()

# 선형 회귀 학습
lr.fit(train_input, train_target)

# 50cm농어에 대해 예측
print(lr.predict([[50]]))

## 계산방법 :   농어 무게 는 = 기울기(lr.coef_) x 농어 길이 + 절편(lr.intercept_)
print(lr.coef_, lr.intercept_)

# # 훈련 세트의 산점도 구하기
# plt.scatter(train_input, train_target)
# plt.plot([15, 50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_])
# plt.scatter(50, 1241.8, marker="^")
# plt.xlabel("length")
# plt.ylabel("weight")
# plt.show()

# 훈련세트 테스트세트 점수 확인
print(lr.score(train_input, train_target))
print(lr.score(test_input, test_target))



#곡선  무게 = a x 길이^2 + b x 길이 + c

# 넘파이로 배열 만들기
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))

lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.predict([[50**2, 50]]))
print(lr.coef_, lr.intercept_)
# ㄴ => 무게 = 1.01 x 길이^2 - 21.6 x 길이 + 116.05


point = np.arange(15, 50)
plt.scatter(train_input, train_target)
plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)
plt.scatter(50, 1571, marker="^")
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

print(lr.score(train_poly, train_target))
print(lr.score(test_poly, test_target))