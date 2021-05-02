import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
###################################
# 다중회귀
# 1개의 특성을 사용하면 직선을 학습하지만
# 2개 이상의 특성을 사용하면 평면을 학습한다.
###################################


# https://raw.githubusercontent.com/rickiepark/hg-mldl/master/perch_full.csv
# https://gist.github.com/rickiepark/2cd82455e985001542047d7d55d50630

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


# 판다스를 이용해서 csv파일을 읽고 넘파이로 바꾸기
df = pd.read_csv("https://raw.githubusercontent.com/rickiepark/hg-mldl/master/perch_full.csv")
perch_full = df.to_numpy()
print(perch_full)

# 훈련세트, 테스트세트 만들기
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)


# poly에서 fit을 해야 transform이 가능하다
poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)  
train_poly = poly.transform(train_input)
print(train_poly.shape)

print(poly.get_feature_names())
test_poly = poly.transform(test_input)


# 다중 회귀 모델 훈련하기

lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly,train_target))
print(lr.score(test_poly,test_target))

# 특성을 더 많이 추가해보기  < 테스트 케이스가 음수가 나옴 >

poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)

lr.fit(train_poly, train_target)
print(lr.score(train_poly,train_target))
print(lr.score(test_poly,test_target))


# 머신러닝 모델이 훈련세트를 과도하게 학습 못하게 규제 하기
# 곱해지는 계수(or기울기)의 크기를 작게만들기
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

# 선형회귀 모델에 규제를 추가한 모델을  릿지, 라쏘 라고 부른다
# 릿지는 계수를 제곱한 값을 기준으로 규제를 적용
# 라쏘는 계수의 절댓값을 기준으로 규제를 적용
# 일반적으로는 릿지를 선호한다.



###########################################
# 릿지 회귀 해보기
###########################################


ridge = Ridge()
ridge.fit(train_scaled,train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))


# 하이퍼 피라미터란 머신러닝 모델이 학습할 수 없고 사람이 알려줘야 하는 피라미터 이다.

train_score = []
test_score = []

# alpha 값을 0.001에서 100까지 10배씩 늘리면서 릿지 회귀 모델을 훈련한 다음 훈련 세트와 테스트 세트의 점수를 저장

alpha_list = [ 0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
  ridge = Ridge(alpha=alpha)
  ridge.fit(train_scaled, train_target)
  train_score.append(ridge.score(train_scaled, train_target))
  test_score.append(ridge.score(test_scaled, test_target))

# 그래프로 보기

# plt.plot(np.log10(alpha_list), train_score)
# plt.plot(np.log10(alpha_list), test_score)
# plt.xlabel('alpha')
# plt.ylabel('R^2')
# plt.show()


# 그래프를 기반으로 alpha값 정하기

ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))

###########################################
# 라쏘 회귀 해보기
###########################################

lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))

# alpha 값을 바꿔가면서 확인해보기

train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
  lasso = Lasso(alpha=alpha, max_iter=10000)
  lasso.fit(train_scaled, train_target)
  train_score.append(lasso.score(train_scaled, train_target))
  test_score.append(lasso.score(test_scaled, test_target))

# 결과를 그래프로 보기

# plt.plot(np.log10(alpha_list), train_score)
# plt.plot(np.log10(alpha_list), test_score)
# plt.xlabel('alpha')
# plt.ylabel('R^2')
# plt.show()
 
lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))