import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#https://gist.github.com/rickiepark/b90b9f725890653e69a1a52943671c62      # 도미와 빙어 데이터
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 데이터를 연결 , 튜플 형태로 반환  zip() 함수와 비슷함
fish_data = np.column_stack((fish_length, fish_weight))

# 정답 데이터  1을 39개 , 0을 14개  일열로 연결
fish_target = np.concatenate((np.ones(35),np.zeros(14)))

# 사이킷런으로 훈련세트, 테스트 세트 나누기 stratify를 입력하면 균등하게 나눠준다.
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)

kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
print(kn.score(test_input, test_target))

print(kn.predict([[25, 150]]))
# kn.kneighbors는 가장 가까운 이웃을 찾아준다.
# 이웃간의 거리와 이웃 샘플의 인덱스를 반환
distances, indexes = kn.kneighbors([[25,150]])

plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes,0], train_input[indexes,1], marker='D')
plt.xlabel('length')
plt.ylabel('weigth')
plt.show()

# 평균과 표준편차를 계산  axis 0은 세로 , 1은 가로
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)

# 원본에서 평균을 빼고 표준편차로 나눈다 ( 브로드 캐스팅 )
train_scaled = (train_input - mean) / std
new = ([25, 150] - mean) / std

plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker="^")
plt.xlabel("length")
plt.ylabel("weight")
plt.show()

kn.fit(train_scaled, train_target)
test_scaled = (test_input - mean) / std
print(kn.score(test_scaled, test_target))
