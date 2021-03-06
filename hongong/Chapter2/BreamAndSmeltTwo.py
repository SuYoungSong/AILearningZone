import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# http://bit.ly/bream_list     도미의 길이, 무게 데이터
# http://bit.ly/smelt_list     빙어의 길이, 무게 데이터
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# k-최근접 이웃
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

# 2차원 리스트만들기 zip() 함수는 나열된 리스트에서 각각 하나씩 원소를 꺼내 반환
fish_data = [[l,w] for l, w in zip(length, weight)]

fish_target = [1] * 35 + [0] * 14
kn = KNeighborsClassifier()

# 훈련세트를 만들기
train_inpit = fish_data[:35]
train_target = fish_target[:35]

# 테스트 세트 만들기
test_input = fish_data[35:]
test_target = fish_target[35:]

kn = kn.fit(train_inpit, train_target)
print(kn.score(test_input,test_target))

# 넘파이 
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

# 샘플 수, 특성 수 출력
print(input_arr.shape)

np.random.seed(42)            # 시드 설정
index = np.arange(49)         # 0에서 48까지 1씩 증가
np.random.shuffle(index)      # 인덱스를 랜덤하게 섞는다.

# 랜덤 설정한 값으로 샘플, 훈련 케이스 생성

train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]

test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

# # 미리 산점도 확인하기
# plt.scatter(train_input[:,0], train_input[:,1])
# plt.scatter(test_input[:,0], test_input[:,1])
# plt.xlabel('length')
# plt.xlabel('weigth')
# plt.show()

kn = kn.fit(train_input, train_target)
print(kn.score(test_input, test_target))

# 테스트 정답 물어보기
print(kn.predict(test_input))

# 테스트 정답 확인
print(test_target)