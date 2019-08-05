from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# 1. 데이터
x_data = [[0,0], [1, 0], [0,1], [1, 1]]
y_data = [0,0,0,1]

# 2. 모델
model = LinearSVC()
# └ Deep learning과 달리 미리 정해진 최적화된 모델이 존재하여 훈련 시 최적화된 값이 제공된다.
# └ LinearSVC는 선 1개로 분류할 수 없는 데이터의 경우에는 accuracy가 1이 나오지 않는다



# 3. 실행
model.fit(x_data, y_data)

# 4. 평가 예측
x_test = [[0,0], [1, 0], [0,1], [1, 1]]
y_predict = model.predict(x_test)

print(x_test, "의 예측결과 :", y_predict)
print("acc =", accuracy_score([0,0,0,1], y_predict))
