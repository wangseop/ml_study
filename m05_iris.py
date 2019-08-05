import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("../data/iris.csv", encoding="UTF-8", names=['a','b','c','d','y'])
print(iris_data)
print(iris_data.shape)
print(type(iris_data))

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, "y"]

x = iris_data.loc[:, ["a",'b','c','d']]

# y2 = iris_data.iloc[:,4]
# x2 = iris_data.iloc[:36]

# 학습 전용과 테스트 전용 분리하기
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, train_size=0.8, shuffle=True)
# └ Iris Data 전체적으로 섞어주기 위해서 shuffle=True 주어 섞이게 한다

# 학습하기
# clf = SVC()
# clf = KNeighborsClassifier(n_neighbors=1)
clf = LinearSVC()
# └ 모델 결과가 비슷하게 나온다...

clf.fit(x_train, y_train)

# 평가하기
y_pred = clf.predict(x_test)
print("정답률 :", accuracy_score(y_test, y_pred))       # 0.933 ~ 1.0
