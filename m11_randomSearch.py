# gridSearchCV 코드를 RandomizedSearchCV 코드로 변환

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# 붓꽃 데이터 읽어 들이기
iris_data = pd.read_csv("./data/iris2.csv", encoding='utf-8')

# 붓꽃 데이터를 레이블과 입력 데이터로 분리하기
y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength","PetalWidth"]]

# 학습 전용과 테스트 전용 분리하기
# warnings.filterwarnings("ignore")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size = 0.8, shuffle=True)

# 그리드 서치에서 사용할 매개 변수 --- (*1)
parameters = {      # SVM
    "C": [1,10,100,1000], "kernel":["linear", "rbf", "sigmoid"], "gamma":[0.001, 0.0001]
}

# 그리드 서치 --- (*2)
kfold_cv = KFold(n_splits=5, shuffle=True)
clf = RandomizedSearchCV(estimator=SVC(), param_distributions=parameters, cv=kfold_cv,  n_iter=20, n_jobs=1, verbose=1)      # SVM
# ┗ 모델과 모델에 맞는 파라미터들을 수정해주면 된다
#clf.fit(x, y)
clf.fit(x_train, y_train)
# └x,y 로 fit해주게 되면 validation data와 test data가 겹칠 수 있다
#  └ 그러므로 train 데이터를 가지고 fit 시켜주면 train data를 기준으로 validation data를 선정하여 하므로 체계적인 data 구성이 가능해진다
print("최적의 매개 변수 =", clf.best_estimator_)

# 최적의 매개 변수로 평가하기 --- (*3)
y_pred = clf.predict(x_test)
print("최종 정답률 =", accuracy_score(y_test, y_pred))
# last_score = clf.score(x_test, y_test)
# print("최종 정답률 =", last_score)


'''
Fitting 5 folds for each of 20 candidates, totalling 100 fits
[Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
[Parallel(n_jobs=1)]: Done 100 out of 100 | elapsed:    0.2s finished
최적의 매개 변수 = SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.001, kernel='linear',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
최종 정답률 = 1.0
최종 정답률 = 1.0
'''