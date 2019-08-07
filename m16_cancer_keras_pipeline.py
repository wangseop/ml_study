# m13_cancer_keras_hyperParameter.py 의 모델에 pipeline적용

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() # 분류

y = cancer['target']
x = cancer['data']

print(x.shape)  # (569, 30)
print(y.shape)  # (569, )

from keras.utils import np_utils
# y = np_utils.to_categorical(y)

from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Dropout, Input

import numpy as np

def build_network(keep_prob=0.5, optimizer='adam'):

    inputs = Input(shape=(30, ), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(keep_prob)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(256, activation='relu', name='hidden4')(x)
    x = Dropout(keep_prob)(x)
    x = Dense(256, activation='relu', name='hidden5')(x)
    x = Dropout(keep_prob)(x)
    prediction = Dense(1, activation='sigmoid', name='output')(x)
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_hyperparameters():
    batches = [10,20,30,40,50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = np.linspace(0.1,0.5, 5)
    epochs = [20,40,60,80,100]
    return {"model__batch_size":batches, "model__optimizer":optimizers, "model__keep_prob":dropout, "model__epochs":epochs}      # Map 형태로 반환

from keras.wrappers.scikit_learn import KerasClassifier


from sklearn.model_selection import train_test_split, RandomizedSearchCV
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, train_size=0.8
)

model = KerasClassifier(build_fn=build_network, verbose=1)
hyperparameters = create_hyperparameters()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

pipeline = Pipeline([('minmax', MinMaxScaler()), ('model', model)])

search = RandomizedSearchCV(estimator=pipeline, param_distributions=hyperparameters,
                            n_iter=10, n_jobs=1, cv=5)

search.fit(x_train, y_train)
print(search.best_params_)
print('score : ', search.score(x_test, y_test))

# y_pred = search.predict(x_test)

'''
score :  0.9385965079591986
'''