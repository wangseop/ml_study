###### 데이터 ######
from keras.datasets import cifar10

import numpy as np
(x_train, _), (x_test, _) = cifar10.load_data()
# └비지도학습이므로 y data가 필요하지 않아요


x_train = x_train.astype('float32')/ 255
x_test = x_test.astype('float32')/ 255
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)        # (60000, 784)
print(x_test.shape)         # (10000, 784)

################# 모델 구성 ####################
from keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2DTranspose
from keras.models import Model


# 입력 플레이스 홀더
input_img = Input(shape=(32*32*3,))
# 'encoded'는 입력의 인코딩된 표현
# x = Dense(256, activation='relu')(input_img)
# x = Dense(512, activation='relu')(x)
# x = Dropout(0.2)(x)
# x = Dense(1024, activation='relu')(x)
# x = Dropout(0.2)(x)
# x = Dense(10, activation='relu')(x)
x = Dense(10, activation='relu')(input_img)
x = Dense(20, activation='relu')(x)
x = Dense(23, activation='relu')(x)
x = Dense(40, activation='relu')(x)
x = Dropout(0.15)(x)


# "decoded"는 입력의 손실있는 재구성(lossy reconstruction)
# decoded = Dense(32*32*3, activation='sigmoid')(x)
decoded = Dense(32*32*3, activation='relu')(x)
# └ sigmoid는 255의 픽셀데이터를 0과 1로 판단
#   relu는 255의 픽셀 데이터로 근사치에 매핑(불일치하게 점 찍히는 부분 생긴다.)

# 입력을 입력의 재구성으로 매핑할 모델
autoencoder = Model(input_img, decoded) # 784 -> 32 -> 784

'''
# 이 모델은 입력을 입력의 인코딩된 입력의 표현으로 매핑
encoder = Model(input_img, encoded) # 784 -> 32

# 인코딩된 입력을 위한 플레이스 홀더
decoder_input = Input(shape=(encoding_dim * 2, ))
# 오토인코더 모델의 마지막 레이어 얻기

decoder_layer = Dense(encoding_dim, activation='relu')(decoder_input)
decoder_layer = Dense(784, activation='sigmoid')(decoder_layer)
# ┗ autoencoder의 마지막 레이어 -> Dense(784, activation='sigmoid')(encoded)
# 디코더 모델 생성
decoder = Model(decoder_input, decoder_layer)    # 32 -> 784

encoder.summary()
decoder.summary()
'''
autoencoder.summary()


autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
# autoencoder.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])

history = autoencoder.fit(x_train, x_train, epochs=50, batch_size=256,
                            shuffle=True, validation_data=(x_test, x_test))
# └ fitting 시 x_train data로 x_train, y_train data로 x_train 넣는다
#  └ 같은 data를 

# 숫자들을 인코딩 / 디코딩
# test_set에서 숫자들을 가져왔다는 것을 유의
decoded_imgs = autoencoder.predict(x_test)

'''
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

print(encoded_imgs)
print(decoded_imgs)
print(encoded_imgs.shape)       # (10000, 32)
print(encoded_imgs.shape)       # (10000, 784)
'''

########################### 이미지 출력 ###############################

# Matplotlib 사용
import matplotlib.pyplot as plt

n = 10  # 몇 개의 숫자를 나타낼 것인지
plt.figure(figsize=(20,4))
for i in range(n):
    # 원본 데이터
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(32,32,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # 재구성된 데이터
    ax = plt.subplot(2, n, i+1 + n)
    plt.imshow(decoded_imgs[i].reshape(32,32,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


################### 그래프 출력 ######################
def plot_acc(history, title=None):
    # summarize history for accuracy
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Traning data', 'Validation data'], loc=0)
    # plt.show()

def plot_loss(history, title=None):
    # summarizer history for loss
    if not isinstance(history, dict):
        history = history.history

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    if title is not None:
        plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Traning data', 'Validation data'], loc=0)
    # plt.show()

plot_acc(history, '(a) 학습 경과에 따른 정확도 변화 추이')
plt.show()
plot_loss(history, '(b) 학습 경과에 따른 손실값 변화 추이')
plt.show()

loss, acc = autoencoder.evaluate(x_test, x_test)
print(loss, acc)

