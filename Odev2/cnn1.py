import keras
from keras import layers

model = keras.Sequential()
model.add(keras.Input(shape=(100, 100, 1)))  # 100x100x1 RGB görüntü
model.add(layers.Conv2D(10, 2, strides=1,use_bias=False, activation="relu"))
model.add(layers.Conv2D(2, 2, strides=2,use_bias=False,activation="relu"))
model.add(layers.MaxPooling2D(3))


model.summary()


