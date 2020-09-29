import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.utils import np_utils
from keras.utils import to_categorical

np.random.seed(10)


Timesteps = 28
inputsize = 28
batch_size = 50
batch_index = 0
output_size = 10
cell_size = 50
lr = 0.001

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28, 28) /255.
X_test = X_test.reshape(-1, 28, 28)/255.

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

model = Sequential()

model.add(SimpleRNN( batch_input_shape=(None, Timesteps, inputsize), unroll=True, units=cell_size))
model.add(Dense(output_size))
model.add(Activation('softmax'))

adam = Adam(lr)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

for trainingsteps in range(5001):
    x_batch = X_train[batch_index:batch_index+batch_size, :, :]
    y_batch = y_train[batch_index:batch_index+batch_size, :]
    cost = model.train_on_batch(x_batch, y_batch)
    batch_index += batch_size
    batch_index = 0 if batch_index >= X_train.shape[0] else batch_index

    if trainingsteps % 500 == 0:
        cost, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=False)
        print('test cost: {0}, accuracy: {1}'.format(cost, accuracy))

