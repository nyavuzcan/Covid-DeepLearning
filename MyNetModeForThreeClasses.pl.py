from idlelib import history

import numpy as np

import keras
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from keras import backend as K
from tensorflow.python.keras.layers import Conv2D, Dropout
import matplotlib.pyplot as plt

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())

    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


dataset=np.load("datasetThreeClasses.npy",allow_pickle=True)




X=[]
y=[]

for i in dataset:
  X.append(i[0])
  y.append(i[1])

X=np.array(X)
y=np.array(y)

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.20, random_state=42)

X_train=X_train.reshape((5159,28,28,1))
X_test=X_test.reshape((1290,28,28,1))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


######

model = keras.Sequential()
model.add(keras.layers.Conv2D(filters=2, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(keras.layers.Conv2D(filters=4, kernel_size=(2, 2), activation='relu', input_shape=(28,28,1)))
model.add(keras.layers.Conv2D(filters=2, kernel_size=(1, 1), activation='relu', input_shape=(28,28,1)))
model.add(keras.layers.Dense(units=16, activation='relu'))
model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(16, activation = 'relu'))
model.add(keras.layers.Dense(3, activation = 'softmax'))


checkpoint = ModelCheckpoint("detect_model_v1", monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')

optimizer = keras.optimizers.Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc',f1_m,precision_m, recall_m])


history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test),callbacks=[])
print(model.summary())
import matplotlib.pyplot as plt


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Mloss')
plt.ylabel('Vloss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

