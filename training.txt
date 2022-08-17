import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np


X = pickle.load(open('X.pkl','rb'))
y = pickle.load(open('y.pkl','rb'))

print(len(X))
print(len(y))

X = X/255 #pixel values converted to between 0-1

model = Sequential()

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, input_shape = X.shape[1:], activation = "relu"))

model.add(Dense(2, activation = "softmax"))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

X = np.array(X)
y = np.array(y)

model.fit(X, y, epochs=3, validation_split=0.2)