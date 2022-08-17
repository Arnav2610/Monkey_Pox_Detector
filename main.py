import numpy as np
from skimage import io, transform
import os
import random
import pickle

DIRECTORY = 'dataset'
CATEGORIES = ['MonkeyPox','Others']

IMG_SIZE = 100

data = []

for category in CATEGORIES:
  folder = os.path.join(DIRECTORY, category)
  label = CATEGORIES.index(category)
  for img in os.listdir(folder):
    img_path = os.path.join(folder, img)
    img_arr = io.imread(img_path)
    img_arr = transform.resize(img_arr, output_shape=(IMG_SIZE,IMG_SIZE))
    data.append([img_arr, label])

#shuffling the data to make an ideal model
random.shuffle(data)

X = []
y = []

for features, labels in data:
  X.append(features)
  y.append(labels)

X = np.array(X)
print(len(X))
Y = np.array(y)
print(len(y))

pickle.dump(X, open('X.pkl','wb'))
pickle.dump(y, open('y.pkl','wb'))



