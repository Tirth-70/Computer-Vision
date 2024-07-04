# libraries
import os
import pickle
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# prepare the dataset
input_path = './Image Classification/Parking slots/data'
categories = ['empty', 'occupied']

data = []
labels = []

for idx, category in enumerate(categories):
    path = os.path.join(input_path, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = imread(img_path)
        image_resized = resize(image, (15, 15))
        data.append(image_resized.flatten())
        labels.append(idx)

data = np.asarray(data)
labels = np.asarray(labels)

# split the dataset
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# train the model
classifier = SVC()
parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.1, 0.001, 0.0001]}
model = GridSearchCV(classifier, parameters)

model.fit(x_train, y_train)

# evaluate the model
best_model = model.best_estimator_

y_pred = best_model.predict(x_test)

print(f"Accuracy: {accuracy_score(y_pred, y_test) * 100}%")

# save the model

pickle.dump(best_model, open('./Image Classification/Parking slots/model.p', 'wb'))