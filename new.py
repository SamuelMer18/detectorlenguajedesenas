import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from tensorflow import keras

DATA_DIR = './data'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img = keras.preprocessing.image.load_img(os.path.join(
            DATA_DIR, dir_, img_path), target_size=(224, 224))
        img_array = keras.preprocessing.image.img_to_array(img)
        data.append(img_array)
        labels.append(dir_)

data = np.asarray(data)
labels = np.asarray(labels)

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Normalize pixel values between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

num_classes = len(np.unique(labels))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.applications.MobileNetV2(
    include_top=True, weights=None, classes=num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=10,
          validation_data=(x_test, y_test))

y_predict = np.argmax(model.predict(x_test), axis=-1)
y_test_labels = np.argmax(y_test, axis=-1)
score = accuracy_score(y_predict, y_test_labels)

print('{}% of samples were classified correctly!'.format(score * 100))

model.save('model.h5')

# Convert the model to TensorFlow.js format
keras.models.save_model(model, 'model_tfjs')
