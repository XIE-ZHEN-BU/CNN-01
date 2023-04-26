from holoviews.ipython import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from tensorflow.python import keras
from keras.layers import Dense, Flatten, Conv2D
from keras import Model
from keras.datasets import cifar10

import numpy as np
import pandas as pd

#导入SSL
#ssl._create_default_https_context = ssl._create_unverified_context

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 最初 y_train 和 y_test 的形状为 (50000,1) 和 (10000,1)，
# 为简单起见，让我们将它们重新整形为 1D
y_train = y_train.reshape(50000,)
y_test = y_test.reshape(10000,)
display(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#step A
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.utils import to_categorical

import numpy as np
# Load the data
# from tensorflow.keras.datasets import cifar10
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Scale the features
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Convert the labels to one-hot encoded format
#将标签转换为单一热编码格式
# y_train = to_categorical(y_train, num_classes=10)
# y_test = to_categorical(y_test, num_classes=10)
y_train_new = to_categorical(y_train)
y_test_new = to_categorical(y_test)

print("X_train shape",X_train.shape,"X_test shape",X_test.shape)
print("y_train",y_train.shape,"y_test",y_test.shape)
print("y_train_new",y_train_new.shape,"y_test_new",y_test_new.shape)


# Build the model
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
# Display model summary
model.summary()

# Compile the model模型编译
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 模型拟合
model.fit(X_train,
        # 使用重塑的 x_train  # use reshaped x_train
        y_train_new,
        # 使用重塑的 y_train  # use reshaped y_train
        epochs=5,
        batch_size= 10000,
        # use a bigger batch size since we have big training set here
        # 使用更大的 batch size 因为我们这里有很大的训练集
        validation_split = 0.2,
        shuffle = True)

#(1)检查测试集中前10张图像的预测类标签(不是概率);
import matplotlib.pyplot as plt
import numpy as np

predictions = model.predict(X_test)
predictions_labels = np.argmax(predictions[:10], axis=1)
print("Predicted labels for the first 10 test images:")
print(predictions_labels)
class_labels = ['airplane', 'automobile', 'bird','cat','deer','dog', 'frog', 'horse','ship','truck']
for label in predictions_labels:
    print(class_labels[label])

# (2)在测试集上检查模型的损耗和准确性;
loss, accuracy = model.evaluate(X_test, y_test_new)

display(loss, accuracy)

# (3) 定位错误预测并显示至少 12 张预测错误的图像
# （请同时显示它们的实际标签和预测标签）。
predicted_labels = np.argmax(predictions, axis = 1)
predicted_labels

X_test2 = X_test[predicted_labels != y_test]
y_test2 = y_test[predicted_labels != y_test]
y_pred2 = predicted_labels[predicted_labels != y_test]

display(X_test2.shape, y_test2.shape, y_pred2.shape)

figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(16, 12))

# loop over the axes, image pixels, actual and predicted label
# 遍历轴、图像像素、实际和预测标签
for axes, image, actual, pred in zip(axes.ravel(), X_test2[:24], y_test2[:24], y_pred2[:24]):
    axes.matshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])

    axes.set_title(f"Actual: {class_labels[actual]}; \n Predicted: {class_labels[pred]}")
    # 将实际类和预测类设置为子图标题
    # set actual and predicted class as subplot title