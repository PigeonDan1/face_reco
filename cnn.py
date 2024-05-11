import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import roc_curve, auc

# 设置图像大小
IMAGE_SIZE = (100, 100)

# 指定数据集路径
data_folder = 'Caltech_WebFaces/train'

# 初始化数据和标签列表
data = []
labels = []

# 遍历数据集文件夹中的所有图像文件
for filename in os.listdir(data_folder):
    if filename.endswith(".jpg"):
        # 加载图像并转换为灰度图像
        image = load_img(os.path.join(data_folder, filename), target_size=IMAGE_SIZE, color_mode='grayscale')
        image = img_to_array(image)

        # 提取标签（positive 或 negative）
        label = 1 if 'positive' in filename else 0

        # 将图像和标签添加到列表中
        data.append(image)
        labels.append(label)

# 将列表转换为 numpy 数组
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# 将数据划分为训练集和验证集
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

# 构建卷积神经网络模型
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(trainX, trainY, epochs=5, batch_size=32, validation_data=(testX, testY))

# 评估模型
_, accuracy = model.evaluate(testX, testY)
print('Accuracy: %.2f' % (accuracy * 100))

# # 将训练好的模型保存到磁盘上
# model.save('D:\python\\face_recognize\\face_detection_model.h5')
#
# # 绘制训练过程中的准确率和损失曲线
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0, 1])
# plt.legend(loc='lower right')
# plt.show()

# 预测测试集
predicted_probabilities = model.predict(testX)

# 计算真阳性率（TPR）和假阳性率（FPR）
fpr, tpr, thresholds = roc_curve(testY, predicted_probabilities)

# 计算曲线下面积（AUC）
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()
