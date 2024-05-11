import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 加载 Haar 特征分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 指定测试数据集路径
test_folder_path = 'Caltech_WebFaces/test'

# 初始化真实与预测列表
true_list = []
pred_list = []

# 遍历测试文件夹中的所有图像文件
for filename in os.listdir(test_folder_path):
    if filename.endswith(".jpg"):
        # 加载图像
        image = cv2.imread(os.path.join(test_folder_path, filename))

        # 将图像转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用 Haar 特征分类器检测人脸
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(30, 30))

        # 如果检测到人脸，则为真阳性
        if "positive" in filename:
            true_list.append(1)
        else:
            true_list.append(0)

        if len(faces) == 0:
            pred_list.append(0)
        else:
            pred_list.append(1)

# 计算 ROC 曲线
fpr, tpr, _ = roc_curve(true_list, pred_list)

# 计算曲线下面积（AUC）
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
