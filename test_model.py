import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model


# 加载模型
model = load_model('face_detection_model.h5')

# 指定测试数据集路径
test_folder_path = 'Caltech_WebFaces/test'

# 初始化真阳性率和假阳性率列表
tpr_list = []
fpr_list = []

# 遍历测试文件夹中的所有图像文件
for filename in os.listdir(test_folder_path):
    if filename.endswith(".jpg"):
        # 加载图像并转换为模型所需的大小（例如：224x224）
        image = load_img(os.path.join(test_folder_path, filename), target_size=(224, 224))
        image = img_to_array(image) / 255.0  # 归一化像素值
        image = np.expand_dims(image, axis=0)  # 添加批次维度

        # 使用模型进行预测
        prediction = model.predict(image)

        # 获取概率
        probability = prediction[0][0]

        # 如果概率大于阈值，则认为检测到人脸，标记为真阳性
        if probability > 0.5:
            tpr_list.append(1)
        else:
            tpr_list.append(0)

        # 假阳性为未检测到人脸的图像数量除以总图像数量
        if probability <= 0.5:
            fpr_list.append(1)
        else:
            fpr_list.append(0)

# 计算 ROC 曲线
fpr, tpr, _ = roc_curve(tpr_list, fpr_list)

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
