import cv2
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 加载 Haar 特征分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 读取图片和相应的人脸位置信息
def load_data(image_folder):
    images = []
    labels = []
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg"):
            label = filename.split('_')[0]  # 提取标签，positive 或 negative
            image = cv2.imread(os.path.join(image_folder, filename))
            images.append(image)
            labels.append(1 if label == 'positive' else 0)  # 将 positive 标签设为 1，negative 标签设为 0
    return images, labels

# 提取特征
def extract_features(images):
    features = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # for (x, y, w, h) in faces:
        #     face_roi = gray[y:y + h, x:x + w]
        #     haar_features = cv2.resize(face_roi, (100, 100)).flatten()
        resized_gray = cv2.resize(gray, (100, 100)).flatten()
        features.append(resized_gray)
    return features

# 训练模型
def train_model(features, labels):
    X_train = np.array(features)
    y_train = np.array(labels)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# 测试模型并绘制 ROC 曲线
# 测试模型并绘制 ROC 曲线
def test_model_and_plot_roc(model, test_images, test_labels):
    predicted_probs = []
    true_labels = []
    for image, label in zip(test_images, test_labels):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_gray = cv2.resize(gray, (100, 100)).flatten()
        predicted_prob = model.predict_proba(np.array([resized_gray]))[:, 1]
        predicted_probs.append(predicted_prob)
        true_labels.append(label)

    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
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

# 主函数
def main():
    # 加载训练集和测试集数据
    train_images, train_labels = load_data(image_folder='Caltech_WebFaces/train')
    test_images, test_labels = load_data(image_folder='Caltech_WebFaces/test')

    # 提取训练集和测试集特征
    train_features = extract_features(train_images)
    test_features = extract_features(test_images)

    # 训练模型
    model = train_model(train_features, train_labels)

    # 测试模型并绘制 ROC 曲线
    test_model_and_plot_roc(model, test_images, test_labels)

if __name__ == "__main__":
    main()
