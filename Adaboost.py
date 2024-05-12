import numpy as np
import cv2
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import os

class AdaBoostClassifier:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        n_samples = len(X)
        weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            model = DecisionTreeClassifier(max_depth=1)
            model.fit(X, y, sample_weight=weights)
            y_pred = model.predict(X)

            error = np.sum(weights * (y_pred != y))
            alpha = 0.5 * np.log((1 - error) / max(error, 1e-10))

            weights *= np.exp(-alpha * y * y_pred)
            weights /= np.sum(weights)

            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        preds = np.zeros(len(X))
        for alpha, model in zip(self.alphas, self.models):
            preds += alpha * model.predict(X)
        return np.sign(preds)

def extract_features(images):
    features = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_gray = cv2.resize(gray, (50, 50))  # Resize image to reduce feature dimension
        features.append(resized_gray.flatten())
    return np.array(features)

# 读取图片和相应的人脸位置信息
def load_data(image_folder):
    images = []
    labels = []
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg"):
            label = filename.split('_')[0]  # 提取标签，positive 或 negative
            image = cv2.imread(os.path.join(image_folder, filename))
            images.append(image)
            labels.append(1 if label == 'positive' else -1)  # 将 positive 标签设为 1，negative 标签设为 -1
    return images, np.array(labels)  # 将标签转换为numpy数组

# 加载训练集和测试集数据
train_images, train_labels = load_data(image_folder='Caltech_WebFaces/train')
test_images, test_labels = load_data(image_folder='Caltech_WebFaces/test')

# 提取训练集和测试集特征
X_train = extract_features(train_images)
X_test = extract_features(test_images)

# Initialize and train AdaBoost classifier
adaboost = AdaBoostClassifier(n_estimators=50)
adaboost.fit(X_train, train_labels)

# Predict on test set
y_score = adaboost.predict(X_test)

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(test_labels, y_score)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
