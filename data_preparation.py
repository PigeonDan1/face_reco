import os
import cv2
import numpy as np
import math
from sklearn.model_selection import train_test_split

# 加载 Haar 特征分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 指定文件夹路径和注释文件
folder_path = 'Caltech_WebFaces/'
annotation_file = 'WebFaces_GroundThruth.txt'


# 读取图片和相应的人脸位置信息
def load_data(image_folder, annotation_file):
    images = []
    annotations = []
    with open(annotation_file, 'r') as file:
        lines = file.readlines()
        current_image = None
        for line in lines:
            parts = line.strip().split()
            filename = parts[0]
            annotation = list(map(float, parts[1:]))
            if filename != current_image:  # 如果是新的图片
                images.append(cv2.imread(os.path.join(image_folder, filename)))
                annotations.append([annotation])
                current_image = filename
            else:  # 如果是同一张图片
                annotations[-1].append(annotation)
    return images, annotations


# 根据标注信息截取人脸和非人脸
def extract_samples(images, annotations):
    face_samples = []
    non_face_samples = []
    for image, faces in zip(images, annotations):
        # 提取人脸
        for face_annotation in faces:
            face = extract_face(image, face_annotation)
            if face is not None:
                face_samples.append(face)

        # 提取非人脸
        non_faces = extract_non_faces(image, faces)
        non_face_samples.extend(non_faces)

    return face_samples, non_face_samples


# 根据标注信息提取人脸
def extract_face(image, annotation):
    # 提取左眼、右眼、鼻子、嘴巴的坐标
    left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_x, nose_y, mouth_x, mouth_y = annotation
    # 计算人脸矩形框的位置
    x = math.floor(left_eye_x) - 10
    y = math.floor(left_eye_y) - 10 # 随意取整
    w = math.ceil(right_eye_x) + 10
    h = math.ceil(mouth_y) + 10

    # 使用 np.clip() 确保裁剪区域不超出图像边界
    x = np.clip(x, 0, image.shape[1])
    y = np.clip(y, 0, image.shape[0])
    h = np.clip(h, 0, image.shape[0])

    # 截取人脸
    face = image[y:h, x:w]

    # 检查是否成功截取了人脸
    if face.shape[0] > 0 and face.shape[1] > 0:
        return face
    else:
        return None
# 提取非人脸样本
def extract_non_faces(image, faces_annotations, num_samples=1):
    non_face_samples = []
    height, width, _ = image.shape
    for _ in range(num_samples):
        x1 = np.random.randint(0, width)
        y1 = np.random.randint(0, height)
        x2 = np.random.randint(x1, width)
        y2 = np.random.randint(y1, height)
        # 检查非人脸样本是否与任何人脸重叠
        overlap = False
        for face_annotation in faces_annotations:
            left_eye_x, left_eye_y, right_eye_x, right_eye_y, _, _, _, _ = face_annotation
            if x1 < left_eye_x < x2 and y1 < right_eye_y < y2:
                overlap = True
                break
        if not overlap:
            non_face = image[y1:y2, x1:x2]
            non_face_samples.append(non_face)
    return non_face_samples


# 划分训练集和测试集
def split_dataset(face_samples, non_face_samples, test_size=0.1):
    # 划分人脸样本的训练集和测试集
    face_train, face_test = train_test_split(face_samples, test_size=test_size, random_state=42)
    # 划分非人脸样本的训练集和测试集
    non_face_train, non_face_test = train_test_split(non_face_samples, test_size=test_size, random_state=42)
    # 合并人脸和非人脸的训练集和测试集
    X_train = face_train + non_face_train
    X_test = face_test + non_face_test
    # 生成相应的标签
    y_train = [1] * len(face_train) + [0] * len(non_face_train)
    y_test = [1] * len(face_test) + [0] * len(non_face_test)
    return X_train, X_test, y_train, y_test


# 保存数据集到文件夹中
def save_dataset(folder_path, X_train, X_test, y_train, y_test):
    # 创建文件夹
    train_folder = os.path.join(folder_path, 'train')
    test_folder = os.path.join(folder_path, 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    # 保存训练集
    for i, (image, label) in enumerate(zip(X_train, y_train)):
        if not image.any():  # 检查图像是否为空
            continue
        if label == 1:
            cv2.imwrite(os.path.join(train_folder, f'positive_{i}.jpg'), image)
        else:
            cv2.imwrite(os.path.join(train_folder, f'negative_{i}.jpg'), image)
    # 保存测试集
    for i, (image, label) in enumerate(zip(X_test, y_test)):
        if not image.any():  # 检查图像是否为空
            continue
        if label == 1:
            cv2.imwrite(os.path.join(test_folder, f'positive_{i}.jpg'), image)
        else:
            cv2.imwrite(os.path.join(test_folder, f'negative_{i}.jpg'), image)

# 主函数
def main():
    # 读取数据
    images, annotations = load_data(folder_path, annotation_file)
    # 提取人脸和非人脸样本
    face_samples, non_face_samples = extract_samples(images, annotations)
    print(len(face_samples))
    print(len(non_face_samples))
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = split_dataset(face_samples, non_face_samples, test_size=0.1)
    # 保存数据集到文件夹中
    save_dataset(folder_path, X_train, X_test, y_train, y_test)
    print("Datasets created successfully!")


if __name__ == "__main__":
    main()
