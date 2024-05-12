**Data Preprocessing:**

In this experimental project, I created training and testing sets for facial recognition based on the Caltech dataset. I performed the following steps for data preprocessing:

1. **Creating Positive and Negative Samples:** I distinguished between positive (faces) and negative (non-faces) samples by using the position of facial features. Specifically, I calculated the distance between the left and right eyes, added the height from eyes to mouth, and expanded the region by 5 to 10 pixels to determine the face region.

2. **Splitting into Training and Testing Sets:** I divided the dataset into training and testing sets with a ratio of 9:1.

**Adaboost:**

Adaboost is a cascade classifier composed of multiple weak learners. Each learner acts like a judge in a courtroom, contributing a decision based on its assigned weight. During training, Adaboost emphasizes misclassified samples by adjusting their weights, allowing the classifier to focus on problematic data points. Here's a simplified code snippet for Adaboost:

[Code snippet for Adaboost]

After training, I evaluated Adaboost's performance using ROC curve analysis, yielding an area under the curve (AUC) of 0.82.

**OpenCV-Haar-Adaboost:**

The OpenCV library provides functions for facial detection using Haar features and Adaboost. However, this approach is not suitable for binary classification, as it directly outputs bounding boxes instead of binary labels. Therefore, plotting the ROC curve for this method is not meaningful.

**Logistic Regression:**

Logistic regression is implemented straightforwardly by extracting grayscale features from images, resizing and flattening them, and training the model using labeled data. After training, I plotted the ROC curve, achieving satisfactory results.

**CNN (Convolutional Neural Network):**

I constructed a CNN architecture consisting of convolutional and pooling layers, followed by fully connected layers with ReLU and sigmoid activation functions. Training the CNN for 5 epochs achieved an accuracy of 97%. ROC analysis showed promising results, indicating superior performance compared to other methods.

**Conclusion:**

In conclusion, while the CNN demonstrated the best performance in this experiment, it's essential to consider variations in dataset partitioning and sample selection. The provided Python script allows testing different models easily by replacing the model in the `load_model` function. Despite time constraints, this project significantly improved my algorithmic, coding, and image processing skills. There are still many avenues for exploration, such as using alternative feature extractors and classification algorithms.

