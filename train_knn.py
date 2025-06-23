import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib

dataset_dir = "dataset_siswa"
IMG_SIZE = 64
X, y, labels = [], [], []

# Hanya gunakan label 'baca' dan 'tidur'
allowed_labels = ["baca", "tidur"]

for label_index, class_name in enumerate(allowed_labels):
    labels.append(class_name)
    class_dir = os.path.join(dataset_dir, class_name)
    for file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img.flatten())
            y.append(label_index)

# Latih model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Simpan model dan label
joblib.dump(knn, "knn_model.pkl")
joblib.dump(labels, "labels.pkl")
print("Model KNN berhasil disimpan.")
