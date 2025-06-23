import cv2
import joblib
import numpy as np
from sklearn.metrics import pairwise_distances

# Load model dan data
knn = joblib.load("knn_model.pkl")
labels = joblib.load("labels.pkl")
X_train = joblib.load("X_train.pkl")
IMG_SIZE = 64

# Threshold untuk keputusan
MAX_DISTANCE_THRESHOLD = 6000    # Semakin besar, semakin longgar
MOTION_THRESHOLD = 1000000       # Batas perubahan piksel antar frame
MOTION_FRAME_LIMIT = 30          # Berapa frame diam sebelum dianggap tidak aktif

cam = cv2.VideoCapture(0)
print("Tekan 'q' untuk keluar")

prev_frame = None
diam_frame_count = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)).flatten().reshape(1, -1)

    # Prediksi dari model KNN
    prediction = knn.predict(resized_img)[0]
    label = labels[prediction]
    distances = pairwise_distances(resized_img, X_train)
    min_distance = np.min(distances)

    # Deteksi gerakan
    motion_detected = True
    if prev_frame is not None:
        frame_diff = cv2.absdiff(gray, prev_frame)
        diff_sum = np.sum(frame_diff)

        if diff_sum < MOTION_THRESHOLD:
            diam_frame_count += 1
            if diam_frame_count >= MOTION_FRAME_LIMIT:
                motion_detected = False
        else:
            diam_frame_count = 0
            motion_detected = True

    prev_frame = gray.copy()

    # Tampilkan hasil deteksi
    if min_distance > MAX_DISTANCE_THRESHOLD or not motion_detected:
        text = "AKTIVITAS TIDAK DIKENALI"
        color = (0, 0, 255)  # Merah
    else:
        text = f"Aktivitas: {label.upper()}"
        color = (0, 255, 0)  # Hijau

    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Deteksi Aktivitas Siswa", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
