import cv2
import os

# Pilih antara 'baca' atau 'tidur'
target_label = "baca"  # Ganti ke 'tidur' jika ingin ambil data tidur
save_dir = f"dataset_siswa/{target_label}"
os.makedirs(save_dir, exist_ok=True)

cam = cv2.VideoCapture(0)
count = 0

print(f"Ambil gambar untuk: {target_label}. Tekan 's' untuk simpan, 'q' untuk berhenti.")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    cv2.imshow("Ambil Data Webcam", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        filename = os.path.join(save_dir, f"{target_label}_{count}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Gambar disimpan: {filename}")
        count += 1
    elif key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
