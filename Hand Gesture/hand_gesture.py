import cv2
import mediapipe as mp
import logging
import os

#Hide peringatan MediaPipe dan TensorFlow
logging.getLogger('mediapipe').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Inisialisasi MediaPipe Hand dan Drawing Utility
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#Konfigurasi deteksi tangan
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# =Fungsi mengenali gesture
def recognize_gesture(hand_landmarks):
   
    landmarks = hand_landmarks.landmark
    fingers = []

    # Thumb
    thumb_open = landmarks[mp_hands.HandLandmark.THUMB_TIP].x < landmarks[mp_hands.HandLandmark.THUMB_IP].x
    fingers.append(1 if thumb_open else 0)

    # Jari lainnya (Index, Middle, Ring, Pinky)
    for i in [8, 12, 16, 20]:  # Ujung jari
        fingers.append(1 if landmarks[i].y < landmarks[i - 2].y else 0)

    if fingers == [1, 0, 0, 0, 0]:
        return "Ibu Jari"
    elif fingers == [0, 1, 0, 0, 0]:
        return "Telunjuk"
    elif fingers == [0, 0, 1, 0, 0]:
        return "Anda Tidak Sopan"  # Jari tengah saja yang terangkat
    elif fingers == [0, 0, 0, 1, 0]:
        return "Jari Manis"
    elif fingers == [0, 0, 0, 0, 1]:
        return "Kelingking"
    elif sum(fingers) == 0:
        return "Batu"
    elif sum(fingers) == 5:
        return "Kertas"
    elif fingers[1] == 1 and fingers[2] == 1 and sum(fingers) == 2:
        return "Gunting"
    else:
        return "Tidak Dikenali"

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat membaca frame dari kamera.")
        break

    # Membalikkan frame secara horizontal untuk tampilan mirroring
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)
    gesture = "Tidak Dikenali"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Gambar landmark tangan
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = recognize_gesture(hand_landmarks)

    #Tampilan informasi gestur
    cv2.putText(frame, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    #Frame
    cv2.imshow("Deteksi Hand Gesture", frame)

    #keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#kamera dan tutup jendela
cap.release()
cv2.destroyAllWindows()
