import time
from gpiozero import OutputDevice, InputDevice, PWMOutputDevice
import cv2
import numpy as np
from picamera2 import Picamera2
import sys

# Tespit edilen işaretleri buffer içinde tutarak sabitleme
detection_buffer = []
buffer_size = 5  # Tespitlerin geçerliliği için yeterli zaman
detection_threshold = 2 # Kaç kere tespit edildiğinde kesin tespit sayılır

# 1. Tekerlek Pin Tanımlamaları
in1 = 17
in2 = 27
ena = 25

# 2. Tekerlek Pin Tanımlamaları
in3 = 6
in4 = 5
enb = 16

# Uzaklık sensörü pinleri
TRIG = 23
ECHO = 24

# Motor Pinlerini Ayarlama
motor1_in1 = OutputDevice(in1) 
motor1_in2 = OutputDevice(in2)
motor1_pwm = PWMOutputDevice(ena)

motor2_in1 = OutputDevice(in3) 
motor2_in2 = OutputDevice(in4)
motor2_pwm = PWMOutputDevice(enb)

# Uzaklık sensörü fonksiyonu
trigger = OutputDevice(TRIG)
echo = InputDevice(ECHO)

closest = 0.3
medium = 0.7
farest = 1.3

move_forward_1_speed = 0.80 # Sol teker
move_forward_2_speed = 0.80  # Sağ teker

sharpening_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])

picam2 = Picamera2()  # Global olarak tanımla

def setup_camera():
    picam2.preview_configuration.main.size = (840, 840)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.preview_configuration.align()
    picam2.configure("preview")
    picam2.start()

# Uygulamanın başlangıcında çağır:
setup_camera()


def detect_no_entry_sign(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Kırmızı renk maskesi (No Entry işaretinin kırmızı olması muhtemeldir)
    lower_red1 = np.array([0, 70, 50])  # Daha düşük doygunluk
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([150, 70, 50])  # Kırmızı için üst spektrum
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # Morfolojik işlem
    kernel = np.ones((7, 7), np.uint8)  # Kernel boyutunu artır
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

    # Medyan filtre
    mask_red = cv2.medianBlur(mask_red, 5)

    # Kontur tespiti
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 2000:  # Daha büyük alanı tespit et
            (x, y), radius = cv2.minEnclosingCircle(contour)
            circularity = 4 * np.pi * cv2.contourArea(contour) / (cv2.arcLength(contour, True) ** 2)
            if 0.8 < circularity < 1.2:  # Dairesel konturları hedefle
                # ROI kontrolü
                roi_x, roi_y, roi_w, roi_h = cv2.boundingRect(contour)
                roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, roi_thresh = cv2.threshold(roi_gray, 200, 255, cv2.THRESH_BINARY)

                # Beyaz çevre konturları tespiti
                white_contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for white_contour in white_contours:
                    if cv2.contourArea(white_contour) > 500:
                        return True, contour

    return False, None

def is_arrow(contour, frame):
    """Konturun ok olup olmadığını kontrol eder."""
    if cv2.contourArea(contour) < 100:
        return False

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h

    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        return False

    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) != 7:  # Yalnızca 7 kenarlı okları al
        return False

    # 90 derece açılar kontrolü
    if not has_valid_angles(approx):
        return False

    # Renk kontrolü
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    mean_color = cv2.mean(frame, mask=mask)

    if mean_color[0] < 90 and mean_color[1] < 90 and mean_color[2] < 90:
        return True

    return False

def has_valid_angles(approx):
    """90 derece açılar kontrolünü yapar."""
    # Konturun tüm köşe noktalarını al
    points = approx[:, 0, :]  # (x, y) noktalarını çıkar

    # İki komşu nokta arasındaki açıları hesapla
    angles = []
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]  # Sonraki nokta
        p3 = points[(i + 2) % len(points)]  # İki sonraki nokta

        # Açı hesaplama (vektörler arası)
        v1 = p2 - p1
        v2 = p3 - p2
        dot_product = np.dot(v1, v2)
        mag_v1 = np.linalg.norm(v1)
        mag_v2 = np.linalg.norm(v2)
        angle = np.arccos(dot_product / (mag_v1 * mag_v2))
        angles.append(np.degrees(angle))

    # 90 derece açılar sayısını kontrol et
    ninety_degree_count = sum(80 <= angle <= 100 for angle in angles)
    if ninety_degree_count >= 2:
        return True
    return False

def detect_arrow_direction(frame):
    """Ok yönünü algıla ve yönü döndür."""
    # Görüntüyü keskinleştir
    sharpened_frame = cv2.filter2D(frame, -1, sharpening_kernel)

    gray = cv2.cvtColor(sharpened_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None, None

    arrow_contours = [cnt for cnt in contours if is_arrow(cnt, frame)]
    if not arrow_contours:
        return None, None

    largest_contour = max(arrow_contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None, None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    leftmost = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
    rightmost = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])

    direction = None
    if rightmost[0] > cx and abs(rightmost[0] - cx) > abs(leftmost[0] - cx):
        direction = "Right"
    elif leftmost[0] < cx and abs(leftmost[0] - cx) > abs(rightmost[0] - cx):
        direction = "Left"
    return direction, largest_contour


def turn_right(speed=0.82, duration=0.65):
    # Sadece sol motor ileri gider, sağ motor sabit kalır
    motor1_in1.off()  # Sol motor ileri
    motor1_in2.on()
    motor1_pwm.value = speed

    motor2_in1.off()  # Sağ motor duruyor
    motor2_in2.off()
    motor2_pwm.value = 0

    time.sleep(duration)  # Belirtilen süre boyunca dön
    stop_motors()

    time.sleep(0.3)
    distance = measure_distance()
    time.sleep(0.3)
    
    if distance > 55:
        move_forward(farest)  # 50 cm'den büyükse 2.5 saniye ileri gider
    elif 45 <= distance <= 55:
        move_forward(medium)  # 30-50 cm arasında 1.8 saniye ileri gider
    elif 35 <= distance < 45:
        move_forward(closest)  # 20-30 cm arasında 0.7 saniye ileri gider
    else:
        stop_motors()  # 20 cm ve altında durur
        process_image_and_act()


def turn_left(speed=0.80, duration=0.74):
    # Sadece sağ motor ileri gider, sol motor sabit kalır
    motor1_in1.off()  # Sol motor duruyor
    motor1_in2.off()
    motor1_pwm.value = 0

    motor2_in1.off()  # Sağ motor ileri
    motor2_in2.on()
    motor2_pwm.value = speed

    time.sleep(duration)  # Belirtilen süre boyunca dön
    stop_motors()

    time.sleep(0.3)
    distance = measure_distance()
    time.sleep(0.3)

    if distance > 55:
        move_forward(farest)  # 50 cm'den büyükse 2.5 saniye ileri gider
    elif 45 <= distance <= 55:
        move_forward(medium)  # 30-50 cm arasında 1.8 saniye ileri gider
    elif 35 <= distance < 45:
        move_forward(closest)  # 20-30 cm arasında 0.7 saniye ileri gider
    else:
        stop_motors()  # 35 cm ve altında durur
        process_image_and_act()


    
    
    
def process_image_and_act():
    
    
    while True:
        # Kameradan kareyi al
        frame = picam2.capture_array()
                
        # No Entry işareti tespiti
        no_entry_detected, detected_contour = detect_no_entry_sign(frame)
        if no_entry_detected:
            # print("stopped!")
            stop_motors()
            # cv2.destroyAllWindows()
            sys.exit()
            break
             
        # Ok yönünü algıla
        direction, arrow_contour = detect_arrow_direction(frame)
        if direction == "Left":
            motor1_in1.off()  # Motor1 ileri
            motor1_in2.on()  # Motor1 geri
            motor2_in1.off()  # Motor2 ileri
            motor2_in2.on()  # Motor2 geri
            motor1_pwm.value = move_forward_1_speed  # %65 hız
            motor2_pwm.value = move_forward_2_speed  # %65 hız
            time.sleep(0.5)
            stop_motors()
            time.sleep(0.3)
            # print("left")
            turn_left()
            
        elif direction == "Right":
            # print("right")
            motor1_in1.off()  # Motor1 ileri
            motor1_in2.on()  # Motor1 geri
            motor2_in1.off()  # Motor2 ileri
            motor2_in2.on()  # Motor2 geri
            motor1_pwm.value = move_forward_1_speed  # %65 hız
            motor2_pwm.value = move_forward_2_speed  # %65 hız
            time.sleep(0.5)
            stop_motors()
            time.sleep(0.3)
            turn_right()
        
        """
        # Görüntüyü çiz ve bilgiyi göster
        annotated_frame = frame.copy()
        
        # No Entry işareti tespiti
        if no_entry_detected:
            cv2.putText(annotated_frame, "No Entry Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if detected_contour is not None:
                cv2.drawContours(annotated_frame, [detected_contour], -1, (0, 0, 255), 2)
        
        # Ok yönü tespiti
        
        if direction:
            cv2.putText(annotated_frame, f"Arrow: {direction}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if arrow_contour is not None:
                cv2.drawContours(annotated_frame, [arrow_contour], -1, (0, 255, 0), 2)
        
        
        # Görüntüyü ekranda göster
        cv2.imshow("Detection", annotated_frame)
        
        # Çıkmak için 'q' tuşuna basın
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        """
        
        
        


def measure_distance():
    """Mesafeyi ölçmek için fonksiyon"""
    # TRIG pinini düşük yaparak sensörün sıfırlanmasını sağla
    trigger.off()
    time.sleep(0.1)

    # TRIG pinine 10 mikro saniyelik bir darbe gönder
    trigger.on()
    time.sleep(0.00001)
    trigger.off()

    # ECHO pininin yüksek olmasını bekle
    while not echo.is_active:
        pulse_start = time.time()

    # ECHO pininin düşük olmasını bekle
    while echo.is_active:
        pulse_end = time.time()

    # Puls süresini hesapla
    pulse_duration = pulse_end - pulse_start

    # Mesafeyi hesapla (sesin hava içindeki hızı 34300 cm/s)
    distance = pulse_duration * 17150
    return round(distance, 2)

def move_forward(duration=1):
    """Motorları ileri hareket ettir"""
    motor1_in1.off()  # Motor1 ileri
    motor1_in2.on()  # Motor1 geri
    motor2_in1.off()  # Motor2 ileri
    motor2_in2.on()  # Motor2 geri
    motor1_pwm.value = move_forward_1_speed  # %65 hız
    motor2_pwm.value = move_forward_2_speed # %65 hız
    time.sleep(duration)
    stop_motors()
    
    time.sleep(0.3)
    new_distance = measure_distance()
    # print(f"New Distance: {new_distance}")
    time.sleep(0.3)
    
    if new_distance > 55:
        # print(f"Distance: {new_distance}")
        # print("Mesafe > 50 cm: 4 saniye ileri gidiyor.")
        move_forward(farest)  # 50 cm'den büyükse 2.5 saniye ileri gider
    elif 45 <= new_distance <= 55:
        # print(f"Distance: {new_distance}")
        # print("30 cm <= Mesafe <= 50 cm: 3 saniye ileri gidiyor.")
        move_forward(medium)  # 30-50 cm arasında 1.8 saniye ileri gider
    elif 35 <= new_distance < 45:
        # print(f"Distance: {new_distance}")
        # print("20 cm <= Mesafe < 30 cm: 1 saniye ileri gidiyor.")
        move_forward(closest)  # 20-30 cm arasında 0.7 saniye ileri gider
    else:
        # print(f"Distance: {new_distance}")
        # print("Mesafe < 20 cm: Duruyor.")
        stop_motors()
        process_image_and_act()
    
    
    
def stop_motors():
    """Motorları durdur"""
    motor1_in1.off()
    motor1_in2.off()
    motor2_in1.off()
    motor2_in2.off()
    motor1_pwm.value = 0
    motor2_pwm.value = 0

# Mesafeyi ölçüp, hareket etmeyi kontrol et
distance = measure_distance()
# print(f"Mesafe: {distance} cm")

# Mesafeye göre hareket et
if distance > 55:
    # print("Mesafe > 50 cm: 4 saniye ileri gidiyor.")
    move_forward(farest)  # 50 cm'den büyükse 2.5 saniye ileri gider
elif 45 <= distance <= 55:
    # print("30 cm <= Mesafe <= 50 cm: 3 saniye ileri gidiyor.")
    move_forward(medium)  # 30-50 cm arasında 1.8 saniye ileri gider
elif 35 <= distance < 45:
    # print("20 cm <= Mesafe < 30 cm: 1 saniye ileri gidiyor.")
    move_forward(closest)  # 20-30 cm arasında 0.7 saniye ileri gider
else:
    # print("Mesafe < 30 cm: Duruyor.")
    stop_motors()  # 20 cm ve altında durur
    process_image_and_act()


