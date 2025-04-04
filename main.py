import cv2
import sys
import numpy as np
import speech_recognition as sr
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from ultralytics import YOLO
import pyttsx3
import threading
import queue


# Sesli konuşma kuyruğu
speak_queue = queue.Queue()

def speak_worker():
    while True:
        text = speak_queue.get()
        if text is None:
            break
        engine = pyttsx3.init()  # pyttsx3 motoru başlat
        engine.setProperty('rate', 150)  # Konuşma hızını ayarlayabilirsin
        engine.setProperty('volume', 1)  # Ses seviyesini ayarlayabilirsin (0.0 - 1.0)
        engine.say(text)
        engine.runAndWait()
        speak_queue.task_done()


# Thread oluştur ve başlat
threading.Thread(target=speak_worker,daemon=True).start()

def speak(text):
    speak_queue.put(text)

# YOLO modelini yükle
model = YOLO("yolov8n.pt")

# Tanınacak nesneler
TARGET_OBJECTS = ["cell phone", "clock", "key", "glass", "cup", "cat", "person", "people"]

class VoiceCommandThread(QThread):
    command_signal = pyqtSignal(str)  # Komutları ana thread'e göndermek için sinyal
    
    def run(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Sesli komut bekleniyor...")
            self.command_signal.emit("Sesli komut bekleniyor...")
            speak("sesli komut bekleniyor")

            try:
                audio = recognizer.listen(source)  # 5 saniye dinle
                command = recognizer.recognize_google(audio, language="en-US").lower()
                print(f"Algılanan komut: {command}")
                self.command_signal.emit(f"Komut: {command}")
                speak(f"Komut: {command}")  # Komutu sesli olarak ilet
            except sr.UnknownValueError:
                self.command_signal.emit("Ses anlaşılamadı, tekrar deneyin.")
                speak("Ses anlaşılamadı, tekrar deneyin.")  # Ses anlaşılamadı mesajı

            except sr.RequestError:
                self.command_signal.emit("Google API bağlantı hatası.")
                speak("Google API bağlantı hatası.")  # Bağlantı hatası mesajı

class YOLOApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLOv8 Nesne Tanıma - Sesli Komut")
        self.setGeometry(100, 100, 800, 600)

        # Video Label
        self.video_label = QLabel(self)

        # Durum Metni
        self.result_label = QLabel("Durum: Bekleniyor...", self)
        self.voice_label = QLabel("Ses Komutu: Bekleniyor...", self)

        # Butonlar
        self.start_button = QPushButton("Kamerayı Aç", self)
        self.start_button.clicked.connect(self.start_camera)

        self.stop_button = QPushButton("Kamerayı Kapat", self)
        self.stop_button.clicked.connect(self.stop_camera)

        self.voice_button = QPushButton("Sesli Komut Ver", self)
        self.voice_button.clicked.connect(self.listen_voice_command)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.voice_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.voice_button)
        self.setLayout(layout)

        # Kamera
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Sesli komut thread
        self.voice_thread = VoiceCommandThread()
        self.voice_thread.command_signal.connect(self.process_voice_command)
        self.target_object = None  # Kullanıcının aradığı nesne

    def start_camera(self):
        """Kamerayı başlatır"""
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def stop_camera(self):
        """Kamerayı durdurur"""
        if self.cap:
            self.cap.release()
        self.timer.stop()
        self.video_label.clear()

    def listen_voice_command(self):
        """Sesli komutu dinler"""
        self.voice_thread.start()

    def process_voice_command(self, command):
        """Sesli komuttan gelen veriyi işler"""
        self.voice_label.setText(command)
        speak(command)
        for obj in TARGET_OBJECTS:
            if obj in command:
                self.target_object = obj
                self.voice_label.setText(f"Aranan nesne: {obj}")
                speak(f"Aranan nesne: {obj}")
                return

        self.voice_label.setText("Belirtilen nesne listede yok.")
        speak("Belirtilen nesne listede yok.")

    def detect_objects(self, frame):
        """YOLO ile nesne tespiti yapar"""
        results = model(frame)
        detected_objects = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinatları al
                label = model.names[int(box.cls[0])]  # Etiket al
                score = box.conf[0].item()  # Güven skoru

                if label in TARGET_OBJECTS:
                    detected_objects.append(f"{label} ({score:.2f})")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Eğer kullanıcı bir nesne arıyorsa ve bulunduysa ekranda mesaj ver
                if self.target_object and label == self.target_object:
                    cv2.putText(frame, f"{label} BULUNDU!", (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                    self.result_label.setText(f"{label} bulundu!")
                    speak(f"{label} bulundu!")

        if not detected_objects:
            self.result_label.setText("Nesne Bulunamadı")
            speak("Nesne Bulunamadı")

    def update_frame(self):
        """Kameradan görüntüyü alır ve ekrana yansıtır"""
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.detect_objects(frame)  # Nesne tanıma işlemini yap
                
                # OpenCV görüntüsünü PyQt formatına çevir
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(qimg))

# Uygulamayı çalıştır
app = QApplication(sys.argv)
window = YOLOApp()
window.show()
sys.exit(app.exec_())
