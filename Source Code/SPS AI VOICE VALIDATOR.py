import sys
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QVBoxLayout, QGroupBox, QGridLayout, QLabel, 
    QComboBox, QLineEdit, QPushButton, QWidget, QFileDialog, QListWidget, 
    QMessageBox, QProgressBar
)
import pyqtgraph as pg
from PyQt5.QtCore import QThread, pyqtSignal
import requests
import joblib
import librosa
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------------------
# Konfigurasi Panjang Audio Tetap
# ---------------------------
FIXED_DURATION = 3.0  # durasi audio dalam detik
SR = 22050  # sample rate yang digunakan
FIXED_WIDTH = 128  # jumlah frame/time-step setelah ekstraksi mel-spectrogram

# ---------------------------
# Fungsi bantu untuk memastikan audio berdurasi tetap
# ---------------------------
def load_fixed_length_audio(file_path, sr=SR, duration=FIXED_DURATION):
    y, original_sr = librosa.load(file_path, sr=sr)
    target_length = int(duration * sr)
    if len(y) < target_length:
        # Pad dengan zero hingga panjang mencapai target_length
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    else:
        # Jika lebih panjang, potong
        y = y[:target_length]
    return y, sr

# ---------------------------
# Bagian ML: Ekstraksi fitur dan prediksi
# ---------------------------
MODEL_RF_PATH = "dataset_rf.pkl"
MODEL_CNN_PATH = "dataset_cnn.h5"

def extract_features(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        # Fitur = rata-rata mfcc (13 dimensi) + rata-rata spectral contrast (7 dimensi) = total 20
        return np.hstack((np.mean(mfcc, axis=1), np.mean(spectral_contrast, axis=1)))
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return np.zeros(20)  # Fallback jika gagal ekstrak fitur

def extract_cnn_features(audio_file):
    # Muat audio dengan durasi tetap
    y, sr = load_fixed_length_audio(audio_file, sr=SR, duration=FIXED_DURATION)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Sekarang spectrogram.shape = (n_mels, time)
    # Pastikan semua input memiliki panjang time (width) yang sama
    # Jika time < FIXED_WIDTH, kita pad dengan zero
    # Jika time > FIXED_WIDTH, kita potong
    if spectrogram.shape[1] < FIXED_WIDTH:
        diff = FIXED_WIDTH - spectrogram.shape[1]
        spectrogram = np.pad(spectrogram, ((0,0),(0,diff)), mode='constant')
    else:
        spectrogram = spectrogram[:, :FIXED_WIDTH]

    # Expand dimension untuk channel dan batch
    spectrogram = np.expand_dims(spectrogram, axis=-1)  # channel
    spectrogram = np.expand_dims(spectrogram, axis=0)   # batch
    return spectrogram

def perform_classification(file_path):
    result = {}
    features = extract_features(file_path)

    # Random Forest
    rf_proba = []
    if os.path.exists(MODEL_RF_PATH):
        rf_model = joblib.load(MODEL_RF_PATH)
        # Cek kelas yang dikenali oleh model
        classes_rf = rf_model.classes_
        print(f"Random Forest Classes: {classes_rf}")
        rf_proba = rf_model.predict_proba([features])[0]
        print(f"Random Forest Probabilities: {rf_proba}")
    else:
        print("Model RF not found.")

    # CNN
    cnn_proba = []
    if os.path.exists(MODEL_CNN_PATH):
        cnn_model = load_model(MODEL_CNN_PATH)
        cnn_input = extract_cnn_features(file_path)
        cnn_proba = cnn_model.predict(cnn_input)[0]
        print(f"CNN Probabilities: {cnn_proba}")
    else:
        print("Model CNN not found.")

    # Menggabungkan probabilitas jika kedua model ada
    if rf_proba.size > 0 and cnn_proba.size > 0:
        # Pastikan kedua model mengeluarkan probabilitas untuk kelas yang sama
        # Misalnya: [AI, Human]
        # Rata-rata probabilitas
        combined_proba = (rf_proba + cnn_proba) / 2
        print(f"Combined Probabilities: {combined_proba}")

        # Tentukan kelas berdasarkan probabilitas rata-rata
        if combined_proba[1] > combined_proba[0]:
            combined_prediction = ("Human", combined_proba[1] * 100)
        else:
            combined_prediction = ("AI", combined_proba[0] * 100)
        
        result['combined'] = combined_prediction
    elif rf_proba.size > 0:
        # Hanya Random Forest yang tersedia
        if len(rf_proba) == len(rf_model.classes_):
            # Mapping probabilitas ke kelas
            rf_human_prob = 0
            rf_ai_prob = 0
            for idx, cls in enumerate(classes_rf):
                if cls == 1:
                    rf_human_prob = rf_proba[idx]
                elif cls == 0:
                    rf_ai_prob = rf_proba[idx]
            # Pastikan kedua probabilitas terdefinisi
            if 'rf_human_prob' in locals() and 'rf_ai_prob' in locals():
                if rf_human_prob > rf_ai_prob:
                    combined_prediction = ("Human", rf_human_prob * 100)
                else:
                    combined_prediction = ("AI", rf_ai_prob * 100)
                result['combined'] = combined_prediction
            else:
                # Jika salah satu kelas tidak ditemukan
                result['combined'] = ("Unknown", 0)
        else:
            # Jika hanya satu kelas yang dikenali
            cls = classes_rf[0]
            prob = rf_proba[0]
            if cls == 1:
                combined_prediction = ("Human", prob * 100)
            else:
                combined_prediction = ("AI", prob * 100)
            result['combined'] = combined_prediction
    elif cnn_proba.size > 0:
        # Hanya CNN yang tersedia
        if len(cnn_proba) == 2:
            cnn_ai_prob = cnn_proba[0]
            cnn_human_prob = cnn_proba[1]
            if cnn_human_prob > cnn_ai_prob:
                combined_prediction = ("Human", cnn_human_prob * 100)
            else:
                combined_prediction = ("AI", cnn_ai_prob * 100)
            result['combined'] = combined_prediction
        else:
            # Jika jumlah kelas berbeda
            result['combined'] = ("Unknown", 0)
    else:
        # Jika tidak ada model yang tersedia
        result['combined'] = ("No models available", 0)

    return result

# ---------------------------
# Bagian Training Worker
# ---------------------------
class TrainingWorker(QThread):
    training_progress = pyqtSignal(str)
    training_completed = pyqtSignal(str)

    def __init__(self, human_files, ai_files):
        super().__init__()
        self.human_files = human_files
        self.ai_files = ai_files

    def run(self):
        try:
            # Pastikan ada file dari kedua kelas
            if not self.human_files or not self.ai_files:
                self.training_completed.emit("Error: Both Human and AI audio files are required for training.")
                return

            self.training_progress.emit("Ekstraksi fitur...")
            features = []
            labels = []
            # Ekstraksi fitur untuk kelas Human
            for file in self.human_files:
                feat = extract_features(file)
                features.append(feat)
                labels.append(1)  # Human

            # Ekstraksi fitur untuk kelas AI
            for file in self.ai_files:
                feat = extract_features(file)
                features.append(feat)
                labels.append(0)  # AI

            features = np.array(features)
            labels = np.array(labels)

            # Pastikan kedua kelas ada
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                self.training_completed.emit("Error: Training data must include both Human and AI classes.")
                return

            self.training_progress.emit("Melatih model Random Forest...")
            # Melatih Random Forest
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
            rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_clf.fit(X_train, y_train)
            y_pred_rf = rf_clf.predict(X_test)
            rf_report = classification_report(y_test, y_pred_rf, output_dict=True)
            joblib.dump(rf_clf, MODEL_RF_PATH)
            self.training_progress.emit("Model Random Forest selesai dilatih dan disimpan.")

            self.training_progress.emit("Melatih model CNN...")
            # Melatih CNN
            # Ekstraksi fitur untuk CNN dengan dimensi tetap
            X_cnn = []
            for file in self.human_files + self.ai_files:
                spect = extract_cnn_features(file)  # sudah ukuran tetap
                X_cnn.append(spect)

            # Gabungkan X_cnn (list of np.array dengan shape (1, n_mels, fixed_width, 1))
            # karena sudah sama ukuran, np.vstack akan aman
            X_cnn = np.vstack(X_cnn)  # shape: (n_samples, n_mels, fixed_width, 1)
            y_cnn = np.array(labels)

            # Pastikan y_cnn memiliki dua kelas
            unique_labels_cnn = np.unique(y_cnn)
            if len(unique_labels_cnn) < 2:
                self.training_completed.emit("Error: CNN training data must include both Human and AI classes.")
                return

            input_shape = X_cnn.shape[1:]
            cnn_model = Sequential([
                Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
                MaxPooling2D((2,2)),
                Conv2D(64, (3,3), activation='relu'),
                MaxPooling2D((2,2)),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(2, activation='softmax')  # 2 kelas: AI dan Human
            ])
            cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Melatih model
            cnn_model.fit(X_cnn, y_cnn, epochs=20, batch_size=16, validation_split=0.2)

            # Evaluasi model
            loss, accuracy = cnn_model.evaluate(X_cnn, y_cnn, verbose=0)
            cnn_model.save(MODEL_CNN_PATH)
            self.training_progress.emit("Model CNN selesai dilatih dan disimpan.")

            self.training_completed.emit(
                "Pelatihan model selesai.\nRandom Forest Accuracy: {:.2f}%\nCNN Accuracy: {:.2f}%".format(
                    rf_report['accuracy'] * 100, accuracy * 100
                )
            )
        except Exception as e:
            self.training_completed.emit(f"Error during training: {e}")

# ---------------------------
# Bagian DFT Worker (dari kode awal)
# ---------------------------
class DFTWorker(QThread):
    dft_completed = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, audio_data, sampling_rate):
        super().__init__()
        self.audio_data = audio_data
        self.sampling_rate = sampling_rate

    def run(self):
        # Perform the DFT without any window function
        N = len(self.audio_data)
        dft_data = np.fft.fft(self.audio_data)
        amplitude = np.abs(dft_data)[:N // 2]
        frequencies = np.fft.fftfreq(N, 1 / self.sampling_rate)[:N // 2]

        # Emit the result when done
        self.dft_completed.emit(frequencies, amplitude)

# ---------------------------
# Bagian GUI PyQt
# ---------------------------
class Ui_MainWindow(QtWidgets.QMainWindow):  # Pastikan ini subclass dari QMainWindow
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # Panggil setupUi dengan `self` sebagai parameter utama

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 900)

        # Buat central widget
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)  # Atur central widget

        # Main Layout
        self.layout = QVBoxLayout(self.centralwidget)

        # Parameters Group Box
        self.groupBox = QGroupBox("Parameters", self.centralwidget)
        self.gridLayout = QGridLayout(self.groupBox)

        # Audio Device Selection
        self.label_audio_device = QLabel("Audio Device")
        self.combo_audio_device = QComboBox()
        devices = sd.query_devices()
        input_devices = [device["name"] for device in devices if device["max_input_channels"] > 0]
        self.combo_audio_device.addItems(input_devices)
        self.gridLayout.addWidget(self.label_audio_device, 0, 0)
        self.gridLayout.addWidget(self.combo_audio_device, 0, 1)

        # Amplitude Input
        self.label_amplitude = QLabel("Amplitude")
        self.lineEdit_amplitude = QLineEdit("100")
        self.gridLayout.addWidget(self.label_amplitude, 1, 0)
        self.gridLayout.addWidget(self.lineEdit_amplitude, 1, 1)

        # Sampling Rate Input
        self.label_sampling_rate = QLabel("Sampling Rate (>1000 Hz)")
        self.lineEdit_sampling_rate = QLineEdit("44100")
        self.gridLayout.addWidget(self.label_sampling_rate, 2, 0)
        self.gridLayout.addWidget(self.lineEdit_sampling_rate, 2, 1)

        # Frequency Input
        self.label_frequency = QLabel("Frequency")
        self.lineEdit_frequency = QLineEdit("440")
        self.gridLayout.addWidget(self.label_frequency, 0, 2)
        self.gridLayout.addWidget(self.lineEdit_frequency, 0, 3)

        # Update Interval Input
        self.label_update_interval = QLabel("Update Interval (1 to 100 ms)")
        self.lineEdit_update_interval = QLineEdit("30")
        self.gridLayout.addWidget(self.label_update_interval, 1, 2)
        self.gridLayout.addWidget(self.lineEdit_update_interval, 1, 3)

        # Main Buttons
        self.pushButton_plot = QPushButton("Plot It!")
        self.pushButton_stop = QPushButton("Stop")
        self.pushButton_record = QPushButton("Record")
        self.pushButton_load = QPushButton("Load File")
        self.gridLayout.addWidget(self.pushButton_plot, 2, 2)
        self.gridLayout.addWidget(self.pushButton_stop, 2, 3)
        self.gridLayout.addWidget(self.pushButton_record, 2, 4)
        self.gridLayout.addWidget(self.pushButton_load, 2, 5)

        # Analysis Buttons
        self.pushButton_analyze = QPushButton("Analyze")
        self.gridLayout.addWidget(self.pushButton_analyze, 3, 0)

        self.pushButton_reset = QPushButton("Reset")
        self.gridLayout.addWidget(self.pushButton_reset, 3, 1)

        # Upload to Edge Impulse
        self.pushButton_edge = QPushButton("Upload to Edge Impulse")
        self.gridLayout.addWidget(self.pushButton_edge, 3, 2)

        # Add Parameter Group Box to Main Layout
        self.layout.addWidget(self.groupBox)

        # Original Signal Plot
        self.plot_widget_original = pg.PlotWidget(self.centralwidget)
        self.plot_widget_original.setBackground("k")
        self.plot_widget_original.setLabel("bottom", "Time")
        self.plot_widget_original.setLabel("left", "Amplitude")
        self.plot_widget_original.setTitle("Original Signal")
        self.layout.addWidget(self.plot_widget_original)

        # DFT Plot
        self.plot_widget_dft = pg.PlotWidget(self.centralwidget)
        self.plot_widget_dft.setBackground("k")
        self.plot_widget_dft.setLabel("bottom", "Frequency (Hz)")
        self.plot_widget_dft.setLabel("left", "Amplitude")
        self.plot_widget_dft.setTitle("DFT of Signal")
        self.layout.addWidget(self.plot_widget_dft)

        # Analysis Results
        self.resultGroupBox = QGroupBox("Analysis Results", self.centralwidget)
        self.resultLayout = QGridLayout(self.resultGroupBox)

        self.label_combined = QLabel("Combined Prediction:")
        self.label_combined_result = QLabel("N/A")
        self.progressBar_combined = QProgressBar()
        self.progressBar_combined.setValue(0)
        self.resultLayout.addWidget(self.label_combined, 0, 0)
        self.resultLayout.addWidget(self.label_combined_result, 0, 1)
        self.resultLayout.addWidget(self.progressBar_combined, 0, 2)

        self.layout.addWidget(self.resultGroupBox)

        # Training AI Models
        self.trainingGroupBox = QGroupBox("Training AI Models", self.centralwidget)
        self.trainingLayout = QGridLayout(self.trainingGroupBox)

        self.pushButton_add_human = QPushButton("Add Human Audio Files")
        self.listWidget_human = QListWidget()
        self.trainingLayout.addWidget(self.pushButton_add_human, 0, 0)
        self.trainingLayout.addWidget(self.listWidget_human, 1, 0)

        self.pushButton_add_ai = QPushButton("Add AI Audio Files")
        self.listWidget_ai = QListWidget()
        self.trainingLayout.addWidget(self.pushButton_add_ai, 0, 1)
        self.trainingLayout.addWidget(self.listWidget_ai, 1, 1)

        self.pushButton_train = QPushButton("Train AI")
        self.trainingLayout.addWidget(self.pushButton_train, 2, 0, 1, 2)

        self.label_training_status = QLabel("Training Status: Not started.")
        self.trainingLayout.addWidget(self.label_training_status, 3, 0, 1, 2)

        self.layout.addWidget(self.trainingGroupBox)

        # Central Widget Setup
        MainWindow.setCentralWidget(self.centralwidget)
        MainWindow.setWindowTitle("PyShine Live Voice Plot with AI Classification and Training")

        # Signal Connections
        self.pushButton_plot.clicked.connect(self.start_plotting)
        self.pushButton_stop.clicked.connect(self.stop_plotting)
        self.pushButton_record.clicked.connect(self.start_recording)
        self.pushButton_load.clicked.connect(self.load_file)
        self.pushButton_analyze.clicked.connect(self.analyze_audio)
        self.pushButton_reset.clicked.connect(self.reset_analysis)

        self.pushButton_edge.clicked.connect(lambda: self.upload_audio_to_edge_impulse(self.audio_filename))
        self.pushButton_add_human.clicked.connect(self.add_human_files)
        self.pushButton_add_ai.clicked.connect(self.add_ai_files)
        self.pushButton_train.clicked.connect(self.train_models)

        # Variables
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.stream = None
        self.recording = False
        self.history_data = np.array([])
        self.current_file_path = None
        
        #audio loader
        self.audio_filename = None

        # Edge Impulse API
        self.api_url = "https://ingestion.edgeimpulse.com/api/training/files"
        self.api_key = "ei_7ad26b1c5b85e410ad146685a3f23d8c6e4b7d0138d6e8a6"
        self.label = "edge_Ai"

        # Training Lists
        self.training_human_files = []
        self.training_ai_files = []

    def upload_audio_to_edge_impulse(self, audio_filename):
        try:
            with open(audio_filename, "rb") as f:
                response = requests.post(
                    self.api_url,
                    headers={
                        "x-label": self.label,
                        "x-api-key": self.api_key,
                    },
                    files={"data": (os.path.basename(audio_filename), f, "audio/wav")},
                )

            if response.status_code == 200:
                QMessageBox.information(None, "Upload Successful", f"File '{audio_filename}' uploaded to Edge Impulse!")
            else:
                QMessageBox.warning(None, "Upload Failed", f"Upload failed. Status code: {response.status_code}\nResponse: {response.text}")
        except FileNotFoundError:
            QMessageBox.critical(None, "File Not Found", f"File '{audio_filename}' not found.")

    # ---------------------------
    # Fungsi untuk Memuat File Training
    # ---------------------------
    def add_human_files(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(
            None, "Select Human Audio Files", "", "Audio Files (*.wav *.mp3)", options=options
        )
        if files:
            for file in files:
                if file not in self.training_human_files:
                    self.training_human_files.append(file)
                    self.listWidget_human.addItem(file)

    def add_ai_files(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(
            None, "Select AI Audio Files", "", "Audio Files (*.wav *.mp3)", options=options
        )
        if files:
            for file in files:
                if file not in self.training_ai_files:
                    self.training_ai_files.append(file)
                    self.listWidget_ai.addItem(file)

    def train_models(self):
        if not self.training_human_files or not self.training_ai_files:
            QMessageBox.warning(
                None, "Training Error", "Please add both Human and AI audio files for training."
            )
            return

        self.pushButton_train.setEnabled(False)
        self.label_training_status.setText("Training Status: Training in progress...")

        # Mulai thread pelatihan
        self.training_worker = TrainingWorker(self.training_human_files, self.training_ai_files)
        self.training_worker.training_progress.connect(self.update_training_status)
        self.training_worker.training_completed.connect(self.training_finished)
        self.training_worker.start()

    def update_training_status(self, message):
        self.label_training_status.setText(f"Training Status: {message}")

    def training_finished(self, message):
        self.label_training_status.setText(f"Training Status: {message}")
        self.pushButton_train.setEnabled(True)
        QMessageBox.information(None, "Training Completed", message)

    # ---------------------------
    # Fungsi Audio Callback dan Plotting
    # ---------------------------
    def audio_callback(self, indata, frames, time, status):
        indata = np.clip(indata, -1, 1)  # Normalize input
        # Tambahkan data baru ke buffer
        self.history_data = np.append(self.history_data, indata[:, 0])

        if self.recording:
            self.recorded_frames.append(indata.copy())

    def update_plot(self):
        try:
            sampling_rate = int(self.lineEdit_sampling_rate.text())
            current_length = len(self.history_data)

            if current_length <= 1:
                return

            time_axis = np.linspace(0, current_length / sampling_rate, current_length)

            self.plot_widget_original.plot(time_axis, self.history_data, clear=True, pen='m')
            self.plot_widget_original.setYRange(-1, 1)  
            self.plot_widget_original.setLabel('bottom', 'Time (s)')
            self.plot_widget_original.setTitle("Real-Time Signal")

            dft_data = np.abs(np.fft.fft(self.history_data))
            freq_axis = np.fft.fftfreq(len(dft_data), d=1/sampling_rate)
            positive_freqs = freq_axis[:len(freq_axis)//2]
            positive_dft_data = dft_data[:len(dft_data)//2]

            self.plot_widget_dft.plot(positive_freqs, positive_dft_data, clear=True, pen='b')
            self.plot_widget_dft.setLabel('bottom', 'Frequency (Hz)')
            if positive_dft_data.size > 0:
                self.plot_widget_dft.setYRange(0, max(positive_dft_data) * 1.1)
        except Exception as e:
            print(f"Error updating plot: {e}")

    def start_plotting(self):
        try:
            device_name = self.combo_audio_device.currentText()
            device_id = sd.query_devices(device_name)['index']
            sampling_rate = int(self.lineEdit_sampling_rate.text())
            update_interval = int(self.lineEdit_update_interval.text())

            self.stream = sd.InputStream(
                device=device_id, channels=1, samplerate=sampling_rate, callback=self.audio_callback
            )
            self.stream.start()
            self.timer.start(update_interval) 

            self.history_data = np.array([])  
            print(f"Started streaming on device {device_name} with sampling rate {sampling_rate} Hz.")
        except Exception as e:
            print(f"Error starting audio stream: {e}")

    def stop_plotting(self):
        try:
            if self.stream:
                self.stream.stop()
                self.stream = None
            self.timer.stop()
            if self.recording:
                self.stop_recording()
            print("Stopped streaming.")
        except Exception as e:
            print(f"Error stopping audio stream: {e}")

    def start_recording(self):
        if not self.recording:
            self.recording = True
            self.recorded_frames = []
            print("Recording started...")

    def stop_recording(self):
        if self.recording:
            self.recording = False
            audio_data = np.concatenate(self.recorded_frames, axis=0)
            sf.write("recorded_audio.wav", audio_data, 44100)
            self.current_file_path = "recorded_audio.wav"
            print("Recording stopped and saved.")

    def load_file(self):
        # Open a file dialog to select an audio file
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.mp3)",
            options=options
        )
        
        # Check if a file was selected
        if file_name:
            self.audio_filename = file_name
            print(f"Loaded file: {self.audio_filename}")
        else:
            print("No file selected.")

        if file_name:
            try:
                data, sampling_rate = sf.read(file_name)
                print(f"Loaded file: {file_name} with sampling rate {sampling_rate} Hz")

                # Konversi ke mono jika stereo
                if data.ndim > 1:
                    data = data[:, 0]
                    print("Converted stereo to mono by selecting the first channel.")

                self.history_data = data
                self.current_file_path = file_name

                # Plot sinyal asli
                self.plot_widget_original.plot(self.history_data, clear=True, pen='m')

                # Hitung dan plot DFT
                if len(self.history_data) > 0:
                    dft_data = np.abs(np.fft.fft(self.history_data))
                    freq_axis = np.fft.fftfreq(len(dft_data), d=1 / sampling_rate)
                    positive_freqs = freq_axis[:len(freq_axis) // 2]
                    positive_dft_data = dft_data[:len(dft_data) // 2]

                    self.plot_widget_dft.plot(positive_freqs, positive_dft_data, clear=True, pen='b')
                    self.plot_widget_dft.setLabel('bottom', 'Frequency (Hz)')
                    if positive_dft_data.size > 0:
                        self.plot_widget_dft.setYRange(0, max(positive_dft_data) * 1.1)
                    print("DFT plot updated successfully.")
                else:
                    print("Loaded audio data is empty. DFT plot not updated.")

            except Exception as e:
                QMessageBox.critical(self, "File Load Error", f"Failed to load file: {e}")
                print(f"Error loading file: {e}")

    def upload_audio_to_edge_impulse(self, audio_filename):
        try:
            with open(audio_filename, "rb") as f:
                response = requests.post(
                    self.api_url,
                    headers={
                        "x-label": self.label,
                        "x-api-key": self.api_key,
                    },
                    files={"data": (os.path.basename(audio_filename), f, "audio/wav")},
                )

            if response.status_code == 200:
                QMessageBox.information(self, "Upload Successful", f"File '{audio_filename}' uploaded to Edge Impulse!")
            else:
                QMessageBox.warning(self, "Upload Failed", f"Upload failed. Status code: {response.status_code}\nResponse: {response.text}")

        except FileNotFoundError:
            QMessageBox.critical(self, "File Not Found", f"File '{audio_filename}' not found.")
        except Exception as e:
            QMessageBox.critical(self, "Upload Error", f"An error occurred: {e}")

    def upload_to_edge_impulse(self):
        if self.current_file_path:
            self.upload_audio_to_edge_impulse(self.current_file_path)
        else:
            QMessageBox.warning(self, "No File Selected", "Please load an audio file before uploading.")

    def analyze_audio(self):
        if self.current_file_path is None:
            QMessageBox.warning(None, "Analysis Error", "No file to analyze.")
            self.label_combined_result.setText("N/A")
            self.progressBar_combined.setValue(0)
            self.progressBar_combined.setFormat("")
            return

        result = perform_classification(self.current_file_path)

        # Tampilkan hasil gabungan
        if 'combined' in result:
            combined_res, combined_prob = result['combined']
        else:
            combined_res, combined_prob = ("Unknown", 0)

        # Update Label dan Progress Bar untuk Combined Prediction
        self.label_combined_result.setText(f"{combined_res}")
        self.progressBar_combined.setValue(int(combined_prob))
        if combined_res == "Human":
            self.progressBar_combined.setFormat(f"Human: {combined_prob:.2f}%")
        elif combined_res == "AI":
            self.progressBar_combined.setFormat(f"AI: {combined_prob:.2f}%")
        else:
            self.progressBar_combined.setFormat(f"{combined_res}: {combined_prob:.2f}%")

        # Opsional: Tampilkan pesan
        QMessageBox.information(
            None, "Analysis Result", 
            f"Combined Prediction: {combined_res} ({combined_prob:.2f}%)"
        )
        print(f"Analysis Result:\nCombined: {combined_res} ({combined_prob:.2f}%)")

    # ---------------------------
    # Fungsi Reset untuk Mengatur Ulang Analisis
    # ---------------------------
    def reset_analysis(self):
        reply = QMessageBox.question(
            None, 'Reset Confirmation', 
            "Are you sure you want to reset the analysis? This will clear all current data.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Hentikan stream jika sedang berjalan
            if self.stream:
                self.stop_plotting()

            # Hentikan rekaman jika sedang berlangsung
            if self.recording:
                self.stop_recording()

            # Reset buffer data
            self.history_data = np.array([])

            # Reset plot widgets
            self.plot_widget_original.clear()
            self.plot_widget_dft.clear()

            # Reset hasil analisis
            self.label_combined_result.setText("N/A")
            self.progressBar_combined.setValue(0)
            self.progressBar_combined.setFormat("")

            # Reset current file path
            self.current_file_path = None

            QMessageBox.information(None, "Reset", "Analysis has been reset. You can record or load a new audio file.")

# ---------------------------
# Fungsi Utama
# ---------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
