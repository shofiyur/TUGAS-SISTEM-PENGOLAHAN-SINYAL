![image](https://github.com/user-attachments/assets/c1e5de6e-2045-4a10-8f0e-eb0d2d3e8a22)


PENGEMBANGAN SOFTWARE PENDETEKSI SUARA SINTESIS HASIL GENERATE AI BERBASIS MACHINE LEARNING CONVOLUTIONAL NEURAL NETWORK (CNN) DAN RANDOM FOREST (RF)

Dengan perkembangan pesat teknologi kecerdasan buatan (AI) dalam pengolahan suara, suara sintetis kini semakin mirip dengan suara manusia. Hal ini menimbulkan tantangan serius dalam sistem otentikasi berbasis suara, karena risiko meningkatnya tingkat penerimaan yang salah (false acceptance rate) dapat mengancam keamanan siber dan kepercayaan publik terhadap informasi digital. Untuk mengatasi masalah ini, kami mengembangkan perangkat lunak yang mampu menganalisis dan membedakan suara yang dihasilkan oleh AI dan manusia dengan akurasi tinggi.

Pengembang;
1.	MUHAMMAD HADID QUSHAIRI (2042231025)
2.	MOCHAMMAD SHOFIYUR ROCHMAN (2042231031)
3.	MAULIDAN ARRIDLO (2042231059)
4.	WILDAN RIZKI AUZAY (2042231061)
5.  Ahmad Radhy (Supervisor)

Fitur;

Antarmuka Pengguna Intuitif: Dibangun dengan PyQt5 untuk pengalaman pengguna yang responsif dan menarik.
Perekaman dan Pemutaran Audio: Rekam audio secara real-time atau muat file audio untuk analisis.
Visualisasi Sinyal: Plot sinyal asli dan Transformasi Fourier Diskrit (DFT) secara real-time.
Klasifikasi Suara AI dan Manusia: Gunakan model Random Forest dan Convolutional Neural Network (CNN) untuk membedakan suara AI dan manusia dengan akurasi tinggi.
Pelatihan Model AI: Tambahkan file audio untuk melatih model Random Forest dan CNN sesuai kebutuhan.
Integrasi Edge Impulse: Unggah audio ke Edge Impulse untuk analisis lebih lanjut.

Teknologi yang digunakan;

Bahasa Pemrograman: Python
Framework GUI: PyQt5
Analisis Audio: Librosa, SoundDevice, SoundFile
Pembelajaran Mesin: Scikit-learn (Random Forest), TensorFlow/Keras (CNN)
Visualisasi: PyQtGraph
Manajemen Proyek: GitHub

Langkah langkah;

Clone Repository
git clone https://github.com/username/repo-name.git
cd repo-name

Buat Virtual Environment (Opsional)
python -m venv venv
source venv/bin/activate  # Untuk Windows: venv\Scripts\activate

Instal Dependensi;
pip install -r requirements.txt

requirements.txt;
numpy
sounddevice
soundfile
PyQt5
pyqtgraph
requests
joblib
librosa
tensorflow
scikit-learn

Penggunaan;
1. Jalankan Aplikasi
2. Antarmuka Pengguna
Parameters: Atur perangkat audio, amplitudo, laju sampling, frekuensi, dan interval pembaruan.
Plot Sinyal: Visualisasi sinyal asli dan DFT.
Analysis Results: Tampilkan hasil klasifikasi suara sebagai AI atau Human.
Training AI Models: Tambahkan file audio untuk melatih model Random Forest dan CNN.
Upload to Edge Impulse: Unggah file audio ke platform Edge Impulse untuk analisis lebih lanjut.
3. Langkah-Langkah Analisis
Rekam atau Muat Audio: Rekam audio langsung atau muat file audio yang sudah ada.
Visualisasi: Lihat sinyal asli dan DFT dari audio.
Klasifikasi: Klik "Analyze" untuk membedakan suara sebagai AI atau Human.
Lihat Hasil: Hasil klasifikasi akan ditampilkan dengan probabilitas di Progress Bar.
4. Melatih Model AI
Tambah File Audio: Klik "Add Human Audio Files" dan "Add AI Audio Files" untuk menambahkan dataset pelatihan.
Mulai Pelatihan: Klik "Train AI" untuk melatih model Random Forest dan CNN. Status pelatihan akan ditampilkan di antarmuka.

KONTAK;
Email: Shofiyur2015@gmail.com
LinkedIn: linkedin.com/in/Mochamad Shofiyur Rochman
GitHub: github.com/ShofiyurRochman

