![Screenshot 2024-12-24 012908](https://github.com/user-attachments/assets/21e53d79-9e2a-4998-982f-aef2c9952a9b)

DEVELOPMENT OF SOFTWARE FOR DETECTING AI-GENERATED SYNTHETIC VOICES BASED ON MACHINE LEARNING CONVOLUTIONAL NEURAL NETWORK (CNN) AND RANDOM FOREST (RF)

With the rapid advancement of artificial intelligence (AI) technologies in voice processing, synthetic voices have become increasingly indistinguishable from human speech. This evolution poses significant challenges for voice-based authentication systems, as the heightened false acceptance rate threatens cybersecurity and undermines public trust in digital information. To address this issue, we have developed a software solution capable of analyzing and distinguishing between AI-generated and human voices with high precision.

Developers:

Muhammad Hadid Qushairi (2042231025)

Mochammad Shofiyur Rochman (2042231031)

Maulidan Arridlo (2042231059)

Wildan Rizki Auzay (2042231061)

Ahmad Radhy (Supervisor)


Features:

Intuitive User Interface: Developed using PyQt5 to ensure a responsive and user-friendly experience.
Audio Recording and Playback: Capable of real-time audio recording or loading existing audio files for analysis.
Signal Visualization: Real-time plotting of original audio signals and their Discrete Fourier Transform (DFT).
Voice Classification (AI vs. Human): Utilizes Random Forest and Convolutional Neural Network (CNN) models to accurately differentiate between AI-generated and human voices.
Model Training: Allows users to add audio files to train the Random Forest and CNN models as needed.
Edge Impulse Integration: Facilitates the uploading of audio files to Edge Impulse for enhanced analysis.
Technologies Used:

Programming Language: Python
GUI Framework: PyQt5
Audio Analysis Libraries: Librosa, SoundDevice, SoundFile
Machine Learning Frameworks: Scikit-learn (Random Forest), TensorFlow/Keras (CNN)
Visualization Tools: PyQtGraph
Project Management: GitHub
Installation Steps:


Clone the Repository:

bash
Copy code
git clone https://github.com/username/repo-name.git
cd repo-name
Create a Virtual Environment (Optional):

bash
Copy code
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
Install Dependencies:

bash
Copy code
pip install -r requirements.txt
requirements.txt:

Copy code
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
Usage Instructions:

Run the Application: Launch the application to access the user interface.


Configure Parameters:

Audio Devices: Select input and output devices.
Amplitude: Adjust the audio amplitude levels.
Sampling Rate: Set the sampling rate for audio processing.
Frequency: Define the frequency parameters.
Update Interval: Specify the intervals for updating visualizations.
Visualize Signals:

Original Signal: View the raw audio waveform.
Discrete Fourier Transform (DFT): Analyze the frequency components of the audio signal in real-time.
Classify Voice:

Click the "Analyze" button to classify the voice as either AI-generated or human.
View the classification results along with probability scores displayed on a progress bar.
Train AI Models:

Add Audio Files: Use the "Add Human Audio Files" and "Add AI Audio Files" options to include datasets for training.
Initiate Training: Click "Train AI" to commence training of the Random Forest and CNN models. Training status will be updated on the interface.
Upload to Edge Impulse:

Upload processed audio files to the Edge Impulse platform for further in-depth analysis.
Contact Information:

Email: Shofiyur2015@gmail.com
LinkedIn: linkedin.com/in/Mochamad-Shofiyur-Rochman
GitHub: github.com/ShofiyurRochman

This software leverages advanced machine learning techniques to enhance the security and reliability of voice-based authentication systems by effectively distinguishing between synthetic and human-generated voices. The integration of Convolutional Neural Networks and Random Forest algorithms ensures high accuracy and robustness, addressing critical challenges in the realm of cybersecurity and digital trust.
