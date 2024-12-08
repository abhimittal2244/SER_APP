import streamlit as st
import librosa
import numpy as np
import os
import tempfile
import joblib
import noisereduce as nr
import random
import sounddevice as sd
import wave


def record_audio(filename, duration=5, sample_rate=44100):
    """
    Records audio from the microphone and saves it as a .wav file.

    Args:
        filename (str): The name of the output .wav file.
        duration (int): Duration of the recording in seconds. Default is 5 seconds.
        sample_rate (int): Sample rate for the recording. Default is 44100 Hz.
    """
    try:
        print("Recording started...")
        # Record audio data
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()  # Wait for the recording to complete
        print("Recording completed.")

        # Save audio data to a .wav file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)  # Mono audio
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        print(f"Audio saved as {filename}")

    except Exception as e:
        print(f"An error occurred while recording: {e}")



# Load your trained model (placeholder for now)
@st.cache_resource
def load_model():
    return joblib.load('svm_model2.pkl')

def augment_audio(y, sr):
    augmented_audio = []

    # Original audio
    augmented_audio.append(y)

    # 1. Pitch Shifting
    y_pitch_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=3)  # Shift up
    y_pitch_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-3)  # Shift down
    augmented_audio.extend([y_pitch_up, y_pitch_down])

    # 2. Time Stretching
    y_speed_up = librosa.effects.time_stretch(y, rate=1.25)  # Speed up
    y_slow_down = librosa.effects.time_stretch(y, rate=0.75)  # Slow down
    augmented_audio.extend([y_speed_up, y_slow_down])

    # 3. Adding Gaussian Noise
    noise = np.random.normal(0, 0.005, len(y))  # Adjust noise level as needed
    y_noisy = y + noise
    augmented_audio.append(y_noisy)

    # 4. Shifting Audio
    shift_amount = int(0.2 * sr)  # Shift by 0.2 seconds
    y_shift_right = np.roll(y, shift_amount)  # Right shift
    y_shift_left = np.roll(y, -shift_amount)  # Left shift
    augmented_audio.extend([y_shift_right, y_shift_left])

    # Randomly select an augmented sample
    random.seed(67)  # Ensure reproducibility
    return random.choice(augmented_audio)



def clean_audio(filename, noise_sample_duration=1, top_db=15, trim_margin=0.2, pre_emphasis_coef=0.7, augment=False):
    y, sr = librosa.load(filename, sr=None)
    if augment:
        y = augment_audio(y, sr)
    y_preemphasized = np.append(y[0], y[1:] - pre_emphasis_coef * y[:-1])
    noise_sample = y_preemphasized[:int(sr * noise_sample_duration)]
    y_denoised = nr.reduce_noise(y=y_preemphasized, sr=sr, y_noise=noise_sample, prop_decrease=0.9)
    y_trimmed, _ = librosa.effects.trim(y_denoised, top_db=top_db)
    if len(y_trimmed) > int(2 * trim_margin * sr):
        y_trimmed = y_trimmed[int(trim_margin * sr):-int(trim_margin * sr)]
    target_length = sr * 5
    if len(y_trimmed) < target_length:
        y_trimmed = np.pad(y_trimmed, (0, target_length - len(y_trimmed)), mode='constant')
    elif len(y_trimmed) > target_length:
        y_trimmed = y_trimmed[:target_length]
    y_normalized = librosa.util.normalize(y_trimmed)
    return y_normalized, sr



def extract_features(filename, augment=False):
    y, sr = clean_audio(filename, augment=augment)
    rmse = np.mean(librosa.feature.rms(y=y, frame_length=2))
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=55), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=256, roll_percent=0.75))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y, n_fft=128), sr=sr), axis=1)
    return np.hstack([mfccs, rmse, chroma, spectral_centroid, spectral_rolloff, zcr, tonnetz])





def main():
    st.title("Speech Emotion Recognition")
    st.write("Upload an audio file or record one to predict its emotion.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

    # Record audio
    if st.button("Record Audio"):
        st.warning("Recording functionality is not implemented in this version.")

    # Process the uploaded file
    if uploaded_file is not None:
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.audio("temp_audio.wav", format="audio/wav")

        st.write("Preprocessing audio...")

        # Feature extraction
        features = extract_features("temp_audio.wav", augment=True)
        # Predict
        st.write("Predicting emotion...")
        model = load_model()
        emotion = model.predict(features)
        st.success(f"Detected Emotion: {emotion}")

if __name__ == "__main__":
    main()
    record_audio("output.wav", duration=5)