import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import noisereduce as nr
import sounddevice as sd
import wave

# Audio recording functionality
def record_audio(filename, duration=5, sample_rate=44100):
    try:
        st.info("Recording started...")
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()  # Wait for the recording to complete
        st.success("Recording completed!")

        # Save audio data to a .wav file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)  # Mono audio
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        return True
    except Exception as e:
        st.error(f"An error occurred during recording: {e}")
        return False

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('svm_model2.pkl')

# Load the scaler
scaler = joblib.load("scaler.pkl")


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



def extract_features(filename):
    y, sr = clean_audio(filename)
    rmse = np.mean(librosa.feature.rms(y=y))
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=55), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr), axis=1)
    return np.hstack([mfccs, rmse, chroma, spectral_centroid, spectral_rolloff, zcr, tonnetz])

# Visualization: Waveform
def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

# Visualization: Spectrogram
def plot_spectrogram(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='viridis')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title("Mel Spectrogram")
    st.pyplot(fig)

# Main function
def main():
    st.title("Speech Emotion Recognition")
    st.write("Upload an audio file or record one to predict its emotion and analyze its properties.")

    # Audio recording
    if st.button("Record Audio"):
        temp_file = "recorded_audio.wav"
        if record_audio(temp_file, duration=5):
            st.audio(temp_file, format="audio/wav")
            y, sr = librosa.load(temp_file, sr=None)
            st.write("### Audio Metrics")
            st.write(f"- **Duration**: {librosa.get_duration(y=y, sr=sr):.2f} seconds")
            st.write(f"- **Sample Rate**: {sr} Hz")
            plot_waveform(y, sr)
            plot_spectrogram(y, sr)
            features = extract_features(temp_file)
            features = scaler.transform([features])
            st.write("Predicting emotion...")
            model = load_model()
            emotion = model.predict(features)[0]
            st.success(f"Detected Emotion: {emotion}")

    # File uploader
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])
    if uploaded_file is not None:
        temp_file = "uploaded_audio.wav"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.audio(temp_file, format="audio/wav")
        y, sr = librosa.load(temp_file, sr=None)
        st.write("### Audio Metrics")
        st.write(f"- **Duration**: {librosa.get_duration(y=y, sr=sr):.2f} seconds")
        st.write(f"- **Sample Rate**: {sr} Hz")
        plot_waveform(y, sr)
        plot_spectrogram(y, sr)
        features = extract_features(temp_file)
        features = scaler.transform([features])
        st.write("Predicting emotion...")
        model = load_model()
        emotion = model.predict(features)[0]
        st.success(f"Detected Emotion: {emotion}")

if __name__ == "__main__":
    main()
