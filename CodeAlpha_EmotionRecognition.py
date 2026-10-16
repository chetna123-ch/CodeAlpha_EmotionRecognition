import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("="*60)
print("EMOTION RECOGNITION FROM SPEECH - CODEALPHA TASK 2")
print("="*60)

# ---------------------------------------------------------
# Step 1: Feature Extraction Function (MFCCs + Spectral Features)
# ---------------------------------------------------------
def extract_audio_features(signal, sr=22050, n_mfcc=40):
    """
    Extracts key speech signal features:
    - MFCCs (Mel-Frequency Cepstral Coefficients)
    - Chroma STFT (pitch/harmonic structure)
    - Spectral Contrast (texture/frequency variation)
    """
    # 1. MFCC Features
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    mfcc_scaled = np.mean(mfccs.T, axis=0)
    
    # 2. Chroma Feature
    stft = np.abs(librosa.stft(signal))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    
    # 3. Spectral Contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
    
    # Combine all feature vectors
    return np.hstack([mfcc_scaled, chroma, contrast])


# ---------------------------------------------------------
# Step 2: Synthetic Audio Signal Generator (For Demo / Benchmark)
# ---------------------------------------------------------
def generate_synthetic_speech_dataset(n_samples=500, sr=22050, duration=2.5):
    """
    Generates synthetic speech-like audio signals with distinct pitch/frequency 
    modulations representing different emotional states (Happy, Sad, Angry, Neutral, Fearful).
    """
    print("\n[INFO] Generating synthetic audio dataset for benchmarking...")
    emotions = ['happy', 'sad', 'angry', 'neutral', 'fearful']
    features_list = []
    labels_list = []
    
    t = np.linspace(0, duration, int(sr * duration))
    
    for i in range(n_samples):
        emotion = np.random.choice(emotions)
        
        # Base frequency variations per emotion
        if emotion == 'happy':
            freq = np.random.uniform(220, 350)
            noise_level = 0.05
        elif emotion == 'sad':
            freq = np.random.uniform(100, 180)
            noise_level = 0.02
        elif emotion == 'angry':
            freq = np.random.uniform(300, 450)
            noise_level = 0.15
        elif emotion == 'neutral':
            freq = np.random.uniform(150, 220)
            noise_level = 0.03
        else: # fearful
            freq = np.random.uniform(250, 400) + 20 * np.sin(2 * np.pi * 10 * t) # pitch jitter
            noise_level = 0.08
            
        # Synthesize harmonic wave + noise
        signal = np.sin(2 * np.pi * freq * t) + noise_level * np.random.randn(len(t))
        
        # Extract features
        feat = extract_audio_features(signal, sr=sr)
        features_list.append(feat)
        labels_list.append(emotion)
        
    return np.array(features_list), np.array(labels_list)


# Generate simulated dataset (Replace with RAVDESS/TESS dataset loading in production)
X, y = generate_synthetic_speech_dataset(n_samples=600)

print(f"[INFO] Dataset shape: Features={X.shape}, Target={y.shape}")

# ---------------------------------------------------------
# Step 3: Data Preprocessing
# ---------------------------------------------------------
# Encode string labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------
# Step 4: Model Training & Evaluation
# ---------------------------------------------------------
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
}

print("\n[INFO] Training classification models...\n")

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"================ {name} ================")
    print(f"Overall Accuracy: {acc*100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))