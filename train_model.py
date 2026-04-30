import numpy as np
import os
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

NUM_FEATURES = 44
N_SAMPLES = 10000

def generate_large_dataset():
    print(f"Generating massive combinatorial dataset with {N_SAMPLES} samples...")
    X = []
    y = []
    
    np.random.seed(42)
    
    # 12-bin Chroma templates
    templates = {
        'Yaman':   np.array([1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1], dtype=float),
        'Bhupali': np.array([1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0], dtype=float),
        'Bhairav': np.array([1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1], dtype=float),
        'Durga':   np.array([1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0], dtype=float),
        'Kafi':    np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0], dtype=float)
    }
    
    # Normalize templates
    for k in templates:
        templates[k] /= np.linalg.norm(templates[k])
        
    for i in range(N_SAMPLES):
        # Generate a random dense chroma vector representing a real messy song
        chroma = np.random.rand(12)
        
        # Randomly boost some notes to simulate a key/melody
        boost_idx = np.random.choice(12, 3, replace=False)
        chroma[boost_idx] += np.random.rand(3) * 2
        
        # Make pitch invariant by rolling the max value to index 0
        max_idx = np.argmax(chroma)
        chroma = np.roll(chroma, -max_idx)
        
        chroma /= np.linalg.norm(chroma)
        
        # Find closest raga mathematically using Cosine Similarity
        best_raga = None
        best_sim = -1
        for raga, temp in templates.items():
            sim = np.dot(chroma, temp)
            if sim > best_sim:
                best_sim = sim
                best_raga = raga
                
        # Create full 44 feature vector
        sample = np.zeros(NUM_FEATURES)
        sample[0:26] = np.random.randn(26) * 50 # Noise for MFCC
        sample[26:38] = chroma
        sample[38:44] = np.random.randn(6) * 1000 # Noise for Spectral
        
        X.append(sample)
        y.append(best_raga)
        
    return np.array(X), np.array(y)

def train():
    X, y = generate_large_dataset()
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training SVM model with RBF kernel...")
    clf = SVC(kernel='rbf', probability=True, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc * 100:.2f}%")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, 'models/raga_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(le, 'models/label_encoder.joblib')
    print("Model saved to models/ directory.")

if __name__ == "__main__":
    train()
