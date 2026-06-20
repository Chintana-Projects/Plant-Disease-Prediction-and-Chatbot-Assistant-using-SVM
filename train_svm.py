import os
import cv2
import numpy as np
import joblib

from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

# ==========================
# Dataset Path
# ==========================
dataset_path = r"C:\Users\DEll\.vscode\plant disease prediction advanced\dataset1"

data = []
labels = []

print("Loading dataset...")

# ==========================
# Read Dataset
# ==========================
for folder in sorted(os.listdir(dataset_path)):

    folder_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(folder_path):
        continue

    image_count = 0

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        # Resize image
        image = cv2.resize(image, (160, 160))

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Extract HOG Features
        features = hog(
            gray,
            orientations=12,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            transform_sqrt=True,
            feature_vector=True
        )

        data.append(features)
        labels.append(folder)
        image_count += 1

    print(f"{folder} -> {image_count} images")

# ==========================
# Convert to NumPy Arrays
# ==========================
data = np.array(data)
labels = np.array(labels)

# Shuffle Dataset
data, labels = shuffle(data, labels, random_state=42)

print("\n=========================")
print("Dataset Statistics")
print("=========================")
print("Total Images :", len(labels))
print("Total Classes:", len(np.unique(labels)))
print("\nImages per class:")
print(Counter(labels))

# ==========================
# Train-Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

# ==========================
# Build SVM Model
# ==========================
model = make_pipeline(
    StandardScaler(),
    SVC(
        kernel='rbf',
        C=20,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    )
)

print("\nTraining SVM...")
model.fit(X_train, y_train)

# ==========================
# Prediction
# ==========================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n=========================")
print(f"Accuracy : {accuracy * 100:.2f}%")
print("=========================")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==========================
# Save Model
# ==========================
joblib.dump(model, "svm_model.pkl")

# Save Class Names
class_names = sorted(np.unique(labels))
joblib.dump(class_names, "label_names.pkl")

print("\nModel saved as svm_model.pkl")
print("Labels saved as label_names.pkl")