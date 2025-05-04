import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

print("\n--- SVM Cat vs Dog Classification (Using Placeholder Data) ---")
print("NOTE: This uses dummy data. Real implementation requires image loading/processing.")
print("Dataset: https://www.kaggle.com/c/dogs-vs-cats/data\n")

num_samples = 200
image_size = 64 * 64

X_dummy = np.random.rand(num_samples, image_size) 


y_dummy = np.random.randint(0, 2, num_samples)
labels = ['cat', 'dog']


X_train, X_test, y_train, y_test = train_test_split(X_dummy, y_dummy, test_size=0.25, random_state=42, stratify=y_dummy)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


svm_model = SVC(kernel='rbf', C=1.0, random_state=42, probability=True) 

print("Training SVM model...")
svm_model.fit(X_train_scaled, y_train)
print("Training complete.")


y_pred = svm_model.predict(X_test_scaled)


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=labels)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\nModel Evaluation (on dummy data):")
print(f"  Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)
print("\nConfusion Matrix:")
print(conf_matrix)


plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix - SVM (Dummy Data)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
print("\nShowing Confusion Matrix Visualization...")
plt.show()
