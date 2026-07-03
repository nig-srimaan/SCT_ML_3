import os
import zipfile
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

if os.path.exists('dataset.zip'):
    with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
        zip_ref.extractall('extracted_data')

data_dir = 'extracted_data/PetImages'
if not os.path.exists(data_dir):
    for root, dirs, files in os.walk('extracted_data'):
        if 'PetImages' in dirs:
            data_dir = os.path.join(root, 'PetImages')
            break
if not os.path.exists(data_dir) and os.path.exists('PetImages'):
    data_dir = 'PetImages'

def load_images(base_dir, image_size=(64, 64)):
    images = []
    labels = []
    categories = ['Cat', 'Dog']
    
    for category in categories:
        path = os.path.join(base_dir, category)
        if not os.path.exists(path):
            continue
        class_num = categories.index(category)
        
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, image_size)
                    images.append(img.flatten())
                    labels.append(class_num)
            except Exception:
                pass
                
    return np.array(images), np.array(labels)

X, y = load_images(data_dir)

X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Cats', 'Dogs']))
