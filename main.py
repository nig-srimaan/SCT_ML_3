import os
import zipfile
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

if os.path.exists('dataset.zip'):
    with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
        zip_ref.extractall('dataset')

def load_images(data_dir, image_size=(64, 64)):
    images = []
    labels = []
    categories = ['cats', 'dogs']
    
    for category in categories:
        path = os.path.join(data_dir, category)
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

train_dir = 'dataset/training_set' if os.path.exists('dataset/training_set') else 'training_set'
test_dir = 'dataset/test_set' if os.path.exists('dataset/test_set') else 'test_set'

X_train, y_train = load_images(train_dir)
X_test, y_test = load_images(test_dir)

X_train = X_train / 255.0
X_test = X_test / 255.0

model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Cats', 'Dogs']))
