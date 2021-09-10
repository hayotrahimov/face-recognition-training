from os import listdir
from os.path import isfile, join

import cv2
import cv2 as cv
import numpy as np

faces_dir = 'faces'
files = [f for f in listdir(faces_dir) if isfile(join(faces_dir, f))]

# training
training_data, labels = [], []

for i, file_path in enumerate(files):
    image_path = f"{faces_dir}/{file_path}"
    print(image_path)
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    training_data.append(np.array(image, dtype=np.uint8))
    labels.append(i)
labels = np.asarray(labels, dtype=np.int32)
model = cv.face.LBPHFaceRecognizer_create()
model.train(np.asarray(training_data), np.asarray(labels))
print("Model training completed.")

cascPath = "cascades/haarcascade_frontalface_default.xml"
face_classifier = cv.CascadeClassifier(cascPath)


def face_detector(img, size=0.5):
    gray = cv.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3)

    if faces is ():
        return img, []
    rois = list()
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv.resize(roi, (200, 200))
        rois.append(roi)
    return img, rois


cap = cv.VideoCapture(0)
while True:
    success, img = cap.read()
    img, faces = face_detector(img)
    try:
        faces = [cv.cvtColor(face, cv.COLOR_BGR2GRAY) for face in faces]
        for face in faces:
            result = model.predict(face)
            if result[1] > 500:
                confidence = int(100 * (1 - (result[1])/300))
                display_string = str(confidence) + '% confidence'
                cv.putText(img, display_string, (100, 120), cv.FONT_HERSHEY_SIMPLEX, 1, (250,120,122), 2)
                if confidence > 75:
                    cv.putText(img, "unlocked", (250, 120), cv.FONT_HERSHEY_SIMPLEX, 1, (250, 120, 122), 2)
        cv.imshow("face", img)
    except:
        cv.putText(img, "Face Not Found", (250, 120), cv.FONT_HERSHEY_SIMPLEX, 1, (250, 120, 122), 2)
        cv.imshow("test", img)
    if cv.waitKey(1) == 13:
        break
cap.release()
cv.closeAllWindows()