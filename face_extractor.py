import cv2 as cv
cascPath = "cascades/haarcascade_frontalface_default.xml"
face_classifier = cv.CascadeClassifier(cascPath)


def face_extractor(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces == ():
        return False
    cropped_faces = list()
    for (x, y, w, h) in faces:
        cropped_faces.append(img[y:y + h, x:x + w])
    return cropped_faces


cap = cv.VideoCapture(0)
count = 0
while True:
    success, img = cap.read()
    extracted_faces = face_extractor(img)
    if extracted_faces:
        count += 1
        for extracted_face in extracted_faces:
            face = cv.resize(extracted_face, (200, 200))
            face = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
            file_name_path = f"faces/user_{count}.jpg"
            cv.imwrite(filename=file_name_path, img=face)
            cv.putText(face, str(count), (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv.imshow("Cropped Face", face)
        if cv.waitKey(1) == 13 or count == 150:
            break
cap.release()
cv.destroyAllWindows()