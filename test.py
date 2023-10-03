import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2
from keras.models import load_model

# Load the face cascade classifier
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Set a probability threshold
threshold = 0.90

# Initialize the video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Set the font for text overlay
font = cv2.FONT_HERSHEY_COMPLEX

# Load the trained model
model = load_model('MyTrainingModel.h5')

# Set the dimensions of the display window
cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Result", 800, 600)  # You can adjust the dimensions as needed

def preprocessing(img):
    img = img.astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

def get_className(classNo):
    if classNo == 0:
        return "Mask"
    elif classNo == 1:
        return "No Mask"

while True:
    success, imgOrignal = cap.read()
    faces = facedetect.detectMultiScale(imgOrignal, 1.3, 5)

    for i, (x, y, w, h) in enumerate(faces):
        crop_img = imgOrignal[y:y + h, x:x + w]
        img = cv2.resize(crop_img, (32, 32))
        img = preprocessing(img)
        img = img.reshape(1, 32, 32, 1)

        prediction = model.predict(img)
        class_index = np.argmax(prediction, axis=1)[0]
        probability_value = np.max(prediction)

        if probability_value > threshold:
            class_name = get_className(class_index)
            cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, str(class_name), (x, y - 10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Result", imgOrignal)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
