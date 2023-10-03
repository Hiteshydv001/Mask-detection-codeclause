import cv2

video = cv2.VideoCapture(0)

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

count = 0

while True:
    ret, frame = video.read()
    faces = facedetect.detectMultiScale(frame, 1.3, 5)

    for x, y, w, h in faces:
        count = count + 1
        name = './images/face_with_mask/' + str(count) + '.jpg'
        print("Creating Images........." + name)
        cv2.imwrite(name, frame[y:y + h, x:x + w])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Resize the frame to a smaller size
    frame = cv2.resize(frame, (800, 600))

    cv2.imshow("WindowFrame", frame)
    cv2.waitKey(1)

    if count > 100:
        break

video.release()
cv2.destroyAllWindows()
9