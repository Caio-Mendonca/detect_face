import cv2

cam = cv2.VideoCapture(0)
detectFace = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognize = cv2.face.FisherFaceRecognizer.create()
recognize.read('classifaierFisher.yml')
width, height = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
while True:
    ret, frame = cam.read()
    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facesDetected = detectFace.detectMultiScale(imgGray, scaleFactor=1.5, minSize=(100,100))
    for (x, y, w, h) in facesDetected:
        imageFace = cv2.resize(imgGray[y:y+h, x:x+w], (width, height))
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        id, confidence = recognize.predict(imageFace)
        cv2.putText(frame, str(id), (x,y+(h+30)), font, 2, (0,0,255))
    cv2.imshow("frame", frame)
    
    cv2.waitKey(1)

cam.release()
cv2.destroyAllWindows()
