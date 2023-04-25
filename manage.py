import cv2 
import numpy as np
classifaier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
classifaierEyes = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
cam = cv2.VideoCapture(0)
sample = 1
sampleNum = 25
id = input("Digite seu Nick Name:")
width, height = 220, 220
print("Capturando as faces...")
while True:
    ret, frame = cam.read()
    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifaier.detectMultiScale(grayImg, scaleFactor=1.5, minSize=(100,100))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        region = frame[y:y+h, x:x+w] # Region of interest
        regionGray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        eyes = classifaierEyes.detectMultiScale(regionGray)
        for (ox, oy, ow, oh) in eyes:
            cv2.circle(region, (ox + int(ow/2), oy + int(oh/2)), 10, (0,0,255), -1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                if np.average(grayImg) > 110:
                    imageFace = cv2.resize( grayImg[y:y+h, x:x+w], (width, height))
                    cv2.imwrite("pictures/pessoa." + str(id) + "." + str(sample) + ".jpg", imageFace)
                    print("Foto " + str(sample) + " capturada com sucesso!")
                    sample += 1

    cv2.imshow("frame", frame)
    cv2.waitKey(1)
    if sample >= sampleNum + 1:
        break
print("Faces capturadas com sucesso!")
cam.release()
cv2.destroyAllWindows()