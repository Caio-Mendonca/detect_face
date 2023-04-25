import cv2
import numpy as np
import os

eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()


def getImageWithId():
    paths = [os.path.join('pictures', f) for f in os.listdir('pictures')]
    faces = []
    ids = []
    for image in paths:
        if not (image == 'pictures/.DS_Store'):           
            print('Loading image: ', image)
            faceImg = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)
            id = int(os.path.split(image)[-1].split('.')[1])
            ids.append(id)
            faces.append(faceImg)
            cv2.imshow("Face", faceImg)
            cv2.waitKey(10)
    return np.array(ids), faces
ids, faces = getImageWithId()
print("Treinando...")
eigenface.train(faces, ids)
eigenface.write('classifaierEigen.yml')

fisherface.train(faces, ids)
fisherface.write('classifaierFisher.yml')

lbph.train(faces, ids)
lbph.write('classifaierLBPH.yml')

print("Treinanmento realizado com sucesso!")
