import cv2
import os
import numpy as np
from PIL import Image

detectorFace = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
reconhecedor = cv2.face.LBPHFaceRecognizer_create()
reconhecedor.read("classificadorLBPHYale.yml")

totalAcertos = 0
percentualAcerto = 0.0
totalConfianca = 0.0

caminhos = [os.path.join('pictures', f) for f in os.listdir('pictures')]
for caminhoImagem in caminhos:
    if not (caminhoImagem == 'pictures/.DS_Store'):           
        imagemFace = Image.open(caminhoImagem).convert('L')
        imagemFaceNP = np.array(imagemFace, 'uint8')
        facesDetectadas = detectorFace.detectMultiScale(imagemFaceNP)
        for (x, y, l, a) in facesDetectadas:
            idprevisto, confianca = reconhecedor.predict(imagemFaceNP)
            idatual = int(os.path.split(caminhoImagem)[-1].split(".")[1])
            print(str(idatual) + " foi classificado como " + str(idprevisto) + " - " + str(confianca))
            if idprevisto == idatual:
                totalAcertos += 1
                totalConfianca += confianca
percentualAcerto = (totalAcertos / 30) * 100
totalConfianca = totalConfianca / totalAcertos
print("Percentual de acerto: " + str(percentualAcerto))
print("Total confiança: " + str(totalConfianca))

