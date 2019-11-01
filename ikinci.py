import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
faceCascade = cv2.CascadeClassifier("face.xml")

ad = input("ogretilecek isim : ")
yol = 'yuzverileri'+ad

def get_images_and_labels(yol):
     foto_yollari = [os.path.join(yol, f) for f in os.listdir(yol)]
     fotolar = []
     etiketler = []
     for foto_yol in foto_yollari:
         image_pil = Image.open(foto_yol).convert('L')
         foto = np.array(image_pil, 'uint8')

         nbr = int(os.path.split(foto_yol)[1].split(".")[0].replace("face-", ""))
         print(nbr)

         yuzler = faceCascade.detectMultiScale(foto)
         for (x, y, w, h) in yuzler:
             fotolar.append(foto[y: y + h, x: x + w])
             etiketler.append(nbr)
             cv2.imshow("Ogretildi", foto[y: y + h, x: x + w])
             cv2.waitKey(10)
     return fotolar, etiketler


images, labels = get_images_and_labels(yol)
cv2.imshow('test',images[0])
cv2.waitKey(1)

recognizer.train(images, np.array(labels))
recognizer.write('training/trainer.yml')
cv2.destroyAllWindows()

