import cv2,os

cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('face.xml')
kisi_id=0

while True:
    kisi_id += 1
    ad=input("kaydedilecek kisinin adini giriniz : ")
    os.mkdir("dataset/"+ad)
    i = 0

    while True:
        _, img =cam.read()
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        cv2.imshow("yayin",img)

        for(x,y,w,h) in faces:
            i=i+1
            cv2.imwrite("dataset/"+ad+"/"+ str(i) + ".jpg", gray[y:y + h , x :x + w])
            cv2.rectangle(img, (x-25, y-25), (x + w+25, y + h+25), (225, 0, 0), 2)
            cv2.imshow('resim', img[y-25 :y + h+25, x-25 :x + w+25])
            cv2.waitKey(100)
        if i>=20:
            break
    sor = input("Ekleme islemine devam etmek istiyor musunuz?...e/h... : ")
    if sor == "h":
        cam.release()
        cv2.destroyAllWindows()
        break