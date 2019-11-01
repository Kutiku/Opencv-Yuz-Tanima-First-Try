import cv2

recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('training/trainer.yml')
faceCascade = cv2.CascadeClassifier("face.xml")
path = 'yuzverileri'

cam = cv2.VideoCapture(0)

while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray,1.2,5)

    for(x,y,w,h) in faces:
        tahminEdilenKisi, conf = recognizer.predict(gray[y:y + h, x:x + w])
        cv2.rectangle(im,(x-25,y-25),(x+w+25,y+h+25),(225,0,0),2)


        if tahminEdilenKisi ==0:
            tahminEdilenKisi="Unknown"



        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fontColor = (255, 255, 255)
        cv2.putText(im, str(tahminEdilenKisi), (x, y + h), fontFace, fontScale, fontColor)
        cv2.imshow('im',im)

    if cv2.waitKey(1) & 0xFF==ord("g"):
        break
cam.release()
cv2.destroyAllWindows()