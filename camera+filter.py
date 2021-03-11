import numpy as np
import cv2
import time
import datetime
from PIL import Image

# Face recognition
def getface(img):
    # face
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # eye
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    # Binary, Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        # draw border
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # add filter
        img = filter(img,x,y,w,h)
    return img

def filter(img,x,y,w,h):
    im = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    # sticker
    mark=Image.open("hat2.png")
    height = int(w*987/1024)
    mark = mark.resize((w, height))
    layer=Image.new('RGBA', im.size, (0,0,0,0))
    layer.paste(mark, (x,y-height))
    out=Image.composite(layer,im,layer)
    img = cv2.cvtColor(np.asarray(out),cv2.COLOR_RGB2BGR)
    return img


cap = cv2.VideoCapture(0)
videoWriter = cv2.VideoWriter('testwrite.avi', cv2.VideoWriter_fourcc(*'MJPG'), 15, (1000,563))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # resize pic
        img = cv2.resize(frame,(1000,563))
        # recognize
        img = getface(img)
        # show videl
        cv2.imshow('frame',img)
        # save videl
        videoWriter.write(img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("END")
            break
    else:
        break

cap.release()
videoWriter.release()
cv2.destroyAllWindows()