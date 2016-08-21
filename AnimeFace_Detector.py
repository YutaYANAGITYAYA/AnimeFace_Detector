from PIL import ImageGrab
import cv2
import numpy as np


path = "lbpcascade_animeface.xml"
cascade = cv2.CascadeClassifier(path)

while (cv2.waitKey(1) != 27):
  img = ImageGrab.grab()
  img = np.asarray(img)
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (100, 100))
  
  if len(faces) > 0:
    for (x,y,w,h) in faces:
      cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255), 2)
    
  img = cv2.resize(img, (img.shape[1]/2,img.shape[0]/2))
  cv2.imshow("res", img)

cv2.destroyAllWindows()
