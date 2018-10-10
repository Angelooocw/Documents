import numpy as np
import cv2

#original = cv2.imread("imagenes/hh.jpg")
original = cv2.imread("p1.png")
cv2.imshow("original",original)

gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

gauss = cv2.GaussianBlur(gray, (5,5), 0)

cv2.imshow("suavizado",gauss)

canny = cv2.Canny(gauss, 50, 200)

cv2.imshow("canny",canny)

#(_, contornos, _) = cv2.findContours(canny.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
(contornos, _) = cv2.findContours(canny.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


print("Se han encontrado {} objetos".format(len(contornos)))

cv2.drawContours(original, contornos, -1, (0,0,255),2)


c = contornos[5]
x,y,w,h = cv2.boundingRect(c)

crop = original[y:y+h, x:x+w]
cv2.imshow("recorte",crop)

cv2.rectangle(original,(x,y), (x+w,y+h), (0,255,0),2)
rect = cv2.minAreaRect(c)
box = cv2.cv.BoxPoints(rect)
box = np.int0(box)
cv2.drawContours(original,[box],0,(0,0,255),2)


cv2.imshow("contornos",original)
cv2.waitKey(0)
