import numpy as np
import cv2
import glob
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

tamano_deseado=100
longitud, altura = 100,100

modelo='./modelo/modelo-32b-20e-2000.h5'
pesos='./modelo/pesos-32b-20e-2000.h5'
cnn=load_model(modelo)
cnn.load_weights(pesos)


def prediccion(file):
	#x=load_img(file,target_size=(longitud,altura))
	y=file
	#print y.shape
	x=cv2.resize(y, (longitud,altura))
	x=img_to_array(x)
	x=np.expand_dims(x, axis=0)
	#print(x)
	arreglo=cnn.predict(x) ##[[1,0,0]]
	
	resultado=arreglo[0]
	respuesta=np.argmax(resultado)
	

	if respuesta==0:
		herramienta='Alicate'
	elif respuesta==1:
		herramienta='Calculadora'
	elif respuesta==2:
		herramienta='Cuchillo'
	elif respuesta==3:
		herramienta='Destornillador'
	elif respuesta==4:
		herramienta='Gamepad'
	elif respuesta==5:
		herramienta='Llave'
	elif respuesta==6:
		herramienta='Martillo'
	elif respuesta==7:
		herramienta='Mouse'
	elif respuesta==8:
		herramienta='Reloj'
	elif respuesta==9:
		herramienta='Taladro'
	elif respuesta==10:
		herramienta='Telefono'
	return herramienta

######

def imgprocess(img):
	nombre=[]
	gray=cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
	gauss=cv2.GaussianBlur(gray,(5,5),0)

	canny=cv2.Canny(gauss,50,150)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	dilated = cv2.dilate(canny, kernel)

	(contornos,_)=cv2.findContours(dilated.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	margen_img=20
	imagenes=[]
	for c in contornos:
		area=cv2.contourArea(c)
		x,y,w,h=cv2.boundingRect(c)
		area=w*h
		if area<=50:
			continue

		yi=y-margen_img
		yf=y+h+margen_img
		xi=x-margen_img
		xf=x+w+margen_img

		if yi<0:
			yi=0
	
		if xi<0:
			xi=0

		crop=img[yi:yf, xi:xf]
		#cv2.imshow("crop",crop)

		desired_size=tamano_deseado

		old_size=crop.shape[:2]

		ratio=float(desired_size)/max(old_size)
		new_size= tuple([int(x*ratio)for x in old_size])
		im = cv2.resize(crop, (new_size[1],new_size[0]))

		delta_w = desired_size - new_size[1]
		delta_h = desired_size - new_size[0]
		top, bottom = delta_h//2, delta_h-(delta_h//2)
		left, right = delta_w//2, delta_w-(delta_w//2)

		color = [255,255,255]
		new_im = cv2.copyMakeBorder(im,top,bottom,left,right,cv2.BORDER_CONSTANT,value=color)
		imagenes.append(new_im)
		nombre.append(prediccion(new_im))

	j=0
	for c in contornos:
		area=cv2.contourArea(c)
		x,y,w,h=cv2.boundingRect(c)
		area=w*h
		if area <=50:
			continue
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
		cv2.putText(img,nombre[j],(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,np.array([0,0,0],dtype=np.uint8).tolist(),1,8)
		j=j+1

imagenes = [cv2.imread(file) for file in glob.glob('testing/*.jpg')]


for i,nom in enumerate(imagenes):
	imgprocess(imagenes[i])
	cv2.imshow(str(i),imagenes[i])



"""original1 = cv2.imread("a1.jpg")
imgprocess(original1)
cv2.imshow("original1",original1)

original2 = cv2.imread("a2.jpg")
imgprocess(original2)
cv2.imshow("original2",original2)

original3 = cv2.imread("a3.jpg")
imgprocess(original3)
cv2.imshow("original3",original3)

original4 = cv2.imread("a4.jpg")
imgprocess(original4)
cv2.imshow("original4",original4)

original5 = cv2.imread("a5.jpg")
imgprocess(original5)
cv2.imshow("original5",original5)

original6 = cv2.imread("a6.jpg")
imgprocess(original6)
cv2.imshow("original6",original6)

original7 = cv2.imread("a7.jpg")
imgprocess(original7)
cv2.imshow("original7",original7)

original8 = cv2.imread("a8.jpg")
imgprocess(original8)
cv2.imshow("original8",original8)

original9 = cv2.imread("a9.jpg")
imgprocess(original9)
cv2.imshow("original9",original9)"""

cv2.waitKey(0)
