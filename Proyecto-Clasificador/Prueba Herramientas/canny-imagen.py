import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from math import hypot

####Guardado en Drive

#original = cv2.imread("imagenes/hh.jpg")
original = cv2.imread("a1.jpg")
cv2.imshow("original",original)

#cargar modelo entrenado
#Hasta el momento, los mejores modelos son: 
#8l-rmsprop-32-20epochs-2000img y 8l-rmsprop-32-25epochs-2000img
longitud, altura = 100,100
tamano_deseado = 100
modelo='./modelo/modelo-32b-25e-2000-corregido-rms.h5'
pesos='./modelo/pesos-32b-25e-2000-corregido-rms.h5'
cnn=load_model(modelo)
cnn.load_weights(pesos)

nombre=[]
i=0
###Funcion para que la imagen no pierda el aspect ratio al hacer resize,
### mantiene un cuadrado agregando franjas negras en los bordes
###***Problema: genera una imagen en escala de grises

def get_square(image,square_size):

    (height,width)=image.shape[:2]
    if(height>width):
      differ=height
    else:
      differ=width
    differ+=4

    mask = np.zeros((differ,differ), dtype="uint8")   
    x_pos=int((differ-width)/2)
    y_pos=int((differ-height)/2)
    mask[y_pos:y_pos+height,x_pos:x_pos+width]=image[0:height,0:width,1]
    mask=cv2.resize(mask,(square_size,square_size),interpolation=cv2.INTER_AREA)

    return mask 

###Funcion que hace resize manteniendo el aspect ratio
###***Problema: no genera un cuadrado de la imagen 
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def prediccion(file):
	#x=load_img(file,target_size=(longitud,altura))
	y=file
	print y.shape
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
		herramienta='Tijera'
	return herramienta
######

gray = cv2.cvtColor(original.copy(), cv2.COLOR_BGR2GRAY)

gauss = cv2.GaussianBlur(gray, (5,5), 0)

cv2.imshow("suavizado",gauss)

canny = cv2.Canny(gauss, 50, 150)

cv2.imshow("canny",canny)
#Morphologic, para unir contornos incompletos
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
dilated = cv2.dilate(canny, kernel)

cv2.imshow("imagen dilatada",dilated)
#(_, contornos, _) = cv2.findContours(canny.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
(contornos, _) = cv2.findContours(dilated.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


print("Se han encontrado {} objetos".format(len(contornos)))

#cv2.drawContours(original, contornos, -1, (0,0,255),2)
#cv2.drawContours(original,contornos,-1,(0,0,255),2)

####
"""for i in range(1,8):
	cv2.drawContours(original,contornos,i,(0,0,255),2)
	
	c=contornos[i]
	x,y,w,h=cv2.boundingRect(c)
	crop=original[y:y+h, x:x+w]
	cv2.rectangle(original,(x,y),(x+w,y+h),(0,255,0),2)
	
	###Dibuja rectangulo con area minima del contorno (con angulo)

	rect = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(rect)
	box = np.int0(box)
	cv2.drawContours(original,[box],0,(0,0,255),2)"""

#Predice los objetos detectados
(height,width)=original.shape[:2]
margen_img=20
imagenes=[]
for c in contornos:
	area=cv2.contourArea(c)
	x,y,w,h=cv2.boundingRect(c)
	area=w*h
	if  area<=50 :
		continue

	yi=y-margen_img
	yf=y+h+margen_img
	xi=x-margen_img
	xf=x+w+margen_img

	if yi<0:
		yi=0
	
	if xi<0:
		xi=0

	#crop=original[y-margen_img:y+h+margen_img, x-margen_img:x+w+margen_img]
	crop=original[yi:yf, xi:xf]
	cv2.imshow("crop",crop)

	##############
	##Codigo que permite hacer resize manteniendo el ratio
	##Hace lo mismo que la funcion get_square() pero en RGB

	desired_size=tamano_deseado

	old_size=crop.shape[:2]

	ratio=float(desired_size)/max(old_size)
	new_size = tuple([int(x*ratio) for x in old_size])
	im = cv2.resize(crop, (new_size[1], new_size[0]))

	delta_w = desired_size - new_size[1]
	delta_h = desired_size - new_size[0]
	top, bottom = delta_h//2, delta_h-(delta_h//2)
	left, right = delta_w//2, delta_w-(delta_w//2)

	color = [255, 255, 255]
	
	new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
	#new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_REPLICATE,value=color)
	imagenes.append(new_im)

	#cv2.imshow("imagen", new_im)
	#cv2.imwrite('new.jpg',new_im)

	#############

	#cv2.rectangle(original,(x,y),(x+w,y+h),(0,255,0),2)
	nombre.append(prediccion(new_im))
	#nombre.append(prediccion(crop))

	i=i+1
	#cv2.putText(original,nombre,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,np.array([0,255,0],dtype=np.uint8).tolist(),1,8)
	print nombre
##
j=0
for c in contornos:
	area=cv2.contourArea(c)
	x,y,w,h=cv2.boundingRect(c)
	

	area=w*h
	print area
	if area <=50 :
		continue
	
	cv2.rectangle(original,(x,y),(x+w,y+h),(0,255,0),2)
	cv2.putText(original,nombre[j],(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,np.array([0,0,0],dtype=np.uint8).tolist(),1,8)
	#
	rect = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(rect)
	box = np.int0(box)
	#
	print box
	cv2.drawContours(original,[box],0,(0,0,255),2)

	##Puntos rectangulo minimo, se toma como primer punto el que este mas abajo en el eje y
	##luego los siguientes se toman en el sentido de las agujas del reloj
	xb1,yb1=box[0][0],box[0][1]
	xb2,yb2=box[1][0],box[1][1]
	xb3,yb3=box[2][0],box[2][1]

	dist1=hypot(xb2-xb1,yb2-yb1)
	dist2=hypot(xb3-xb2,yb3-yb2)


	if dist1<=dist2:
		xbb1,ybb1=xb1,yb1
		xbb2,ybb2=xb2,yb2
		#punto medio en el lado mas largo
		xcl,ycl=(xb2+xb3)/2,(yb2+yb3)/2
		dist=dist1

	else:
		xbb1,ybb1=xb2,yb2
		xbb2,ybb2=xb3,yb3
		#punto medio en el lado mas largo
		xcl,ycl=(xb1+xb2)/2,(yb1+yb2)/2
		dist=dist2
		

	mm=(ybb2-ybb1)/(xbb2-xbb1)

	#Punto medio en el lado mas corto
	xcs=(xbb1+xbb2)/2
	ycs=(ybb1+ybb2)/2
	
	#Punto medio en el rectangulo de area minima
	xcentral=xcl+xcs-xb2
	ycentral=ycl+ycs-yb2

	cv2.circle(original,(xcentral,ycentral),6,(255,0,255),-1)


	#circulo en el lado mas corto
	cv2.circle(original,(xcs,ycs),6,(0,255,0),-1)

	#cv2.line(original,(xbm,ybm),(xc,yc),(0,255,0),5)

	print yb3

	j=j+1

	###Dibuja rectangulo con area minima del contorno
	"""rect = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(rect)
	box = np.int0(box)
	cv2.drawContours(original,[box],0,(0,0,255),2)"""


"""c = contornos[7]
x,y,w,h = cv2.boundingRect(c)
print x,y,w,h
crop = original[y:y+h, x:x+w]
#crop=canny[y:y+h, x:x+w]
cv2.imshow("recorte",crop)

cv2.rectangle(original,(x,y), (x+w,y+h), (0,255,0),2)
rect = cv2.minAreaRect(c)
box = cv2.cv.BoxPoints(rect)
box = np.int0(box)
cv2.drawContours(original,[box],0,(0,0,255),2)"""

#crop2 = cv2.cvtColor(crop.copy(), cv2.COLOR_GRAY2RGB)

"""for j in range(1,8):
	c=contornos[j]
	area=cv2.contourArea(c)
	if area<=10:
		continue
	x,y,w,h=cv2.boundingRect(c)
	
	crop=original[y:y+h, x:x+w]
	nombre=prediccion(crop)
	cv2.putText(original,nombre,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,np.array([0,255,255],dtype=np.uint8).tolist(),1,8)
"""

for i,nom in enumerate(nombre):
	cv2.imshow(str(i)+str(nom),imagenes[i])

cv2.imshow("Prediccion",original)

plt.imshow(original)
plt.show()

#prediccion(crop2)
#prediccion(crop)

cv2.waitKey(0)
