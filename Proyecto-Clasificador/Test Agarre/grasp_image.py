'''
Code to run grasp detection given an image using the model learnt in
https://arxiv.org/pdf/1610.01685v1.pdf

Dependencies:   Python 2.7
                argparse ('pip install argparse')
                cv2 ('conda install -c menpo opencv=2.4.11' or install opencv from source)
                numpy ('pip install numpy' or 'conda install numpy')
                tensorflow (version 0.9)

Template run:
    python grasp_image.py --im {'Image path'} --model {'Model path'} --nbest {1,2,...} --nsamples {1,2,...} --gscale {0, ..., 1.0} --gpu {-1,0,1,...}
Example run:
    python grasp_image.py --im ./approach.jpg --model ./models/Grasp_model --nbest 100 --nsamples 250 --gscale 0.1 --gpu 0
'''
import argparse
import cv2
import numpy as np
from grasp_learner import grasp_obj
from grasp_predictor import Predictors
import time

puntos=[]

def drawRectangle(I, h, w, t, gsize=300):
	I_temp = I
	grasp_l = gsize/2.5
	grasp_w = gsize/5.0
	grasp_angle = t*(np.pi/18)-np.pi/2

	points = np.array([[-grasp_l, -grasp_w],
						[grasp_l, -grasp_w],
						[grasp_l, grasp_w],
						[-grasp_l, grasp_w]])
	R = np.array([[np.cos(grasp_angle), -np.sin(grasp_angle)],
				[np.sin(grasp_angle), np.cos(grasp_angle)]])
	rot_points = np.dot(R, points.transpose()).transpose()
	im_points = rot_points + np.array([w,h])
	cv2.line(I_temp, tuple(im_points[0].astype(int)), tuple(im_points[1].astype(int)), color=(0,255,0), thickness=5)
	cv2.line(I_temp, tuple(im_points[1].astype(int)), tuple(im_points[2].astype(int)), color=(0,0,255), thickness=5)
	cv2.line(I_temp, tuple(im_points[2].astype(int)), tuple(im_points[3].astype(int)), color=(0,255,0), thickness=5)
	cv2.line(I_temp, tuple(im_points[3].astype(int)), tuple(im_points[0].astype(int)), color=(0,0,255), thickness=5)
	print "puntos de la caja de grasp y angulo", im_points, grasp_angle
	for p in im_points:
		puntos.append(p)
	print len(im_points)
	return I_temp


#python grasp_image.py --im ./a5.jpg --model ./models/Grasp_model --nbest 5 --nsamples 250 --gscale 0.234 --gpu -1

def get_points():
	return puntos
	

def grasping(img):

	## Parse arguments
	
	##
	I=img
	model_path= 'models/Grasp_model'
	nsamples = 250
	nbest = 1
	gscale = 0.3
	imsize = max(I.shape[:2])
	gsize = int(gscale*imsize) # Size of grasp patch
	max_batchsize = 128
	gpu_id = -1

	## Set up model
	if nsamples > max_batchsize:
	    batchsize = max_batchsize
	    nbatches = int(nsamples/max_batchsize) + 1
	else:
	    batchsize = nsamples
	    nbatches = 1

	print('Loading grasp model')
	st_time = time.time()
	G = grasp_obj(model_path, gpu_id)
	G.BATCH_SIZE = batchsize
	G.test_init()
	P = Predictors(I, G)
	print('Time taken: {}s'.format(time.time()-st_time))

	fc8_predictions=[]
	patch_Hs = []
	patch_Ws = []

	print('Predicting on samples')
	st_time = time.time()
	for _ in range(nbatches):
	    P.graspNet_grasp(patch_size=gsize, num_samples=batchsize);
	    fc8_predictions.append(P.fc8_norm_vals)
	    patch_Hs.append(P.patch_hs)
	    patch_Ws.append(P.patch_ws)

	fc8_predictions = np.concatenate(fc8_predictions)
	patch_Hs = np.concatenate(patch_Hs)
	patch_Ws = np.concatenate(patch_Ws)

	r = np.sort(fc8_predictions, axis = None)
	r_no_keep = r[-nbest]
	print('Time taken: {}s'.format(time.time()-st_time))

	for pindex in range(fc8_predictions.shape[0]):
	    for tindex in range(fc8_predictions.shape[1]):
	        if fc8_predictions[pindex, tindex] < r_no_keep:
	            continue
	        else:
	            I = drawRectangle(I, patch_Hs[pindex], patch_Ws[pindex], tindex, gsize)

	print('displaying image')
	#cv2.imshow('image',I)
	cv2.imwrite('grasp.jpg',I)
	#cv2.waitKey(0)

	cv2.destroyAllWindows()
	G.test_close()
