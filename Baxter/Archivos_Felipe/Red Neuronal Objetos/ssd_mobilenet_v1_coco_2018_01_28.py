import numpy as np
import cv2
import os
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util
import rospy
import cv2,cv_bridge,cv
from sensor_msgs.msg import Image
import baxter_interface
from traeImagen import *
from baxter_core_msgs.srv import *
import std_srvs.srv
from baxter_core_msgs.msg import CameraSettings,EndpointState,NavigatorState
from MueveBaxter import *

x_ini=None
y_ini=None
z_ini=None
button2=None    
button1=None
button0=None

def on_state(msg):  
	global button1,button0,button2
	button1=msg.buttons[1]
	button0=msg.buttons[0]
	button2=msg.buttons[2]
      
def endpoint_callback(msg):
	global x_ini,y_ini,z_ini
	x_ini=msg.pose.position.x
	y_ini=msg.pose.position.y
	z_ini=msg.pose.position.z

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
#ssd_mobilenet_v1_coco_2017_11_17
#ssd_mobilenet_v1_coco_2018_01_28

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


img=Imagen()

def camera_callback(data, camera_name):
        # Convert image from a ROS image message to a CV image
	global img
        try:
            img.setImg(cv_bridge.CvBridge().imgmsg_to_cv2(data))
        except cv_bridge.CvBridgeError, e:
            print e

        # 3ms wait
        cv.WaitKey(3)

# left camera call back function
def left_camera_callback(data):
        camera_callback(data, "Left Hand Camera")

# right camera call back function
def right_camera_callback(data):
        camera_callback(data, "Right Hand Camera")

# head camera call back function
def head_camera_callback(data):
        camera_callback(data, "Head Camera")

def subscribe_to_camera(camera):
        if camera == "left":
            camera_str = "/cameras/left_hand_camera/image"
	    camera_sub = rospy.Subscriber(camera_str, Image, left_camera_callback)	
        elif camera == "right":
            camera_str = "/cameras/right_hand_camera/image"
	    camera_sub = rospy.Subscriber(camera_str, Image, right_camera_callback)
        elif camera == "head":
            camera_str = "/cameras/head_camera/image"
	    camera_sub = rospy.Subscriber(camera_str, Image, head_camera_callback)
        else:
            sys.exit("ERROR - subscribe_to_camera - Invalid camera")

def open_camera(camera, x_res, y_res):
        if camera == "left":
            cam = baxter_interface.camera.CameraController("left_hand_camera")
        elif camera == "right":
            cam = baxter_interface.camera.CameraController("right_hand_camera")
        elif camera == "head":
            cam = baxter_interface.camera.CameraController("head_camera")
        else:
            sys.exit("ERROR - open_camera - Invalid camera")

        # close camera
        #cam.close()

        # set camera parameters
        cam.resolution          = int(x_res), int(y_res)
        cam.exposure            = -1             # range, 0-100 auto = -1
        cam.gain                = -1             # range, 0-79 auto = -1
        cam.white_balance_blue  = -1             # range 0-4095, auto = -1
        cam.white_balance_green = -1             # range 0-4095, auto = -1
        cam.white_balance_red   = -1             # range 0-4095, auto = -1

        # open camera
        cam.open()

def reset_cameras():
        reset_srv = rospy.ServiceProxy('cameras/reset', std_srvs.srv.Empty)
        rospy.wait_for_service('cameras/reset', timeout=10)
        reset_srv()

def close_camera(camera):
        if camera == "left":
            cam = baxter_interface.camera.CameraController("left_hand_camera")
        elif camera == "right":
            cam = baxter_interface.camera.CameraController("right_hand_camera")
        elif camera == "head":
            cam = baxter_interface.camera.CameraController("head_camera")
        else:
            sys.exit("ERROR - close_camera - Invalid camera")
        # close camera
        cam.close()



def main():
	Posiciones=[] # Guardo las posiciones iniciales de configuracion
	rospy.init_node("Red_Neuronal")
	reset_cameras()
	close_camera("left")
        close_camera("right")
        close_camera("head")
	#Instrucciones brazo
	open_camera("left", 1280, 800)
	subscribe_to_camera("left") 
	endpoint_sub = rospy.Subscriber('/robot/limb/left/endpoint_state',EndpointState,callback=endpoint_callback) 
	screen_pub = rospy.Publisher('/robot/xdisplay', Image, queue_size=10)
	state_sub = rospy.Subscriber('robot/navigators/left_navigator/state',NavigatorState,callback=on_state)
	#Fin instrucciones brazo
	print 'Ahora graba posiciones'
	# Posiciones
	########3########
	#################
	#2######0#######1
	#################
	########4########
	#
	while 1:
		if button0:
			Posiciones.append([x_ini,y_ini])
			while button0:
				continue
		if button1 or button2:
			break
	print 'posiciones grabadas'
	Muevee = Mueve('left',Posiciones)  # En este punto debo pasar todas las posiciones iniciales
	with detection_graph.as_default():
	    with tf.Session() as sess:
		# Get handles to input and output tensors
		ops = tf.get_default_graph().get_operations()
		all_tensor_names = {output.name for op in ops for output in op.outputs}
		tensor_dict = {}
		for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
		    tensor_name = key + ':0'
		    if tensor_name in all_tensor_names:
		        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
		if 'detection_masks' in tensor_dict:
		        # The following processing is only for single image
		        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
		        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
		        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
		        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
		        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
		        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
		        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
		            detection_masks, detection_boxes, frame.shape[0], frame.shape[1])
		        detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
		        # Follow the convention by adding back the batch dimension
		        tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
		image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
		while(True):
		        if img.getImg() is None:
				continue
		      	frame = img.getImg()
			#print frame.shape
			c1, c2, c3, c4 = cv2.split(frame)
			frame= cv2.merge([c1,c2,c3])
		        # Run inference
		        output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(frame, 0)})

		        # all outputs are float32 numpy arrays, so convert types as appropriate
		        output_dict['num_detections'] = int(output_dict['num_detections'][0])
		        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
		        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
		        output_dict['detection_scores'] = output_dict['detection_scores'][0]
		        if 'detection_masks' in output_dict:
		            output_dict['detection_masks'] = output_dict['detection_masks'][0]       	
		        # Visualization of the results of a detection.
		        vis_util.visualize_boxes_and_labels_on_image_array(
		          frame,
		          output_dict['detection_boxes'],
		          output_dict['detection_classes'],
		          output_dict['detection_scores'],
		          category_index,
		          instance_masks=output_dict.get('detection_masks'),
		          use_normalized_coordinates=True,
		          line_thickness=8)

		        # Display the resulting frame
		        cv2.imshow('frame',frame)
			msgsub = cv_bridge.CvBridge().cv2_to_imgmsg(frame, encoding="8UC3")
			screen_pub.publish(msgsub)
			libros=[] 
			clase_index=77 # La clase escogida para tomarla con Baxter
			print output_dict['num_detections']
			print output_dict['detection_classes']
			for i in range(output_dict['num_detections']):
				if output_dict['detection_classes'][i]==clase_index:
					libros.append(i)
					print 'celular detectado'
			for i in libros:
				xmin=int(output_dict['detection_boxes'][i][1]*1280)
				xmax=int(output_dict['detection_boxes'][i][3]*1280)
				ymin=int(output_dict['detection_boxes'][i][0]*800)
				ymax=int(output_dict['detection_boxes'][i][2]*800)
				print xmin,xmax,ymin,ymax
				Muevee.CA(int((xmax+xmin)/2),int((ymax+ymin)/2),'Celulares')
			Muevee.randomm()
			Muevee.mover_baxter('base',Muevee.pose[:3],Muevee.pose[3:6])
		        if cv2.waitKey(1) & 0xFF == ord('q'):
		            break

		# When everything done, release the capture
	        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
