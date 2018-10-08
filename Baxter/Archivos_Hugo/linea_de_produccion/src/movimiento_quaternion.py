#!/usr/bin/env 

import baxter_interface
import rospy
from geometry_msgs.msg import Pose, PoseStamped
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
import tf

class movimiento_caja():
	def __init__(self, arm):
		self.boton_brazo = baxter_interface.Navigator(arm)
		self.brazo = baxter_interface.Limb(arm)
		self.pos_ini = PoseStamped()
		self.pos_fin = PoseStamped()
		self.arm = arm

	def endpoint_a_pose_stamped(self, baxter_endpoint, frame):
		'''
		Transforma un msg del tipo endpoint_pose definido por la API de baxter_interface
		en un msg PoseStamped
		'''
		pos = PoseStamped()
		pos.header.stamp = rospy.Time.now()
		pos.header.frame_id = frame
		pos.pose.position.x = baxter_endpoint.endpoint_pose()['position'][0]
		pos.pose.position.y = baxter_endpoint.endpoint_pose()['position'][1]
		pos.pose.position.z = baxter_endpoint.endpoint_pose()['position'][2]
		pos.pose.orientation.x = baxter_endpoint.endpoint_pose()['orientation'][0]
		pos.pose.orientation.y = baxter_endpoint.endpoint_pose()['orientation'][1]
		pos.pose.orientation.z = baxter_endpoint.endpoint_pose()['orientation'][2]
		pos.pose.orientation.w = baxter_endpoint.endpoint_pose()['orientation'][3]
		return pos

	def matriz_a_pose_stamped(self, T, frame):
		'''
		Transforma una matriz homogenenea en un msg del tipo pose_stamped
		'''
	    t = PoseStamped()
	    t.header.frame_id = frame
	    t.header.stamp = rospy.Time.now()
	    translacion = tf.transformations.translation_from_matrix(T)
	    orientacion = tf.transformations.quaternion_from_matrix(T)
	    t.pose.position.x = translacion[0]
	    t.pose.position.y = translacion[1]
	    t.pose.position.z = translacion[2]
	    t.pose.orientation.x = orientacion[0]
	    t.pose.orientation.y = orientacion[1]
	    t.pose.orientation.z = orientacion[2]
	    t.pose.orientation.w = orientacion[3]        
	    return t

	def definir_posicion(self):
		'''
		Guarda las posiciones iniciales y finales.
		'''
		print "Presione el boton de navegacion para guardar posicion inicial"
		while True:
			if self.boton_brazo.button0 == 1:
				print "Guardando..."
				self.pos_ini = endpoint_a_pose_stamped(self.brazo.endpoint_pose(),'/base')
				break

		rospy.sleep(1)
		print "Presione el boton de navegacion para guardar posicion final"
		while True:
			if self.boton_brazo.button0 == 1:
				print "Guardando..."
				self.pos_fin = endpoint_a_pose_stamped(self.brazo.endpoint_pose(),'/base')
				break


	def mover_baxter(self, source_frame, trans, rot, ejecutar):

	    nombre_servicio = '/ExternalTools/'+ self.arm +'/PositionKinematicsNode/IKService'
	    servicio_ik = rospy.ServiceProxy(nombre_servicio,SolvePositionIK)
	    frame = source_frame   

	    # Promedio de velocidad del brazo
	    self.brazo.set_joint_position_speed(0.5)
	    
	    matrix = tf.transformations.concatenate_matrices(tf.transformations.translation_matrix([trans.x,trans.y,trans.z]),
	    	tf.transformations.quaternion_matrix([rot.x,rot.y,rot.z,rot.w])
	        )
	    
	    rospy.wait_for_service(nombre_servicio,10)
	    ik_mensaje = SolvePositionIKRequest()
	    ik_mensaje.pose_stamp.append(self.mensaje_matriz_a_pose(matrix, frame))

	    try:
	        respuesta = servicio_ik(ik_mensaje)
	    except:
	        print "Movimiento no ejecutado"

	    print respuesta.isValid[0]

	    if respuesta.isValid[0] == True:
	        movimiento =  dict(zip(respuesta.joints[0].name,respuesta.joints[0].position))
	        self.brazo.move_to_joint_positions(movimiento)
	    else:
	        print "Movimiento no ejecutado"


	def mover_ini_fin(self):
		while not rospy.is_shutdown():
			self.mover_baxter('base',self.pos_ini.position, self.pos_ini.orientation)
			rospy.sleep(1)
			self.mover_baxter('base',self.pos_fin.position, self.pos_fin.orientation)
			rospy.sleep(1)


def main():
	rospy.init_node("boton_ini")
	pos = movimiento('left')
	pos.posicion()
	pos.mover_ini_fin()

if __name__=='__main__':
	main()
