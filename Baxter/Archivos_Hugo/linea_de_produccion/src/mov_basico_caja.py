#!/usr/bin/env 
# by Hugo Ubilla
import numpy as np
import math
import rospy
import baxter_interface
import tf
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
from geometry_msgs.msg import PoseStamped
from caja import bandeja


def mover_baxter(brazo, source_frame, trans, rot, ejecutar):
    nombre_servicio = '/ExternalTools/'+brazo+'/PositionKinematicsNode/IKService'
    servicio_ik = rospy.ServiceProxy(nombre_servicio,SolvePositionIK)
    frame = source_frame   
    arm = baxter_interface.Limb(brazo)

    # Promedio de velocidad del brazo
    arm.set_joint_position_speed(0.5)

    matrix = tf.transformations.concatenate_matrices(tf.transformations.translation_matrix(trans),
        tf.transformations.euler_matrix(rot[0],rot[1],rot[2])
        )
    
    rospy.wait_for_service(nombre_servicio,10)
    ik_mensaje = SolvePositionIKRequest()
    ik_mensaje.pose_stamp.append(mensaje_matriz_a_pose(matrix, frame))

    try:
        respuesta = servicio_ik(ik_mensaje)
    except:
        print "Movimiento no ejecutado"

    print respuesta.isValid[0]

    if respuesta.isValid[0] == True:
        movimiento =  dict(zip(respuesta.joints[0].name,respuesta.joints[0].position))
        if ejecutar == True:
            arm.move_to_joint_positions(movimiento)
    else:
        print "Movimiento no ejecutado"

    return movimiento

def mensaje_matriz_a_pose(T, frame):
    '''
    Transforma una mensaje 
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

def main():

    rospy.init_node('prueba_de_caja', anonymous = True)

    caja1 = bandeja(0.216, 0.279, 3, 3, -0.2)
    caja1.ubicacion(0.4,-0.2)
    centros_de_caja = caja1.centros

    rot = [math.pi, 0, -math.pi/2]
    
    inicial = mover_baxter('left', 'base', [0.5,0.9,-0.2], rot, False)

    mov = []
    for i in centros_de_caja:
        mover_baxter('left', 'base', [0.5,0.9,0.2], rot, False)
        mover_baxter('left', 'base', [0.5,0.9,-0.2], rot, False)
        mover_baxter('left', 'base', [0.5,0.9,0.2], rot, False)
        mover_baxter('left', 'base', [i[0], i[1], i[2]+0.2], rot, False)
        mover_baxter('left', 'base', i, rot, False)
        mover_baxter('left', 'base', [i[0], i[1], i[2]+0.2], rot, False)


if __name__ == '__main__':
	main()