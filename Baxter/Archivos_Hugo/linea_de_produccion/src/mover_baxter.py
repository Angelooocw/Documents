##!/usr/bin/env 

import rospy
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
import baxter_interface
import math
import tf


def mensaje_matriz_a_pose(T, frame):
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


def mover_baxter(brazo, source_frame, trans, rot):

    nombre_servicio = '/ExternalTools/'+brazo+'/PositionKinematicsNode/IKService'
    servicio_ik = rospy.ServiceProxy(nombre_servicio,SolvePositionIK)
    frame = source_frame   
    arm = baxter_interface.Limb(brazo)

    # Promedio de velocidad del brazo
    arm.set_joint_position_speed(1)

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
        print movimiento
        arm.move_to_joint_positions(movimiento)
    else:
        print "Movimiento no ejecutado"

    print respuesta.joints[0].position
    print respuesta.joints[0].name


def main():
    rospy.init_node('servicio_de_ik_left', anonymous = True)
  
    x_ini       = 0.31
    y_ini       = -0.62
    z_ini       = 1.4
    roll_ini    = 0 
    pitch_ini   = 0 
    yaw_ini     = 0 

    mover_baxter('right','base',[x_ini, y_ini, z_ini],[roll_ini, pitch_ini, yaw_ini])
    mover_baxter('right','base',[x_ini, y_ini, z_ini - 0.3],[roll_ini, pitch_ini, yaw_ini])
    mover_baxter('right','base',[x_ini, y_ini, z_ini],[roll_ini, pitch_ini, yaw_ini])

    x_ini       = 0.31
    y_ini       = 0.62
    z_ini       = 1.4
    roll_ini    = 0
    pitch_ini   = 0 
    yaw_ini     = 0 

    mover_baxter('left','base',[x_ini, y_ini, z_ini],[roll_ini, pitch_ini, yaw_ini])
    mover_baxter('left','base',[x_ini, y_ini, z_ini - 0.3],[roll_ini, pitch_ini, yaw_ini])
    mover_baxter('left','base',[x_ini, y_ini, z_ini],[roll_ini, pitch_ini, yaw_ini])


if __name__== '__main__':
    main()