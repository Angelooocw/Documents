#!/usr/bin/env 

#!/usr/bin/env 

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import baxter_interface
import math
import tf 


print "============ Starting tutorial setup"

# Se debe redirigir topic de joint_states para baxter
sys.argv.append('joint_states:=/robot/joint_states')  
moveit_commander.roscpp_initialize(sys.argv)

rospy.init_node('movimiento_prueba_1',anonymous=True)

robot = moveit_commander.RobotCommander()
#scene = moveit_commander.PlanningSceneInterface()
group = moveit_commander.MoveGroupCommander("left_arm")


# Publicador de trayectorias para que rviz las visualize.
display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',moveit_msgs.msg.DisplayTrajectory, queue_size = 10)


# PLANEANDO UNA POSE GOAL

rot = tf.transformations.quaternion_from_euler(math.pi, 0, -math.pi/2)

print rot


print "============ Generating plan 1"
pose_target = geometry_msgs.msg.Pose()
pose_target.orientation.x = rot[0]
pose_target.orientation.y = rot[1]
pose_target.orientation.z = rot[2]
pose_target.orientation.w = rot[3]
pose_target.position.x = 0.6
pose_target.position.y = 0
pose_target.position.z =  0
group.set_pose_target(pose_target)

# Llama al planificador para que calcule el plan y lo muestre en rviz 
# si tiene exito (no mueve el robot)
plan1 = group.plan()


print "============ Waiting while RVIZ displays plan1..."
#rospy.sleep(5)

# Se le puede peder a rviz que visualice el plan (la trayectoria).
# A traves de otro publicador. puede servir para otras cosas.
print "============ Visualizing plan1"
display_trajectory = moveit_msgs.msg.DisplayTrajectory()

display_trajectory.trajectory_start = robot.get_current_state()
display_trajectory.trajectory.append(plan1)
display_trajectory_publisher.publish(display_trajectory);

print "============ Waiting while plan1 is visualized (again)..."
rospy.sleep(1)

#MOVIENDO A UNA POSE GOAL
# go utiliza el ultimo plan y execute se le entrega el plan computado
# Similar a lo anterior solo que se usa una la funcion go()
# Notar que la la Pose Goal definida anteriormente aun esta activa

# Uncomment below line when working with a real robot

#group.go(wait=True)


#pose_target.position.x = 0.7
#pose_target.position.y = 0.7
#pose_target.position.z = 0.9
#group.set_pose_target(pose_target)


#group.go(wait=True)

# GROUP GO VA MOVIENDOSE ENTRE POSE TARGETS


group.go(wait=True)


pose_target.orientation.x = rot[0]
pose_target.orientation.y = rot[1]
pose_target.orientation.z = rot[2]
pose_target.orientation.w = rot[3]
pose_target.position.x = 0.6
pose_target.position.y = 0
pose_target.position.z = -0.1
group.set_pose_target(pose_target)


#group.go(wait=True)






# Use execute instead if you would like the robot to follow
# the plan that has already been computed


#print group.go(wait = True)


'''
# PLANIFICANDO A UN JOINT-SPACE GOAL

# Limpia el objetivo.
group.clear_pose_targets()

# Retorna el set de joint value para el grupo
group_variable_values = group.get_current_joint_values()
print "============ Joint values: ", group_variable_values

group_variable_values[0] = 1.0
group.set_joint_value_target(group_variable_values)

plan2 = group.plan()

print "============ Waiting while RVIZ displays plan2..."
rospy.sleep(5)
'''
'''

# CARTESIANS PATHS

# Tu puedes planificar un camino cartesiano directamente
# especificando una lista de waypoints para el end effector.


waypoints = []

# first orient gripper and move forward (+x)
wpose = geometry_msgs.msg.Pose()
wpose = group.get_current_pose().pose 

#wpose.orientation.w = 1.0

wpose.position.z -= 0.2
waypoints.append(copy.deepcopy(wpose))

# second move down
#wpose.position.x += 0.2
#waypoints.append(copy.deepcopy(wpose))

# third move to the side
wpose.position.y -= 0.2
waypoints.append(copy.deepcopy(wpose))


(plan3, fraction) = group.compute_cartesian_path(
                             waypoints,   # waypoints to follow
                             0.01,        # eef_step
                             0.0)         # jump_threshold

print "============ Waiting while RVIZ displays plan3..."

collision_object = moveit_msgs.msg.CollisionObject()


rospy.sleep(5)

#group.execute(plan3)
'''