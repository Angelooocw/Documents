#!/usr/bin/env 

import baxter_interface
import rospy
 
rospy.init_node("boton_ini")

class posicion():
	def __init__(self, arm):
		self.boton_brazo = baxter_interface.Navigator(arm)
		self.brazo = baxter_interface.Limb(arm)

	def posicion_inicial(self):

		print "Presione el boton de navegacion para guardar posicion 1"

		while True:
			if self.boton_brazo.button0 == 1:
				print "boton 0"
			if self.boton_brazo.button1 == 1:
				break

def main():
	boton = posicion('left')
	#boton.posicion_inicial()
	brazo = baxter_interface.Limb('left')
	print brazo.endpoint_pose()

if __name__=='__main__':
	main()
