#!/usr/bin/env python
# -*- coding: cp1252 -*-
import numpy as np
import tensorflow as tf
import pandas as pd
import six
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix

class DNN:

	def __init__(self):

		self.g = tf.Graph()
		self.n_epochs = 400
		self.num_classes = 5
		self.x_width = 74
		self.num_neurons = 300
		self.col1cost = 1
		self.col2cost = 1
		self.col3cost = 1
		self.col4cost = 1
		self.col5cost = 1
		self.learning_rate = 0.001
		self.batch_size = 200

	def Lee_Datos(self): 

		pfile=open('featuresaux.txt','r')
		data=pfile.read() 
		pfile.close()
		data=np.genfromtxt(six.StringIO(data)) #Se sobre entiende que los #delimitadores son espacios
		data_pd=pd.DataFrame(data=data[0:,0:],index=data[0:,0],columns=data[0,0:])
		data_pd[1.0]=np.floor(data_pd[0.0] / 5)
		data_pd[2.0]=np.floor(data_pd[0.0] / 5)
		data_pd[3.0]=np.floor(data_pd[0.0] / 5)
		data_pd[4.0]=np.floor(data_pd[0.0] / 5)
		data_pd[1.0].where(data_pd[0.0] != 1.0, 1.0, inplace=True)
		data_pd[2.0].where(data_pd[0.0] != 2.0, 1.0, inplace=True)
		data_pd[3.0].where(data_pd[0.0] != 3.0, 1.0, inplace=True)
		data_pd[4.0].where(data_pd[0.0] != 4.0, 1.0, inplace=True)
		data_pd[0.0].where(data_pd[0.0] == 0.0, 1.0, inplace=True)
		data_pd[0.0] = np.ceil(data_pd[0.0] -1.0)
		data_pd[0.0].where(data_pd[0.0] == 0.0, 1.0, inplace=True)
		self.train_set, self.test_set = train_test_split(data_pd, test_size=0.2, random_state=42)
		self.data_pd_data = self.train_set.drop([0.0,1.0,2.0,3.0,4.0], axis=1)
		self.data_pd_target = self.train_set[[0.0,1.0,2.0,3.0,4.0]].copy()

	def fetch_batch(self,epoch, batch_index, batch_size):
		m, n = self.data_pd_data.shape
		data_pd_data_aux=self.data_pd_data.values
		data_pd_target_aux=self.data_pd_target.values
		#housing_data_plus_bias_target = np.c_[np.ones((m, 1)), housing.target]
		#housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
		data_pd_data_aux = data_pd_data_aux[batch_index*batch_size:(batch_index+1)*batch_size,0:]
		data_pd_target_aux = data_pd_target_aux[batch_index*batch_size:(batch_index+1)*batch_size,]
		#housing_data_plus_bias=pd.DataFrame(data=housing_data_plus_bias[0:,0:],index=housing_data_plus_bias[0:,0],columns=housing_data_plus_bias[0,0:]) 
		#scaled_housing_data_plus_bias = num_pipeline.fit_transform(housing_data_plus_bias)
		#X_batch = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
		#y_batch = tf.constant(housing_data_plus_bias_target.reshape(-1, 1), dtype=tf.float32, name="y")
		X_batch = data_pd_data_aux
		y_batch = data_pd_target_aux
		return X_batch, y_batch

	# Tensorflow convinience functions
	def weight_variable(self,shape, name):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial, name=name)

	def bias_variable(self,shape, name):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial, name=name)

	def variable_summaries(self,var):
	    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	    with tf.name_scope('summaries'):
		    mean = tf.reduce_mean(var)
		    tf.summary.scalar('mean', mean)
		    with tf.name_scope('stddev'):
		    	stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		    tf.summary.scalar('stddev', stddev)
		    tf.summary.scalar('max', tf.reduce_max(var))
		    tf.summary.scalar('min', tf.reduce_min(var))
		    tf.summary.histogram('histogram', var)

	    
	def multilayer_perceptron(self):
	    # Fully connected layer 1:
	    with tf.name_scope('input_layer') as scope:
		w_fc1 = self.weight_variable([self.x_width, self.num_neurons], 'weight_input')   # weights
		self.variable_summaries(w_fc1)
		b_fc1 = self.bias_variable([self.num_neurons], 'bias_input')  # biases
		self.variable_summaries(w_fc1)
		h_fc1 = tf.nn.relu(tf.matmul(self.x, w_fc1) + b_fc1, name='af_relu_input') # activation
		tf.summary.histogram('activations_input', h_fc1)
		h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob, name='dropout_input')          # dropout

	    # Fully connected layer 2:
	    with tf.name_scope('hidden_1') as scope:
		w_fc2 = self.weight_variable([self.num_neurons, self.num_neurons], 'weight_h1')
		self.variable_summaries(w_fc2)
		b_fc2 = self.bias_variable([self.num_neurons], 'bias_h1')
		self.variable_summaries(b_fc2)
		h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, w_fc2) + b_fc2, name='af_relu_h1')
		tf.summary.histogram('activations_hidden1', h_fc2)
		h_fc2_drop = tf.nn.dropout(h_fc2, self.keep_prob, name='dropout_h1')

	    # Fully connected layer 3:
	    with tf.name_scope('hidden_2') as scope:
		w_fc3 = self.weight_variable([self.num_neurons, self.num_neurons], 'weight_h2')
		self.variable_summaries(w_fc3)
		b_fc3 = self.bias_variable([self.num_neurons], 'bias_h2')
		self.variable_summaries(b_fc3)
		h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, w_fc3) + b_fc3, name='af_relu_h2')
		tf.summary.histogram('activations_hidden2', h_fc3)
		h_fc3_drop = tf.nn.dropout(h_fc3, self.keep_prob, name= 'dropout_h2')

	    # Fully connected layer 4:
	    with tf.name_scope('hidden_3') as scope:
		    w_fc4 = self.weight_variable([self.num_neurons, self.num_neurons], 'weight_h3')
		    self.variable_summaries(w_fc4)
		    b_fc4 = self.bias_variable([self.num_neurons], 'bias_h3')
		    self.variable_summaries(b_fc4)
		    h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, w_fc4) + b_fc4, name='af_relu_h3')
		    tf.summary.histogram('activations_hidden3', h_fc4)
		    h_fc4_drop = tf.nn.dropout(h_fc4, self.keep_prob, name= 'dropout_h3')
		    
	    # Fully connected layer 5:
	    with tf.name_scope('hidden_4') as scope:
		    w_fc5 = self.weight_variable([self.num_neurons, self.num_neurons], 'weight_h4')
		    self.variable_summaries(w_fc5)
		    b_fc5 = self.bias_variable([self.num_neurons], 'bias_h4')
		    self.variable_summaries(b_fc5)
		    h_fc5 = tf.nn.relu(tf.matmul(h_fc4_drop, w_fc5) + b_fc5, name='af_relu_h4')
		    tf.summary.histogram('activations_hidden4', h_fc5)
		    h_fc5_drop = tf.nn.dropout(h_fc5, self.keep_prob, name= 'dropout_h4')  	

	    # Readout layer
	    with tf.name_scope('read_out') as scope:
		w_fc_out = self.weight_variable([self.num_neurons, self.num_classes], 'weight_out')
		self.variable_summaries(w_fc_out)
		b_fc_out = self.bias_variable([self.num_classes], 'bias_out')
		self.variable_summaries(b_fc_out)
		# The softmax function will make probabilties of Good vs Bad score at the output
		self.logits = tf.matmul(h_fc5_drop, w_fc_out) + b_fc_out
		y_ = tf.nn.softmax(self.logits, name='af_softmax')
		tf.summary.histogram('activations_out', y_)
	    return y_

	def create_graph(self):
	    with self.g.as_default() as g:    
		
		### Placeholders ###
		with tf.name_scope('placeholder') as scope:
		    self.y = tf.placeholder(tf.float32, [None, self.num_classes], name='y')
		    self.x = tf.placeholder(tf.float32, [None, self.x_width], name='X') # Placeholder values
		    self.keep_prob = tf.placeholder("float", name='keep_prob') # Placeholder values

		### Create Network ###
		self.y_ = self.multilayer_perceptron()
		
		### Accuracy Metrics trainning ###
		with tf.name_scope('accuracy'):
		    with tf.name_scope('correct_prediction'):
		        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
		        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		    tf.summary.scalar('accuracy_summary', self.accuracy)
		
		### Customized Weighted Loss ###
		square_diff = tf.square(self.y - self.y_) # compute the prediction difference
		col1, col2, col3, col4, col5 = tf.split(square_diff,5,1) # split the (m,2) vector in two (m,1) vectors
		l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.005, scope=None)
		weights = tf.trainable_variables()
	 
		with tf.name_scope("cost_function") as scope:
		    regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
		    costwise_loss = self.col1cost*tf.reduce_sum(col1) + self.col2cost*tf.reduce_sum(col2) +self.col3cost*tf.reduce_sum(col3) + self.col4cost*tf.reduce_sum(col4) +self.col5cost*tf.reduce_sum(col5)
		    costwise_loss = costwise_loss + regularization_penalty
		    tf.summary.scalar('cost_function_summary', costwise_loss)

		# Train the algorithm using gradient descent
		with tf.name_scope("train") as scope:
		    self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(costwise_loss)
		
		### Write Summary Out ###
		self.merged = tf.summary.merge_all()

            	self.init = tf.global_variables_initializer()  
            	self.saver = tf.train.Saver()
		
	    #return g, train_step, x, y, keep_prob, y_, merged

	def Training(self):
		with self.g.as_default() as g:
		    with tf.Session() as sess:
		        self.init.run()
		        for epoch in range(self.n_epochs):
		            for batch_index in range(int(self.data_pd_data.shape[0]/self.batch_size)):
		                X_batch, y_batch = self.fetch_batch(epoch, batch_index, self.batch_size)
		                sess.run(self.train_step, feed_dict={self.x: X_batch, self.y: y_batch,self.keep_prob: 0.7})
		            acc_train = self.accuracy.eval(feed_dict={self.x: X_batch, self.y: y_batch,self.keep_prob: 1.0})
		            acc_test = self.accuracy.eval(feed_dict={self.x: self.data_pd_data.values,self.y: self.data_pd_target.values,self.keep_prob: 1.0})
		            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
		        save_path = self.saver.save(sess, "./my_model_final_good2.ckpt")

	def Test(self): # Para probar con los datos del set de prueba
		with self.g.as_default() as g:   
		    data_pd_test = self.test_set.drop([0.0,1.0,2.0,3.0,4.0], axis=1) 
		    data_pd_test_target = self.test_set[[0.0,1.0,2.0,3.0,4.0]].copy()
		    with tf.Session() as sess:
		        self.saver.restore(sess, "./my_model_final_good2.ckpt")
		        X_new_scaled = data_pd_test.values #Lo transformo de pd a numpy array
		        Z = self.logits.eval(feed_dict={self.x: X_new_scaled,self.keep_prob:1.0})
		        y_pred = np.argmax(Z, axis=1)
			data_pd_test_target = np.argmax(data_pd_test_target.values, axis=1)


		    print (confusion_matrix(data_pd_test_target, y_pred))


	def Entrada(self,feature):
		feature = np.asarray(feature) #Lo transformo de list a numpy array		
		feature=np.array(feature)[np.newaxis]
		Z = self.y_.eval(feed_dict={self.x: feature,self.keep_prob:1.0})
		y_pred = np.argmax(Z, axis=1)
		return y_pred

if __name__ == "__main__":
    DNN=DNN()
    DNN.Lee_Datos() 
    DNN.create_graph()
    DNN.Training() 
    DNN.Test()
