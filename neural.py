
import numpy as np
# import numba as nb
import scipy.special as sp
import pickle
import random

class NeuralNetworksample():
	def __init__(self,learing_rate):
		self.inputnodes = 4
		self.hidennodes = 4
		self.outputnodes = 4
		self.learing_rate = learing_rate
		self.activation_function = lambda x:sp.expit(x)

		self.weight_hi = np.random.normal(0,pow(self.hidennodes,-0.5),(self.inputnodes,self.hidennodes))
		self.weight_ho = np.random.normal(0,pow(self.outputnodes,-0.5),(self.hidennodes,self.outputnodes))
		# self.weight_hi = np.random.rand(self.inputnodes,self.hidennodes)
		# self.weight_ho = np.random.rand(self.hidennodes,self.outputnodes)

	def matrix_generator(self,input_number):
		self.input_matrix = np.array( [int(i) for i in str(input_number)], ndmin=1 ).T
		# self.input_matrix = np.array( input_number).T
		self.hiddeninput = np.dot(self.weight_hi,self.input_matrix)
		self.hiddenoutput = self.activation_function(self.hiddeninput)
		self.final_input = np.dot(self.weight_ho,self.hiddenoutput)
		self.finaloutput = self.activation_function(self.final_input)

	def train(self,input_number):
		self.matrix_generator(input_number)
		
		output_fuction = [0.0,0.99,0.0,0.0]
		
		output_error = np.array(output_fuction).T - self.finaloutput 
		hidden_error = np.dot( self.weight_ho.T,output_error ) 
		a = self.activation_function(self.final_input)
		b = self.activation_function(self.hiddeninput)
		
		self.weight_ho += self.learing_rate * np.dot(  (output_error * a * (1 - a) ), self.hiddenoutput.T)
		self.weight_hi += self.learing_rate * np.dot(  (hidden_error * b * (1 - b) ), self.input_matrix.T)
		
	def train_loop(self,number_of_train_set):
		print self.weight_hi
		print self.weight_ho
		for k in range(0,number_of_train_set):
			self.train(random.randint(1000,9999))


		print self.weight_hi
		print self.weight_ho
		with open('net_w11','wb') as file1:
			pickle.dump(self.weight_hi,file1)
		with open('net_w22','wb') as file2:
			pickle.dump(self.weight_ho,file2)
		# return

	def tell(self,input_number):
		with open('net_w11','rb') as f1:
			w1=pickle.load(f1)
		with open('net_w22','rb') as f2:
			w2=pickle.load(f2)

		self.weight_hi=w1
		self.weight_ho=w2
		self.matrix_generator(input_number)
		print self.finaloutput
		return 




c = NeuralNetworksample(0.4)
print "Enter number of train_loop"
train_loop = input()
c.train_loop(train_loop)

