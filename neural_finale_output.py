import numpy as np
# import numba as nb
import scipy.special as sp
import pickle
import random

class NeuralNetworksample():
	def __init__(self):
		with open('net_w1','rb') as file1:
			self.weight_hi = pickle.load(file1)
		with open('net_w2','rb') as file2:
			self.weight_ho = pickle.load(file2)
		self.activation_function = lambda x:sp.expit(x)

	def matrix_generator(self,input_number):
		self.input_matrix = np.array( [int(i) for i in str(input_number)], ndmin=1 ).T
		# self.input_matrix = np.array( input_number ).T
		self.hiddeninput = np.dot(self.weight_hi,self.input_matrix)
		self.hiddenoutput = self.activation_function(self.hiddeninput)
		self.final_input = np.dot(self.weight_ho,self.hiddenoutput)
		self.finaloutput = self.activation_function(self.final_input)


	def tell(self,input_number):
		with open('net_w1','rb') as f1:
			w1=pickle.load(f1)
		with open('net_w2','rb') as f2:
			w2=pickle.load(f2)

		self.weight_hi=w1
		self.weight_ho=w2
		self.matrix_generator(input_number)
		print self.finaloutput
		return 




c = NeuralNetworksample()
print "Enter a number"
number = input()
c.tell(number)