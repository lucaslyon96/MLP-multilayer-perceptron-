#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import matplotlib.pyplot as plt
#Para rodar: pip install numpy, sudo apt-get install python-tk, pip install matplotlib
#Duvidas: 
'''
funcao np.amax
'''


X = np.array(([2,9],[1,5],[3,6]), dtype = float)
Y = np.array(([92],[86],[89]), dtype = float)
teste = np.array(([4,8]), dtype = float)


X = X/np.amax(X, axis=0)
Y = Y/100
teste = teste/np.amax(teste, axis = 0)

class Rede_neural(object):
	def __init__(self):
		#Parametros da rede
		self.inputSize = 2
		self.outputSize = 1
		self.hiddenSize = 3

		#Pesos inicializados aleatoriamente
		self.W1 = np.random.randn(self.inputSize,self.hiddenSize) #Camada de entrada
		self.W2 = np.random.randn(self.hiddenSize,self.outputSize) #Camada oculta

		#Passo para frente

	def foward(self,X):
		self.z = np.dot(X,self.W1) # Multiplicando entrada pelos pesos
		self.z2 = self.sigmoid(self.z) # Passando pela funcao de ativacao
		self.z3 = np.dot(self.z2,self.W2) #Passando pela camada oculta
		o = self.sigmoid(self.z3) #Funcao de ativacao final
		return o

	def sigmoid(self, s):
		return 1/(1+np.exp(-s))

		#Passo para Trás
	def sigmoid_derivada(self,s):
		return s*(1-s)

	def backward(self, X, Y, o):
		self.o_erro = Y - o #Erro
		self.o_delta = self.o_erro * self.sigmoid_derivada(o) # Derivada da sigmoide

		self.z2_error = self.o_delta.dot(self.W2.T) #Erro na camada oculta
		self.z2_delta = self.z2_error * self.sigmoid_derivada(self.z2) #Aplicando derivada da sigmoide ao erro em z2

		self.W1 += X.T.dot(self.z2_delta) #ajustando os pesos da primeira camada
		self.W2 += self.z2.T.dot(self.o_delta) #ajustando os pesos da camada oculta

	#Treinamento
	def treinamento(self,X,Y):
		o = self.foward(X)
		self.backward(X,Y,o)

	def predicao(self):
		print ('Resultado previsto')
		print ('Entrada '+str(teste))
		print ('Saida '+str(self.foward(teste)))

if __name__ == '__main__':

	RN = Rede_neural()
	erro = []
	for i in range(1000):
		print ('Entrada: '+ str(X))
		print ('Saida: '+ str(RN.foward(X)))
		print ('Esperado: '+str(Y))
		print ('Erro: '+ str(np.mean(np.square(Y-RN.foward(X)))))
		erro.append(np.sum((np.mean(np.square(Y-RN.foward(X))))))
		RN.treinamento(X,Y)
	RN.predicao()
	plt.plot(erro)
	plt.ylabel('erro')
	plt.show()
