#!/usr/bin/env python
# encoding: utf-8
import numpy as np
from numpy import array
import random
import matplotlib
import matplotlib.pyplot as plt 

#Para rodar: pip install numpy, sudo apt-get install python-tk, pip install matplotlib

#Duvidas: 

'''
funcao np.amax
'''


#Entradas
'''
Serie
Filme
Dia da semana ou fim de semana
Criancas ou nao
Ferias ou nao
Notas para cada genero
'''


#Parametros
'''
O que fazer com o bias??
Numero de camadas ocultas = 1
Numero de saidas de cada camada = 5,6,8
Funcao de ativacao = sigmoide
Batch ou online =  online
Numero de iteracoes de treinamento = 1000
Passo de Aprendizado = 0.5
Criterio de parada = erro < 10^-2 na nota
'''

Passo_aprendizado = 0.01
Entradas = []
Saida_esperada = []
Teste =[]

def leitor():
	with open('metade dos dados.csv','r') as arquivo:
		for linha in arquivo:
			linha_lida = linha.split(",")
			trata_linha(linha_lida)


def trata_linha(linha):
	Entrada_NN = []
	#Bias ?
	#Entrada_NN.append(-1)

	#Tratamento primeiro valor: generos do filme assistido
	byte = trata_generos(linha[0])
	Entrada_NN.append(byte)

	#Tratamento do segundo valor: filhos 1 se sim 0 se nao
	Entrada_NN.append(float(linha[1]))
	
	#Tratamento do terceiro valor: Ferias 1 se sim 0 se nao
	Entrada_NN.append(float(linha[2]))

	#Tratamento do quarto valor: Filme
	byte = trata_generos(linha[3])
	Entrada_NN.append(byte)

	#Tratamento do quinto valor: fim de semana 0 se sim 1 se nao
	Entrada_NN.append(float(linha[4]))

	#Tratamento do sexto valor ao 13 notas de cada genero
	temp_list =[]
	for x in range(5,13):
		temp_list.append(float(linha[x]))

	Saida_esperada.append(temp_list)
	Entradas.append(Entrada_NN)

def trata_generos(lista_generos):
	byte = [0,0,0,0,0,0,0,0]
	if ("Acao" in lista_generos):
		byte[0] = 1
	if ("Aventura" in lista_generos):
		byte[1] = 1
	if ("Comedia" in lista_generos):
		byte[2] = 1
	if ("Romance" in lista_generos):
		byte[3] = 1
	if ("Terror" in lista_generos):
		byte[4] = 1
	if ("Suspense" in lista_generos):
		byte[5] = 1
	if ("Animacao" in lista_generos):
		byte[6] = 1
	if ("Drama" in lista_generos):
		byte[7] = 1
	byte =int(''.join(str(e) for e in byte),2)
	return byte

class Rede_neural(object):
	def __init__(self):
		#Tamanho das camadas
		self.TamEntrada = 5
		self.TamOculta = 6
		self.TamSaida = 8

		#Pesos inicializados aleatoriamente
		self.W1 = np.random.randn(self.TamEntrada,self.TamOculta) #Camada de entrada
		self.W2 = np.random.randn(self.TamOculta,self.TamSaida) #Camada oculta

	#Passo para frente
	def Passo_Frente(self,X):
		self.z = np.dot(X,self.W1)
		self.z2 = self.func_at(self.z)
		self.z3 = np.dot(self.z2,self.W2)
		saida = self.func_at(self.z3)
		return saida

	def func_at(self, x):
		#sigmoide
		return 1/(1+np.exp(-x))

	def func_at_derivada(self,x):
		return x*(1-x)

	def Passo_tras(self,X,Y,saida):
		#Camada de saida
		self.saida_erro = Y - saida
		self.gradiente = self.saida_erro*self.func_at_derivada(saida)

		#Camada oculta
		self.z2_erro = self.gradiente.dot(self.W2.T)
		self.gradiente_oculto =  self.z2_erro * self.func_at_derivada(self.z2)
		
		self.W1 += Passo_aprendizado*(X.T.dot(self.gradiente_oculto))
		self.W2 += Passo_aprendizado*(self.z2.T.dot(self.gradiente))
	
	def treinamento(self,X,Y):
		saida = self.Passo_Frente(X)
		self.Passo_tras(X,Y,saida)

if __name__ == '__main__':
	leitor()
	# Shuffle : Entradas = random.shuffle(Entradas)
	Teste = Entradas[31:51]
	Teste_saida_esperada = Saida_esperada[31:51]
	
	Entradas = Entradas[0:31]
	Saida_esperada = Saida_esperada[0:31]

	Teste = array(Teste)
	Teste_saida_esperada = array(Teste_saida_esperada)
	Teste = Teste/np.amax(Teste,axis = 0)
	Teste_saida_esperada = Teste_saida_esperada/np.amax(Teste_saida_esperada,axis = 0)

	X = array(Entradas)
	Y = array(Saida_esperada)
	X = X/np.amax(X, axis=0)
	Y = Y/np.amax(Y, axis=0)

	Rede_TOP = Rede_neural()
	erro = []
	for i in range(100000):
		erro.append(np.sum((np.mean(np.square(Y-Rede_TOP.Passo_Frente(X))))))
		Rede_TOP.treinamento(X,Y)

	#plt.plot(erro)
	#plt.show()
	
	Resultados_teste = Rede_TOP.Passo_Frente(Teste)
	erros_teste = []
	for i in range(0,len(Resultados_teste)-1):
		print("Resultado esperado: " + str(Teste_saida_esperada[i]))
		print("Resultado obtido: "+ str(Resultados_teste[i]))
		print("--------------------------------------")
		erro_teste = np.sum(Resultados_teste[i] - Teste_saida_esperada[i])
		erros_teste.append(erro_teste)
	plt.plot(erros_teste)
	plt.show()
	
#Teste da rede neural
"""
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
"""