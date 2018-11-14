import numpy.random
import numpy as np
import math
from matplotlib import pyplot as plt
from pylab import *
import scipy


def dataset(number_of_data,number_of_cluster,show_initial):

	expectation_vectors = [[] for i in range(number_of_cluster)]

	expectation_vectors[0] = [+12,+6]
	expectation_vectors[1] = [0,-6]
	expectation_vectors[2] = [-13,+7]

	convariance_matrixs = [[[]] for i in range(number_of_cluster)]

	convariance_matrixs[0] = [[3,1],[1,12]]
	convariance_matrixs[1] = [[11,2],[2,9]]
	convariance_matrixs[2] = [[5,0],[0,18]]

	values = [[] for i in range(number_of_cluster)]

	for k in range(number_of_cluster):
		values[k] = np.random.multivariate_normal(expectation_vectors[k],convariance_matrixs[k],int(number_of_data/number_of_cluster))

	data = np.vstack(values)

	if show_initial:
		plt.scatter(data[:,0],data[:,1])
		plt.xlim([-20,20])
		plt.ylim([-20,20])
		plt.xlabel("x",fontsize=16)
		plt.ylabel("y",fontsize=16)
		plt.show()

	return data

def sampling_svectors(number_of_data, number_of_cluster, pi_vector, predict_precision_matrixs, predict_expectation_vectors):
	
	eta = np.zeros((number_of_data,number_of_cluster))

	for n in range(0,number_of_data):
		for k in range(0,number_of_cluster):
			eta[n][k] = math.exp(-0.5*(x[n] - predict_expectation_vectors[k] ) @ predict_precision_matrixs[k] @ (x[n] - predict_expectation_vectors[k] ) + 0.5*math.log(np.linalg.det(predict_precision_matrixs[k])) + math.log(pi_vector[k]))

	for n in range(0,number_of_data):
		normalized_const = 0
		for k in range(0,number_of_cluster):
			normalized_const = normalized_const + eta[n][k]

		s_vectors[n] = np.random.multinomial(1,eta[n]/normalized_const)

	return s_vectors

def sampling_pi_vector(number_of_data,number_of_cluster,s_vectors):

	alpha_vector = np.array([1.0,1.0,1.0])	
	new_alpha_vector = np.zeros(number_of_cluster)


	for k in range(0,number_of_cluster):

		sum2 = 0

		for n in range(0,number_of_data):
			sum2 = sum2 + s_vectors[n][k]

		new_alpha_vector[k] = sum2 + alpha_vector[k]

	pi_vector = np.random.dirichlet(new_alpha_vector)

	return pi_vector

def sampling_predict_precision_matrixs_and_predict_expectation_vectors(number_of_data,number_of_cluster,s_vectors,):

	#ガウスウィシャード分布の超パラメータ
	beta = 1.0
	v = 2 # > D-1
	m_vector = np.ones(2)
	W_matrix = np.identity(2)

	new_beta = np.zeros(number_of_cluster)
	new_m_vector = np.zeros((number_of_cluster,2))


	for k in range(0,number_of_cluster):
		sum1 = 0
		sum2 = 0

		for n in range(0,number_of_data):
			sum1 = sum1 + s_vectors[n][k]*x[n]
			sum2 = sum2 + s_vectors[n][k]

		new_beta[k] = sum2 + beta 
		new_m_vector[k] = (sum1 + beta*m_vector)/new_beta[k]

	for k in range(0,number_of_cluster):
		sum1 = 0
		sum2 = 0
	
		for n in range(0,number_of_data):
			sum1 = sum1 + s_vectors[n][k]*(x[n].reshape(2,1)@x[n].reshape(1,2))
			sum2 = sum2 + s_vectors[n][k]
	
		new_W_inverse = sum1 + beta*(m_vector.reshape(2,1) @ m_vector.reshape(1,2)) - new_beta[k]*(new_m_vector[k].reshape(2,1) @ new_m_vector[k].reshape(1,2)) + np.linalg.inv(W_matrix)
		new_v = sum2 + v
	
		predict_precision_matrixs[k] = new_v * np.linalg.inv(new_W_inverse)
		predict_expectation_vectors[k] = np.random.multivariate_normal(new_m_vector[k],np.linalg.inv(new_beta[k]*predict_precision_matrixs[k]))

	return predict_precision_matrixs,predict_expectation_vectors

def multivariate_Gaussian_distribution(x,y,precision_matrix,expectation_vector):

	cordinate = np.array([x,y])
	convariance_matrix = np.linalg.inv(precision_matrix)
	
	return 1.0/((2.0*math.pi)**2*np.linalg.det(convariance_matrix))*math.exp(-0.5*(cordinate-expectation_vector)@np.linalg.inv(convariance_matrix)@(cordinate-expectation_vector))

def show_answer(number_of_data,number_of_cluster,x,s_vectors,predict_precision_matrixs,predict_expectation_vectors,show_contour):
	
	groups = [[] for i in range(number_of_cluster)] #空の配列をnumber_of_cluster個用意する。

	for n in range(0,number_of_data):
		for k in range(0,number_of_cluster):
			if s_vectors[n][k] == 1:
				groups[k].append(x[n].tolist())

	for k in range(0,number_of_cluster):
		groups[k] = np.array(groups[k])
		plt.scatter(groups[k][:,0],groups[k][:,1])
	
	if show_contour:

		N = 60
		cordinate_x = np.linspace(-20,20,N)
		cordinate_y = np.linspace(-20,20,N)
		cordinate_X,cordinate_Y = meshgrid(cordinate_x,cordinate_y)

		value_of_z = np.zeros((N,N))

		for k in range(0,number_of_cluster):
			for i in range(0,N):
				for j in range(0,N):
					value_of_z[j][i] = multivariate_Gaussian_distribution(cordinate_x[i],cordinate_y[j],predict_precision_matrixs[k],predict_expectation_vectors[k])

			plt.contour(cordinate_X,cordinate_Y,value_of_z) 
	
	plt.ylim([-20,20])
	plt.ylim([-20,20])
	plt.xlabel("x",fontsize=16)
	plt.ylabel("y",fontsize=16)
	plt.show()


if __name__  == "__main__":
	
	number_of_cluster = 3
	trial_times = 80
	number_of_data = 600

	assert number_of_data % number_of_cluster == 0, "入力データ数をクラス数の倍数にしてください。"

	show_initial = True #分類前の２次元データ分布を見るか否か。
	show_contour = True #分類後に正解等高線を表示するか否か。

	#初期化
	predict_expectation_vectors = np.zeros((number_of_cluster,2))
	predict_precision_matrixs = np.zeros((number_of_cluster,2,2))
	for i in range(0,number_of_cluster):
		predict_precision_matrixs[i] = np.identity(2)

	pi_vector = np.ones(number_of_cluster)
	s_vectors = np.zeros((number_of_data,number_of_cluster))

	#データを渡す
	x = dataset(number_of_data,number_of_cluster,show_initial)

	#ギブスサンプリングを実行
	for i in range(0,trial_times):

		#sをサンプリングする。
		s_vectors = sampling_svectors(number_of_data, number_of_cluster, pi_vector, predict_precision_matrixs, predict_expectation_vectors)
		
		#predict_precision_matrixsとpredict_expectation_vectorsをサンプリングする。
		predict_precision_matrixs, predict_expectation_vectors = sampling_predict_precision_matrixs_and_predict_expectation_vectors(number_of_data, number_of_cluster, s_vectors)

		#pi_vectorをサンプリングする。
		pi_vector = sampling_pi_vector(number_of_data, number_of_cluster, s_vectors)

	#答えをプロット
	show_answer(number_of_data,number_of_cluster,x,s_vectors,predict_precision_matrixs,predict_expectation_vectors,show_contour)

	#最終結果を表示
	print("Trial回数:",trial_times,"\n")
	for k in range(0,number_of_cluster):
		print("グループ"+str(k)+"の平均\n",predict_expectation_vectors[k])
		print("グループ"+str(k)+"の共分散行列\n",np.linalg.inv(predict_precision_matrixs[k]),"\n")
