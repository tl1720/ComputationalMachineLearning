'''

Homework 4

Name     : Tailin Lo
NetID    : tl1720
N number : N15116873
Email    : tl1720@nyu.edu

======================================

KMeans Lib 

'''

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# ============================= #
#	dataset load
# ============================= #
iris = load_iris()

# ============================= #
#	dataset normalize
# ============================= #
StandardScaler = preprocessing.StandardScaler()
X, y = iris.data, iris.target

def gen_zeros(dim) :
	zeros = list()
	for i in range(0, dim) :
		zeros.append(0.0)
	# end for
	return zeros

# ============================================= end ============================================= #	

def calc_distance(p1, p2, dim) :
	sum = 0.0
	for d in range(0, dim) :
		sum += math.pow(p1[d] - p2[d], 2)
	# end for
	return sum

# ============================================= end ============================================= #	

def calc_mean(points, cluster, dim) :
	sum = gen_zeros(dim)
	for i in cluster :
		for d in range(0, dim) :
			sum[d] += points[i][d]
		# end for
	# end for

	cluster_len = len(cluster)
	if cluster_len == 0 :
		print(cluster)
	# end if
	for d in range(0, dim) :
		sum[d] /= cluster_len
	# end for
	return sum

# ============================================= end ============================================= #	


def get_closest_cluster(means, point, dim) :
	min_val = 1e12
	min_k   = -1
	for k in range(0, len(means)) :
		dist = calc_distance(means[k], point, dim)
		if min_val > dist :
			min_val = dist
			min_k   = k
		# end if
	# end for
	return min_k, min_val

# ============================================= end ============================================= #	

def mean_exist(check, set) :
	if len(set) == 0:
		return False
	# end if
	for element in set :
		if list(element) == list(check):
			return True
		# end if
	# end for
	return False


# ============================================= end ============================================= #	

def kmeans_core(points, means, train_size, dim, cluster_num, max_iter) :
	start = time.time()
	# iteration
	iter = 0
	old_means = list()
	while iter < max_iter :
		clusters = list()
		for i in range(0, cluster_num) :
			clusters.append(list())
		# end for

		# partition points into corresponding clusters
		for i in range(0, train_size) :
			k, dist = get_closest_cluster(means, points[i], dim)
			clusters[k].append(i)
		# end for
		
		# update mean
		means = list()
		for k in range(0, cluster_num) :
			new_mean = calc_mean(points, clusters[k], dim)
			means.append(new_mean)
		# end for

		if iter == 0 :
			old_means = means
		elif means == old_means :
			#print("Kmeans: ",time.time()-start)
			break
		old_means = means
		iter += 1
	# end while
	#print("Kmeans: ",time.time()-start)
	return means, clusters


# ============================================= end ============================================= #	


def kmeans_for_trajectory(points, means, train_size, dim, cluster_num, max_iter) :
	# iteration
	iter = 0
	old_means         = list()
	distortion_record = list()
	means_record      = list()
	clusters_record   = list()
	while iter < max_iter :
		clusters = list()
		for i in range(0, cluster_num) :
			clusters.append(list())
		# end for

		# partition points into corresponding clusters
		for i in range(0, train_size) :
			k, dist = get_closest_cluster(means, points[i], dim)
			clusters[k].append(i)
		# end for
		
		distortion_record.append(calc_all_dist(points, clusters, means, dim))
		means_record.append(means)
		clusters_record.append(clusters)

		# update mean
		means = list()
		for k in range(0, cluster_num) :
			new_mean = calc_mean(points, clusters[k], dim)
			means.append(new_mean)
		# end for

		if iter == 0 :
			old_means = means
		elif means == old_means :
			break
		old_means = means
		iter += 1
	# end while
	return distortion_record, means_record, clusters_record


# ============================================= end ============================================= #	

def means_points_distance_matrix(means, points, dim) :
	mp_dist = list()
	for point in points:
		dist = list()
		for mean in means:
			dist.append(calc_distance(mean, point, dim))
		# end for
		mp_dist.append(dist)
	# end for
	return mp_dist


def means_means_distance_matrix(means, cluster_num, dim, scale=0.5) :
	mm_dist = list()
	mm_closeest_dist = list()
	for i in range(0, cluster_num):
		dist = list()
		for j in range(0, cluster_num):
			if i == j: 
				dist.append(1e12)
			elif i < j:
				dist.append(scale*calc_distance(means[i], means[j], dim))
			else:
				dist.append(mm_dist[j][i])
			# end if
		# end for
		mm_dist.append(dist)
	# end for

	for k in range(0, cluster_num):
		mm_closeest_dist.append(min(mm_dist[k]))
	# end for
	return mm_closeest_dist, mm_dist



def kmeans_core_speedup(points, means, train_size, dim, cluster_num, max_iter, return_bound_matrix=False) :
	start = time.time()

	# precomputed distance matrix (centers to each point)
	mp_dist = means_points_distance_matrix(means, points, dim)


	# generate bound matrix for each point
	bound_matrix = list()
	for i in range(0, train_size):
		# setting lower bound
		lower_bound = list()
		for k in range(0, cluster_num):
			lower_bound.append(mp_dist[i][k])
		# end for
			
		# setting upper bound
		upper_bound = min(lower_bound)
			
		# setting owner
		bound_matrix.append([lower_bound, upper_bound, lower_bound.index(upper_bound)])
	# end for


	# iteration
	iter = 0
	old_means = list()
	while iter < max_iter :
		mm_closeest_dist, mm_dist = means_means_distance_matrix(means, cluster_num, dim)
		for i in range(0, train_size):
			# if u(x) > s(c(x))
			if bound_matrix[i][1] > mm_closeest_dist[bound_matrix[i][2]] :
				# for each c != c(x), checking (1) u(x) > l(x,c) (2) u(x) > 0.5*d(c(x),c)
				for k in range(0, cluster_num):
					if k == bound_matrix[i][2] :
						continue
					# end if
					if bound_matrix[i][1] > bound_matrix[i][0][k] and \
					   bound_matrix[i][1] > mm_dist[bound_matrix[i][2]][k]:
						pk_dist = calc_distance(points[i], means[k], dim)
						pc_dist = calc_distance(points[i], means[bound_matrix[i][2]], dim)
						if pk_dist < pc_dist:
							bound_matrix[i][2]    = k
							bound_matrix[i][0][k] = pk_dist
							bound_matrix[i][1] 	  = pc_dist
						# end if
					# end if
				# end for		
			# end if
		# end for

		# calculate new mean
		new_means = list()
		for k in range(0, cluster_num):
			sum = gen_zeros(dim)
			num = 0
			for i in range(0, train_size):
				if bound_matrix[i][2] == k:
					for d in range(0, dim) :
						sum[d] += points[i][d]
					# end for
					num += 1
			# end for
			for d in range(0, dim) :
				sum[d] /= num
			# end for
			new_means.append(sum)
		# end for

		# update bound matrix

		# calculate distance matrix for old mean and new mean
		shift_mean = list()
		for k in range(0, cluster_num):
			shift_mean.append(calc_distance(new_means[k], means[k], dim))
		# end for


		for i in range(0, train_size):
			# update lower bound
			for k in range(0, cluster_num):
				bound_matrix[i][0][k] = max([0, bound_matrix[i][0][k] - shift_mean[k]])
			# end for

			# update upper bound
			bound_matrix[i][1] += shift_mean[bound_matrix[i][2]]
		# end for
		if iter == 0:
			means = new_means
		elif means == new_means:
			#print("Kmeans: ",time.time()-start)
			if return_bound_matrix:
				return means, bound_matrix
			else:
				return means
			# end if
		# end if
		means = new_means
		iter += 1
	# end while

	#print("Kmeans: ",time.time()-start)
	if return_bound_matrix:
		return means, bound_matrix
	else:
		return means


# ============================================= end ============================================= #	


def mykmean(points, cluster_num, max_iter, speedup=False) :
	point_dim  = len(points[0])
	train_size = len(points)

	# initialize cluster
	means = list()
	point_len = len(points)
	while len(means) < cluster_num :
		choose_num = int(point_len*np.random.random_sample())
		if not mean_exist(points[choose_num], means) :
			means.append(points[choose_num])
	# end while
	if not speedup :
		return kmeans_core(points, means, train_size, point_dim, cluster_num, max_iter)
	else:
		return kmeans_core_speedup(points, means, train_size, point_dim, cluster_num, max_iter)
	# end if


# ============================================= end ============================================= #	

def calc_all_dist(points, clusters, means, dim) :
	all_dist = 0.0
	for k in range(0, len(clusters)) :
		for i in clusters[k] :
			all_dist += calc_distance(points[i], means[k], dim)
		# end for
	# end for
	return all_dist


def gen_clusters(bound_matrix, cluster_num):
	clusters = list()
	for k in range(0, cluster_num):
		cluster = list()
		for i in range(0, len(bound_matrix)):
			if bound_matrix[i][2] == k:
				cluster.append(i)
			# end if
		# end for
		clusters.append(cluster)
	# end for
	return clusters


def mykmeanmulti(points, cluster_num, max_iter, runs, speedup=False) :
	point_dim  = len(points[0])
	train_size = len(points)
	
	min_dist 	 = 1e12
	min_means 	 = list()
	min_clusters = list() 
	min_initial  = list()
	dist_record  = list()
	means_record = list()
	cluster_record = list()
	for i in range(0, runs) :
		# initialize cluster
		means = list() 
		while len(means) < cluster_num :
			choose_num = int(train_size*np.random.random_sample())
			if not mean_exist(points[choose_num], means) :
				means.append(points[choose_num])
		# end while
		if not speedup:
			new_means, new_clusters = kmeans_core(points, means, train_size, point_dim, cluster_num, max_iter)
		else:
			new_means, bound_matrix = kmeans_core_speedup(points, means, train_size, point_dim, cluster_num, max_iter)
			new_clusters = gen_clusters(bound_matrix, cluster_num)
		# end if
		cluster_record.append(new_clusters)
		means_record.append(new_means)
		new_dist = calc_all_dist(points, new_clusters, new_means, point_dim)
		dist_record.append(new_dist)
		#print(new_dist, " , ", min_dist)
		if new_dist < min_dist :
			min_means 	 = new_means
			min_clusters = new_clusters
			min_dist     = new_dist
			min_initial  = means
		# end if
	# end for
	return min_means, min_clusters, min_dist, dist_record, means_record, cluster_record, min_initial

# ============================================= end ============================================= #	
	
def mykmeanspp(points, cluster_num, max_iter, speedup=False) :
	point_dim  = len(points[0])
	train_size = len(points)

	# initialize cluster
	means = list()
	means.append(points[0])
	for i in range(1, cluster_num):
		dist_list = list()
		acc_dist = 0.0
		for j in range(0, len(points)) :
			k, dist = get_closest_cluster(means, points[j], point_dim)
			acc_dist += dist
			dist_list.append((j, acc_dist))
		# end for 
		choose_num = acc_dist*np.random.random_sample()

		for i in range(0, len(dist_list)) :
			if choose_num > dist_list[i][1] :
				continue
			else :
				means.append(points[dist_list[i][0]])
				break
			# end if
		# end for
	# end for
	if not speedup:
		return kmeans_core(points, means, train_size, point_dim, cluster_num, max_iter)
	else:
		return kmeans_core_speedup(points, means, train_size, point_dim, cluster_num, max_iter)
	# end if


# ============================================= end ============================================= #	

def mykmeanspp_for_trajectory(points, cluster_num, max_iter) :
	point_dim  = len(points[0])
	train_size = len(points)

	# initialize cluster
	means = list()
	means.append(points[0])
	for i in range(1, cluster_num):
		dist_list = list()
		acc_dist = 0.0
		for j in range(0, len(points)) :
			k, dist = get_closest_cluster(means, points[j], point_dim)
			acc_dist += dist
			dist_list.append((j, acc_dist))
		# end for 
		choose_num = acc_dist*np.random.random_sample()

		for i in range(0, len(dist_list)) :
			if choose_num > dist_list[i][1] :
				continue
			else :
				means.append(points[dist_list[i][0]])
				break
			# end if
		# end for
	# end for

	return kmeans_for_trajectory(points, means, train_size, point_dim, cluster_num, max_iter)


# ============================================= end ============================================= #	

def data_recovery(clusters, points) :
	data = list()
	target = list()

	for k in range(len(clusters)) :
		for item in clusters[k] :
			data.append(list(points[item]))
			#data.append(iris.data[item])
			target.append(k)
		# end for	
	# end for
	return data, target


# ============================================= end ============================================= #	

def show_confusion_matrix(clusters, data, target) :
	

	# Print and plot the confusion matrix
	cm = metrics.confusion_matrix(iris.target, target)
	cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	print('Normalized confusion matrix for kmeans')
	print(cm_normalized)


# ============================================= end ============================================= #	

'''

Users can use TailinLo_tl1720_HW4.py as a test

'''

