'''

Homework 4

Name     : Tailin Lo
NetID    : tl1720
N number : N15116873
Email    : tl1720@nyu.edu

'''

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import KMeans


# ============================= #
#	dataset load
# ============================= #
iris = load_iris()

# ============================= #
#	dataset normalize
# ============================= #
StandardScaler = preprocessing.StandardScaler()
X, y = iris.data, iris.target

cluster_num =  3
max_iter    = 50
runs        = 10
rnd_number  = 8131985
#rnd_number = 10
#np.random.seed(rnd_number)

X = StandardScaler.fit_transform(X)


def show_figure_clusters(data_clusters, target_clusters, centroids) :

	cluster_num = len(data_clusters)
	dim = len(data_clusters[0][0])
	subtitle_str = "My Kmeans --- cluster number " + str(cluster_num)
	centroid_color = ["#000000"]
	color = ["#FF0000", "#00FF00", "#0000FF", "#DB7093", "#EEE8AA", "#FFEFD5", "#DDA0DD", "#9ACD32", "#F5DEB3", "#008080"]
	fig = plt.figure(figsize=(15, 10))
	fig.suptitle(subtitle_str , fontsize=16) 
	plt.title("Kmeans : ")


	ndata = list()
	ncentroid = list()
	for i in range(0, cluster_num):
		ndata.append(np.matrix(data_clusters[i]))
		ncentroid.append(np.matrix(centroids[i]))
	# end for


	i = 0
	count = 1
	
	while i < dim :
		j = i+1
		while j < dim :
			plt.subplot(2, 3, count)
			for k in range(0, len(data_clusters)):
				plt.scatter(ndata[k][:,i] , ndata[k][:,j] , c=color[k], s=50)
				plt.scatter(ncentroid[k][:,i], ncentroid[k][:,j] , c=centroid_color, s=150)
				plt.xlabel('dimension: '+str(i))
				plt.ylabel('dimension: '+str(j))
			# end for
			j += 1
			count += 1
		i += 1

	plt.show()

# ============================================= end ============================================= #	

def data_recovery_more(clusters, points) :
	data = list()
	target = list()
	data_clusters = list()
	target_clusters = list()

	for k in range(len(clusters)) :
		data_cluster = list()
		target_cluster = list()
		for item in clusters[k] :
			data.append(list(points[item]))
			#data.append(iris.data[item])
			target.append(k)
			data_cluster.append(points[item])
			target_cluster.append(k)
		# end for	
		data_clusters.append(data_cluster)
		target_clusters.append(target_cluster)
	# end for
	return data, target, data_clusters, target_clusters

# ============================================= end ============================================= #	

def show_distortion_figure(dist_record, prefix_str) :
	x = list()
	for i in range(len(dist_record)):
		x.append(i+1)
	# end for
	fig = plt.figure(figsize=(15, 10))
	fig.suptitle(prefix_str+" --- Distortion" , fontsize=16) 
	#plt.subplot(2, 1, 1)
	plt.plot(x , dist_record)
	plt.xlabel('iteration')
	plt.ylabel('distortion')

	plt.show()

# ============================================= end ============================================= #	

def show_centroid_trajectory(means_record, prefix_str) :
	fig = plt.figure(figsize=(15, 10))
	fig.suptitle(prefix_str+" All Centroids Trajectory" , fontsize=16) 
	#plt.plot(x , dist_record)
	color = ["#FF0000", "#00FF00", "#0000FF", "#DB7093", "#EEE8AA", "#FFEFD5", "#DDA0DD", "#9ACD32", "#F5DEB3", "#008080"]
	dim = len(means_record[0][0])
	cluster_num = len(means_record[0])

	ndata = list()
	for k in range(cluster_num):
		centroid = list()
		for i in range(len(means_record)):
			centroid.append(means_record[i][k])
		# end for
		ndata.append(np.matrix(centroid))
	# end for
	i = 0
	count = 1
	
	while i < dim :
		j = i+1
		while j < dim :
			#print(i," , ",j," , ",count)
			plt.subplot(2, 3, count)
			for k in range(0, cluster_num):
				plt.plot(ndata[k][:,i] , ndata[k][:,j] , c=color[k])
				plt.xlabel('dimension: '+str(i))
				plt.ylabel('dimension: '+str(j))
			# end for
			j += 1
			count += 1
		i += 1
		# end while
	# end while
	plt.show()
	

# ============================================= end ============================================= #	

def show_clusters_by_iteration(data_clusters, target_clusters, centroids, iterations, prefix_str) :
	cluster_num = len(data_clusters)
	dim = len(data_clusters[0][0])
	subtitle_str = prefix_str + " --- Iteration " + str(iterations)
	centroid_color = ["#000000"]
	color = ["#FF0000", "#00FF00", "#0000FF", "#DB7093", "#EEE8AA", "#FFEFD5", "#DDA0DD", "#9ACD32", "#F5DEB3", "#008080"]
	fig = plt.figure(figsize=(15, 10))
	fig.suptitle(subtitle_str , fontsize=16) 
	plt.title("Kmeans : ")


	ndata = list()
	ncentroid = list()
	for i in range(0, cluster_num):
		ndata.append(np.matrix(data_clusters[i]))
		ncentroid.append(np.matrix(centroids[i]))
	# end for


	i = 0
	count = 1
	
	while i < dim :
		j = i+1
		while j < dim :
			#print(i," , ",j," , ",count)
			plt.subplot(2, 3, count)
			for k in range(0, len(data_clusters)):
				plt.scatter(ndata[k][:,i] , ndata[k][:,j] , c=color[k], s=50)
				plt.scatter(ncentroid[k][:,i], ncentroid[k][:,j] , c=centroid_color, s=150)
				plt.xlabel('dimension: '+str(i))
				plt.ylabel('dimension: '+str(j))
			# end for
			j += 1
			count += 1
		i += 1

	plt.show()

# ============================================= end ============================================= #	

np.random.seed(rnd_number)

'''

Section 2.1 (the black point is the centroid).

Summary: 1. If there are no seed, the random number is different in each run, which means KMeans will not be stable.
		 The stable algorithm means the result is the same in each run, which means we can get the same centroids in each run.  

		 2. When k is 1, all data points are in the same group, and the center is at (0,0) in each dimension. This is because 
		 	I normalized dataset before I did kmeans. 
		 	When k is small, there are no or less overlap between groups. However, when k is large, the overlap is serious, which
		 	means there may be more incorrect partition when there are more clusters.
		 	Note: In a normal case, the neighbor points of a centroid must be the same class. But as k becomes large, the adjacent
		 	zone of a centroid may include points with other class.


'''


for k in range(1,11):
	means, clusters = KMeans.mykmean(X, k, max_iter)
	#print(means)
	data, target, data_clusters, target_clusters = data_recovery_more(clusters, X)
	show_figure_clusters(data_clusters, target_clusters, means)
# end for


'''

Section 2.2

Note: I choose the inital centroid from the run with minimum distortion

Summary: 1. By running many times, we can get a more stable result.

		 2. By looking the distortion versus iteration figure, we can know the distortion will reduce to a stable value as iteration 
		 	number increase

'''

means, clusters, dist, dist_record, means_record, cluster_record, min_initial = KMeans.mykmeanmulti(X, cluster_num, max_iter, runs)
distortion_record, means_record, clusters_record = KMeans.kmeans_for_trajectory(X, min_initial, len(X), len(X[0]), cluster_num, max_iter)
show_distortion_figure(distortion_record, "My KMeansMulti")
show_centroid_trajectory(means_record, "My KMeansMulti")
data, target, data_clusters, target_clusters = data_recovery_more(clusters_record, X)
for iter in range(len(means_record)):
	show_clusters_by_iteration(data_clusters[iter], target_clusters[iter], means_record[iter], iter, "My KMeansMulti")
# end for


'''

Section 2.3

Summary: 1. The final distortion value is almost the same as the result of Section 2.2

		 2. By looking the distortion versus iteration figure, we can know the distortion will reduce to a stable value as iteration 
		 	number increase

		 3. The trajectrory of centroids move from the farest margin

'''

distortion_record, means_record, clusters_record = KMeans.mykmeanspp_for_trajectory(X, cluster_num, max_iter)
show_distortion_figure(distortion_record, "My Kmeans++")
show_centroid_trajectory(means_record, "My Kmeans++")
data, target, data_clusters, target_clusters = data_recovery_more(clusters_record, X)
for iter in range(len(means_record)):
	show_clusters_by_iteration(data_clusters[iter], target_clusters[iter], means_record[iter], iter, "My KMeans++")
# end for

'''

Section 2.4

To speedup Kmeans, I implement Elken algorithm in kmeans_core_speedup in KMeans.py. Running mykmeanspp(points, cluster_num, max_iter, True), users
can speedup the Kmeans

'''

'''

Section 3.1

The following code is about Section 3.1

'''
movie_reviews_data_folder = "./txt_sentoken"
dataset = load_files(movie_reviews_data_folder, shuffle=False)
print("=== Section 3.1 ===\n")
print("Samples: %d" % len(dataset.data))
count_vect = CountVectorizer(decode_error='ignore')
X_counts = count_vect.fit_transform(dataset.data)
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)
train_X, test_X, train_y, test_y = train_test_split(X_tfidf, dataset.target, train_size=0.5, random_state=rnd_number)
clf = MultinomialNB().fit(train_X, train_y)
predicted = clf.predict(test_X)
print("Accuracy: ",np.mean(predicted == test_y))

'''

Section 3.2

learnvocabulary function is in FeatureExtraction.py
getbof function in FeatureExtraction.py

'''

'''

Section 3.3

According to the tutorial, the main idea is to represent the data structure efficiently. Because the word
count matrix is a sparse matrix in the text-analysis case, it's suitable for us to use memory-saved data structure.
However, there is no sparse matrix in my project (text recognition). But I still can do an optimization from other way. 
For example, before calculating feature representation for each data point, I can precalculated the feature vector as
an unit vector, and then do projection. Thus, it doesn't have to renormalize the projected vector in each iteration. 

The projection is implemented in trainset_project function in FeatureExtraction.py
This function includes getbof function

'''
'''

Section 4

Due to time issue, I only do some simple grid-search
There are five parameter in the experiment. 
1. Window Size : the size of filter of extract local feature 
2. Class Size : There are 61 classes in my dataset, and use class size to choose the size of class.
3. Subsample Size : There are 92 images in each class, and use subsample size to choose the size of image.
4. Training Ratio : The ratio of size of training and testing 
5. Cluster Number : The size of bag of words.

I fixed some parameter, e.g. window size = 3, class size = 25, subsample size = 4.
I sweeped cluster number in the sequence 16, 32, 64, 128
And for each cluster number, I sweep the neighbor number in the sequence 16, 32, 64, 128, 256
In the document, there are some graphs.

The implementation of grid-search is in GridSearch.py.

The following is the steps I did in the grid-search:
1. sweep cluster numbers first
2. save every centroids of different cluster numbers
3. sweep neighbor number of different centroids
4. predict 

'''
'''

Section 5

1. Over one week, especially waiting for the result from KMeans

'''


