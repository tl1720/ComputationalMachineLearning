'''

Homework 4

Name     : Tailin Lo
NetID    : tl1720
N number : N15116873
Email    : tl1720@nyu.edu

======================================

Classification Lib 

'''

from sklearn.neighbors import KNeighborsClassifier
import FeatureExtraction
import FetchFile
import time

def classifiy(class_num, subsample_size, window_size, cluster_num, max_iter, rnd_number, neighbor_num, train_X, train_y, test_X, test_y):
	print("=== Classify Start ===\n")
	features = FeatureExtraction.read_features(FeatureExtraction.gen_feature_fname(class_num, subsample_size, window_size, cluster_num))
	knn = KNeighborsClassifier(n_neighbors=neighbor_num)
	feature_vectors_train = FeatureExtraction.trainset_project(features, train_X, True)
	feature_vectors_test  = FeatureExtraction.trainset_project(features, test_X, True)
	start = time.time()
	knn.fit(feature_vectors_train, train_y)
	print("KNN Fit Time: ",time.time()-start)
	start = time.time()
	acc = 0
	test_len = len(feature_vectors_test)
	for i in range(0, test_len):
		if knn.predict(feature_vectors_test[i]) == test_y[i]:
			acc += 1
		# end if
	# end for
	print("KNN Predict Time: ",time.time()-start)
	print("Accuracy: ",(acc/test_len),", Neighbor Number: ",neighbor_num)
	print("=== Classify Finish ===")


if __name__ == "__main__":	

	'''

		Unit Test for classifiy
		By uncomment the following code, you can run a simple KNN predict case (1 class/1 image/3 clusters)

	'''
	'''
	rnd_number 	   = 8131985
	class_num 	   = 1
	subsample_size = 1
	window_size    = 3
	cluster_num    = 3
	neighbor_num   = 1
	max_iter 	   = 50
	train_X, test_X, train_y, test_y = FetchFile.gen_data(class_num, subsample_size, window_size, rnd_number)
	classifiy(class_num, subsample_size, window_size, cluster_num, max_iter, rnd_number, neighbor_num, train_X, train_y, test_X, test_y)
	'''