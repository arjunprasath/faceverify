import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pdb 
import cv2
from skimage.feature import hog
from skimage import data, color, exposure
from sklearn import svm
import random
import cPickle

hdf5_file_name = '../data_all.h5'
file = h5py.File(hdf5_file_name,'r')
nimgs = 4000





print type(file)

def train_classifier():
	random_index1 = []
	random_index2 = []

	for j in xrange(nimgs):
		random_index1.append(j)
		random_index2.append(j)
	random.shuffle(random_index1)
	random.shuffle(random_index2)

	for i in xrange(nimgs):
		id_img = file['id'][i]
		cam_img = file['image'][i]

		id_img_uint = id_img.astype(np.uint8)
		cam_img_uint = cam_img.astype(np.uint8)

		id_gray = cv2.cvtColor(id_img_uint, cv2.COLOR_BGR2GRAY)
		cam_gray = cv2.cvtColor(cam_img_uint, cv2.COLOR_BGR2GRAY)


		fd_id, hog_image = hog(id_gray, orientations=8, pixels_per_cell=(16, 16),
	                    cells_per_block=(1, 1), visualise=True)
		
		fd_cam, hog_image = hog(cam_gray, orientations=8, pixels_per_cell=(16, 16),
	                    cells_per_block=(1, 1), visualise=True)
		
		feature_vec = np.concatenate((fd_id,fd_cam))

		feature_vec = feature_vec.reshape(-1,len(feature_vec))
		if i==0:
			train_pdata = feature_vec
		else:
			train_pdata = np.append(train_pdata,feature_vec,0)


		##Negative samples preperation:

		id_img = file['id'][random_index1[i]]
		cam_img = file['image'][random_index2[i]]

		id_img_uint = id_img.astype(np.uint8)
		cam_img_uint = cam_img.astype(np.uint8)

		id_gray = cv2.cvtColor(id_img_uint, cv2.COLOR_BGR2GRAY)
		cam_gray = cv2.cvtColor(cam_img_uint, cv2.COLOR_BGR2GRAY)


		fd_id, hog_image = hog(id_gray, orientations=8, pixels_per_cell=(16, 16),
	                    cells_per_block=(1, 1), visualise=True)
		
		fd_cam, hog_image = hog(cam_gray, orientations=8, pixels_per_cell=(16, 16),
	                    cells_per_block=(1, 1), visualise=True)
		
		feature_vec = np.concatenate((fd_id,fd_cam))

		feature_vec = feature_vec.reshape(-1,len(feature_vec))
		if i==0:
			train_ndata = feature_vec
		else:
			train_ndata = np.append(train_ndata,feature_vec,0)


	train_data = np.append(train_pdata,train_ndata,0)
	labels = np.ones(nimgs).reshape(-1,1)
	labels = np.append(labels, np.zeros(nimgs).reshape(-1,1))
	print labels.shape
	#pdb.set_trace()
	classifier = svm.SVC(gamma=0.001)
	classifier.fit(train_data,labels)

	with open('my_dumped_classifier.pkl', 'wb') as fid:
	    cPickle.dump(classifier, fid) 


def test_classifier():
	ntests = 800
	random_index1 = []
	random_index2 = []

	for j in xrange(nimgs,nimgs+ntests):
		random_index1.append(j)
		random_index2.append(j)
	random.shuffle(random_index1)
	random.shuffle(random_index2)

	for i in xrange(nimgs,nimgs+ntests):
		id_img = file['id'][i]
		cam_img = file['image'][i]

		id_img_uint = id_img.astype(np.uint8)
		cam_img_uint = cam_img.astype(np.uint8)

		id_gray = cv2.cvtColor(id_img_uint, cv2.COLOR_BGR2GRAY)
		cam_gray = cv2.cvtColor(cam_img_uint, cv2.COLOR_BGR2GRAY)


		fd_id, hog_image = hog(id_gray, orientations=8, pixels_per_cell=(16, 16),
	                    cells_per_block=(1, 1), visualise=True)
		
		fd_cam, hog_image = hog(cam_gray, orientations=8, pixels_per_cell=(16, 16),
	                    cells_per_block=(1, 1), visualise=True)
		
		feature_vec = np.concatenate((fd_id,fd_cam))

		feature_vec = feature_vec.reshape(-1,len(feature_vec))
		if i==nimgs:
			test_pdata = feature_vec
		else:
			test_pdata = np.append(test_pdata,feature_vec,0)


		##Negative samples preperation:
		#pdb.set_trace()
		id_img = file['id'][random_index1[i-nimgs]]
		cam_img = file['image'][random_index2[i-nimgs]]

		id_img_uint = id_img.astype(np.uint8)
		cam_img_uint = cam_img.astype(np.uint8)

		id_gray = cv2.cvtColor(id_img_uint, cv2.COLOR_BGR2GRAY)
		cam_gray = cv2.cvtColor(cam_img_uint, cv2.COLOR_BGR2GRAY)


		fd_id, hog_image = hog(id_gray, orientations=8, pixels_per_cell=(16, 16),
	                    cells_per_block=(1, 1), visualise=True)
		
		fd_cam, hog_image = hog(cam_gray, orientations=8, pixels_per_cell=(16, 16),
	                    cells_per_block=(1, 1), visualise=True)
		
		feature_vec = np.concatenate((fd_id,fd_cam))

		feature_vec = feature_vec.reshape(-1,len(feature_vec))
		if i==nimgs:
			test_ndata = feature_vec
		else:
			test_ndata = np.append(test_ndata,feature_vec,0)
    

    	
    	
	with open('my_dumped_classifier.pkl', 'rb') as fid:
		clf_loaded = cPickle.load(fid)
	test_data = np.append(test_pdata,test_ndata,0)
	exp_labels = np.ones(ntests).reshape(-1,1)
	exp_labels = np.append(exp_labels, np.zeros(800).reshape(-1,1))
	pred_labels = clf_loaded.predict(test_data)

	#for k in xrange(len(exp_labels)):
		#print str(exp_labels[k]) + "----" + str(pred_labels[k])
	mask = pred_labels==exp_labels
	correct = np.count_nonzero(mask)
	pdb.set_trace()
	print correct*100.0/pred_labels.size

	







if __name__=="__main__":
	#train_classifier()
	test_classifier()



#training
#SVM.train(train_data, cv2.ml.ROW_SAMPLE, labels)
#SVM.save('svm_data.dat')

	#Normalization of features
	#scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
	#preprocessing.StandardScaler().fit(id_gray)



	# sift = cv2.xfeatures2d.SIFT_create()

	# kp = sift.detect(id_gray,None)
	
	# kp,des_id = sift.compute(id_gray,kp)
	# des_id = np.float32(des_id)
	# print des_id.shape

	
			
		

	# sift = cv2.xfeatures2d.SIFT_create()
	# kp = sift.detect(cam_gray,None)
	# kp,des_cam = sift.compute(cam_gray,kp)
	# des_cam = np.float32(des_cam)
	# print des_cam.shape
	
	# if len(des_cam.shape) > 0 and len(des_id.shape) > 0:
		
	# 	if des_cam.shape[0] > 40 and des_id.shape[0] > 40:
	# 		print 'Minimum 40 features in both images'


	# 		# K-Means Clustering
	# 		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	# 		ret,label,center_id=cv2.kmeans(des_id,41,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

	# 		# K-Means Clustering
	# 		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	# 		ret,label,center_cam=cv2.kmeans(des_cam,41,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

	# 		#cv2.drawKeypoints(id_gray,kp,id_img_uint,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	# 		#cv2.drawKeypoints(cam_gray,kp,cam_img_uint,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	# 		feature_vec = np.append(center_id,center_cam,0) 

			#cv2.imshow('id',id_img_uint)
			#cv2.imshow('cam',cam_img_uint)
			#cv2.waitKey(0)
			#cv2.destroyAllWindows()


	
	

	

	
	#if i==0:
	#	feature_mat = center
	#else:
	#	feature_mat = np.vstack([feature_mat, center]);



#print feature_mat.shape
#print (feature_vec.shape)




