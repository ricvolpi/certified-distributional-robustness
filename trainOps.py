import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import numpy.random as npr
from ConfigParser import *
import os
import cPickle
import scipy.io
import sys
import glob
from numpy.linalg import norm
from scipy import misc

class TrainOps(object):

    def __init__(self, model, exp_dir, run):
        
	self.run = run
	
        self.model = model
	self.exp_dir = exp_dir
        
	self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth=True
	    	
    def load_exp_config(self):
	
	config = ConfigParser()
	config.read(self.exp_dir + '/exp_configuration')
	
	print self.exp_dir + '/exp_configuration'
	
	self.dataset = config.get('EXPERIMENT_SETTINGS', 'dataset')
	    
	self.data_dir = os.path.join('./data', self.dataset)
	
	self.log_dir = os.path.join(self.exp_dir,'logs')
        self.model_save_path = os.path.join(self.exp_dir,'model')
	
	if not os.path.exists(self.log_dir):
	    os.makedirs(self.log_dir)
	
	if not os.path.exists(self.model_save_path):
	    os.makedirs(self.model_save_path)
	
	
	self.train_iters = config.getint('MAIN_SETTINGS', 'train_iters')
	self.batch_size = config.getint('MAIN_SETTINGS', 'batch_size')
	self.model.batch_size=self.batch_size
	self.gamma = config.getfloat('MAIN_SETTINGS', 'gamma')
	self.model.gamma = config.getfloat('MAIN_SETTINGS', 'gamma')
	self.model.learning_rate_min = config.getfloat('MAIN_SETTINGS', 'learning_rate_min')
	self.model.learning_rate_max = config.getfloat('MAIN_SETTINGS', 'learning_rate_max')
	self.T_adv = config.getint('MAIN_SETTINGS', 'T_adv')
	    
    def load_mnist(self, split='train'):
        print ('Loading MNIST dataset.')
        image_file = 'train.pkl' if split=='train' else 'test.pkl'
        image_dir = os.path.join(self.data_dir,image_file)
        with open(image_dir, 'rb') as f:
            mnist = cPickle.load(f)
        images = mnist['X'] / 127.5 - 1
        labels = mnist['y']
        return images, labels

    def load_test_data(self):

        self.train_images, self.train_labels = self.load_mnist(split='train')
        self.test_images, self.test_labels = self.load_mnist(split='test')
        
    def train(self):
        
        # build a graph
        self.model.mode='train_encoder'
	self.model.build_model()
	
	train_images, train_labels = self.load_mnist(split='train')
	test_images, test_labels = self.load_mnist(split='test')
        
	
        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
	    
            summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())
	    
	    for t in range(self.train_iters):
		    
		i = t % int(train_images.shape[0] / self.batch_size)

		batch_z = train_images[i*self.batch_size:(i+1)*self.batch_size]
		batch_labels = train_labels[i*self.batch_size:(i+1)*self.batch_size]
		
		feed_dict = {self.model.z: batch_z, self.model.labels: batch_labels} 
	        
		#initializing z_hat with z
		sess.run([self.model.z_hat_assign_op], feed_dict) 
		
		# running the maximizer for z_hat
		for n in range(self.T_adv): 
		    _, max_l = sess.run([self.model.max_train_op, self.model.max_loss], feed_dict) 
		    
		# running the loss minimizer, using z_hat
		feed_dict[self.model.z] = sess.run(self.model.z_hat, feed_dict)  
		_, min_l = sess.run([self.model.min_train_op, self.model.min_loss], feed_dict) 

		#evaluating and saving the model
		if (t+1) % 100 == 0:

		    summary, min_l, max_l, acc = sess.run([self.model.summary_op, self.model.min_loss, self.model.max_loss, self.model.accuracy], feed_dict)
		    
		    train_rand_idxs = np.random.permutation(train_images.shape[0])[:1000]
		    test_rand_idxs = np.random.permutation(test_images.shape[0])[:1000]
		    
		    train_acc, train_min_loss = sess.run(fetches=[self.model.accuracy, self.model.min_loss], 
					   feed_dict={self.model.z: train_images[train_rand_idxs], 
						      self.model.labels: train_labels[train_rand_idxs]})
		    test_acc, test_min_loss = sess.run(fetches=[self.model.accuracy, self.model.min_loss], 
					   feed_dict={self.model.z: test_images[test_rand_idxs], 
						      self.model.labels: test_labels[test_rand_idxs]})
												      
		    summary_writer.add_summary(summary, t)
		    print ('Step: [%d/%d] train_min_loss: [%.4f] train_acc: [%.4f] test_min_loss: [%.4f] test_acc: [%.4f]'%(t+1, self.train_iters, train_min_loss, train_acc, test_min_loss, test_acc))
		    
		    print 'Saving'
		    saver.save(sess, os.path.join(self.model_save_path, 'encoder'))
	    
		    train_accuracy, train_loss, test_accuracy, test_loss = 0,0,0,0
		    
		    no_splits = 20. # just to fit in the GPU
		    
		    for batch_images, batch_labels in zip(np.array_split(train_images,no_splits),np.array_split(train_labels,no_splits)):		
			feed_dict = {self.model.z: batch_images, self.model.labels: batch_labels} 
			train_accuracy_tmp, train_loss_tmp = sess.run([self.model.accuracy, self.model.min_loss], feed_dict) 
			train_accuracy, train_loss = train_accuracy + train_accuracy_tmp/no_splits, train_loss + train_loss_tmp/no_splits 
		    print ('Train accuracy: [%.4f] train loss: [%.4f]'%(train_accuracy, train_loss))
		    
		    for batch_images, batch_labels in zip(np.array_split(test_images,no_splits),np.array_split(test_labels,no_splits)):		
			feed_dict = {self.model.z: batch_images, self.model.labels: batch_labels} 
			test_accuracy_tmp, test_loss_tmp = sess.run([self.model.accuracy, self.model.min_loss], feed_dict) 
			test_accuracy, test_loss = test_accuracy + test_accuracy_tmp/no_splits, test_loss + test_loss_tmp/no_splits 
		    print ('Test accuracy: [%.4f] test loss: [%.4f] (test)'%(test_accuracy, test_loss))

		    
		    with open(os.path.join(self.exp_dir,'results_'+str(self.run)+'.pkl'),'w') as f:
			cPickle.dump((train_accuracy, train_loss, test_accuracy, test_loss, self.gamma),f,cPickle.HIGHEST_PROTOCOL)
		

    
    def test(self, for_oracle=False):
	
	# test function runs on CPU
	
	self.config = tf.ConfigProto(device_count = {'GPU': 0})
	
        with tf.Session(config=self.config) as sess:
	    
	    print '\n\n\n...........................................................................'
	    print self.exp_dir, self.run

            tf.global_variables_initializer().run()

	    print ('Loading pre-trained model.')
	    variables_to_restore = slim.get_model_variables(scope='encoder')
	    restorer = tf.train.Saver(variables_to_restore)
	    restorer.restore(sess, os.path.join(self.model_save_path,'encoder'))

	    feed_dict = {self.model.z: self.train_images, self.model.labels: self.train_labels} 
	    train_accuracy, train_loss = sess.run([self.model.accuracy, self.model.min_loss], feed_dict) 
	    print ('Train accuracy: [%.4f] train loss: [%.4f] (train)'%(train_accuracy, train_loss))
	    
	    feed_dict = {self.model.z: self.test_images, self.model.labels: self.test_labels} 
	    test_accuracy, test_loss = sess.run([self.model.accuracy, self.model.min_loss], feed_dict) 
	    print ('Train accuracy: [%.4f] train loss: [%.4f] (test)'%(test_accuracy, test_loss))
	    	
	with open(os.path.join(self.exp_dir,'results_'+str(self.run)+'.pkl'),'w') as f:
	    cPickle.dump((train_accuracy, train_loss, test_accuracy, test_loss, self.gamma),f,cPickle.HIGHEST_PROTOCOL)
	
if __name__=='__main__':

    print 'To be implemented.'


