import tensorflow as tf
import tensorflow.contrib.slim as slim

class Model(object):
    """Neural network
    """
    def __init__(self, mode='train', learning_rate=0.0003, gamma = 1.):
	self.gamma = gamma
        self.learning_rate_min = learning_rate
        self.learning_rate_max = learning_rate
	self.no_models = 5
	self.no_classes = 10
	self.no_channels = 1
	self.img_size_1 = 28
	self.img_size_2 = 28
	self.embedding_size = 512
	self.batch_size=64
	
	
	   
    def encoder(self, images, reuse=False, return_fc4=False, is_training=False, from_features = False):
	
	with tf.variable_scope('encoder', reuse=reuse):
	    with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):
		with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.elu, padding='VALID'):
		    
		    net = slim.conv2d(images, 32, 8, scope='conv1')
		    net = slim.conv2d(net, 64, 6, scope='conv2')
		    net = slim.conv2d(net, 128, 5, scope='conv3')
		    net = tf.contrib.layers.flatten(net)
		    net = slim.fully_connected(net, 1024, scope='fc1')
		    net = slim.fully_connected(net, self.no_classes, activation_fn=None, scope='fc5')
		    return net
		
		
		
    def build_model(self):
    
	self.z = tf.placeholder(tf.float32, [None, self.img_size_1, self.img_size_2, self.no_channels], 'z')
	self.labels = tf.placeholder(tf.int64, [None], 'labels')
	
	# variable whose values are learned to maximize the loss
	self.z_hat = tf.get_variable('z_hat', [self.batch_size, self.img_size_1, self.img_size_2, self.no_channels])
	self.z_hat_assign_op = self.z_hat.assign(self.z)
		    
	self.logits = self.encoder(self.z)
	self.logits_hat = self.encoder(self.z_hat, reuse=True)
	
	# for testing
	self.pred = tf.argmax(self.logits, 1)
	self.correct_pred = tf.equal(self.pred, self.labels)
	self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
	
	# cross-entropy loss
	self.min_loss = slim.losses.sparse_softmax_cross_entropy(self.logits, self.labels)
	# cross-entropy loss and cost
	self.max_loss = slim.losses.sparse_softmax_cross_entropy(self.logits_hat, self.labels) - self.gamma * tf.reduce_mean(tf.square(self.z - self.z_hat)) 
	
	# using Adam for the minimizer and vanilla Gradient Descent for the maximizer
	self.min_optimizer = tf.train.AdamOptimizer(self.learning_rate_min) 
	self.max_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate_max) 
	
	t_vars = tf.trainable_variables()
	min_vars = [var for var in t_vars if 'z_hat' not in var.name]
	max_vars = [var for var in t_vars if 'z_hat' in var.name]
	
	
	self.min_train_op = slim.learning.create_train_op(self.min_loss, self.min_optimizer, variables_to_train = min_vars)
	self.max_train_op = slim.learning.create_train_op(-self.max_loss, self.max_optimizer, variables_to_train = max_vars)
	
	min_loss_summary = tf.summary.scalar('min_loss', self.min_loss)
	max_loss_summary = tf.summary.scalar('max_loss', self.max_loss)
	
	accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
	self.summary_op = tf.summary.merge([min_loss_summary, max_loss_summary, accuracy_summary])
    

    
