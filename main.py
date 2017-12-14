import tensorflow as tf
from Model import Model
from TrainOps import TrainOps
import glob
import os
import cPickle

import numpy.random as npr
import numpy as np

flags = tf.app.flags
flags.DEFINE_string('gpu', '0', "GPU to used")
flags.DEFINE_string('run', '0', "run")
flags.DEFINE_string('exp_dir', 'exp_dir', "Experiment directory")
flags.DEFINE_string('seed', 'seed', "Experiment directory")
flags.DEFINE_string('mode', 'mode', "Experiment directory")
FLAGS = flags.FLAGS

def main(_):
     
    GPU_ID = FLAGS.gpu
    SEED = FLAGS.seed
    
    npr.seed(int(SEED))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152 on stackoverflow
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

    EXP_DIR = FLAGS.exp_dir
    RUN = FLAGS.run
    
    model = Model()
    trainOps = TrainOps(model, EXP_DIR, RUN)
    trainOps.load_exp_config()
    
    if FLAGS.mode=='train':
	print 'Training Encoder'
	trainOps.train()

    elif FLAGS.mode=='test':
	print 'Testing'     
	trainOps.load_test_data()
	trainOps.test()
 
if __name__ == '__main__':
    tf.app.run()



    






