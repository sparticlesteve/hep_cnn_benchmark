# System imports
from __future__ import print_function
from __future__ import division
from __Future__ import absolute_import
import os

# Data libraries
import h5py as h5
import numpy as np
from numpy.random import RandomState as rng

# Tensorflow
import tensorflow as tf
import tensorflow.contrib.keras as tfk


class DataSet(object):
    """ATLAS image data handler"""
    
    def reset(self):
        self._epochs_completed = 0
        self._file_index = 0
        self._data_index = 0
    
    def load_next_file(self):
        #only load a new file if there are more than one file in the list:
        if self._num_files > 1 or not self._initialized:
            try:
                with h5.File(self._filelist[self._file_index],'r') as f:
                    #determine total array size:
                    numentries=f['data'].shape[0]
                
                    if self._split_file:
                        blocksize = int(np.ceil(numentries/float(self._num_tasks)))
                        start = self._taskid*blocksize
                        end = (self._taskid+1)*blocksize
                    else:
                        start = 0
                        end = numentries
                
                    #load the chunk which is needed
                    self._images = f['data'][start:end]
                    self._labels = f['label'][start:end]
                    self._normweights = f['normweight'][start:end]
                    self._weights = f['weight'][start:end]
                    self._psr = f['psr'][start:end]
                    f.close()
            except EnvironmentError:
                raise EnvironmentError("Cannot open file "+self._filelist[self._file_index])
                
            #sanity checks
            assert self._images.shape[0] == self._labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (self._images.shape, self_.labels.shape))
            assert self._labels.shape[0] == self._normweights.shape[0], (
                'labels.shape: %s normweights.shape: %s' % (self._labels.shape, self._normweights.shape))
            assert self._labels.shape[0] == self._psr.shape[0], (
                'labels.shape: %s psr.shape: %s' % (self._labels.shape, self._psr.shape))
            self._initialized = True
        
            #set number of samples
            self._num_examples = self._labels.shape[0]
        
            #reshape labels and weights
            self._labels = np.expand_dims(self._labels, axis=1).astype(np.int32, copy=False)
            self._normweights = np.expand_dims(self._normweights, axis=1)
            self._weights = np.expand_dims(self._weights, axis=1)
            self._psr = np.expand_dims(self._psr, axis=1)
            
            #transpose images if data format is NHWC
            if self._data_format == "NHWC":
                #transform for NCHW to NHWC
                self._images = np.transpose(self._images, (0,2,3,1))
            
        #create permutation
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        #shuffle
        self._images = self._images[perm]
        self._labels = self._labels[perm]
        self._normweights = self._normweights[perm]
        self._weights = self._weights[perm]
        self._psr = self._psr[perm]
        
    def __init__(self, filelist, num_tasks=1, taskid=0,
                 split_filelist=False, split_file=False,
                 data_format="NCHW"):
        """Construct DataSet"""
        #multinode stuff
        self._num_tasks = num_tasks
        self._taskid = taskid
        self._split_filelist = split_filelist
        self._split_file = split_file
        self._data_format = data_format
        
        #split filelist?
        self._num_files = len(filelist)
        start = 0
        end = self._num_files
        if self._split_filelist:
            self._num_files = int(np.floor(len(filelist)/float(self._num_tasks)))
            start = self._taskid * self._num_files
            end = start + self._num_files
        
        assert self._num_files > 0, ('filelist is empty')
        
        self._filelist = filelist[start:end]
        self._initialized = False
        self.reset()
        self.load_next_file()

    @property
    def num_files(self):
        return self._num_files
    
    @property
    def num_samples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._data_index
        self._data_index += batch_size
        end=int(np.min([self._num_examples,self._data_index]))
        
        #take what is there
        images = self._images[start:end]
        labels = self._labels[start:end]
        normweights = self._normweights[start:end]
        weights = self._weights[start:end]
        psr = self._psr[start:end]
        
        if self._data_index > self._num_examples:
            #remains:
            remaining = self._data_index-self._num_examples
            
            #first, reset data_index and increase file index:
            self._data_index=0
            self._file_index+=1
            
            #check if we are at the end of the file list
            if self._file_index >= self._num_files:
                #epoch is finished
                self._epochs_completed += 1
                #reset file index and shuffle list
                self._file_index=0
                np.random.shuffle(self._filelist)
            
            #load the next file
            self.load_next_file()
            #assert batch_size <= self._num_examples
            #call rerucsively
            tmpimages,tmplabels,tmpnormweights,tmpweights,tmppsr = self.next_batch(remaining)
            #join
            images = np.concatenate([images,tmpimages],axis=0)    
            labels = np.concatenate([labels,tmplabels],axis=0)
            normweights = np.concatenate([normweights,tmpnormweights],axis=0)
            weights = np.concatenate([weights,tmpweights],axis=0)
            psr = np.concatenate([psr,tmppsr],axis=0)
        
        return images, labels, normweights, weights, psr

class DummySet(object):
    """Dummy data handler"""
    
    def reset(self):
        self._random = rng(self._seed)
        self._data_index = 0
        self._epochs_completed = 0
        
    def __init__(self, input_shape, samples_per_epoch, task_index=1):
        self._seed = task_index * 13
        self._shape = input_shape
        self._datasize = int(np.prod(self._shape))
        self._samples_per_epoch = samples_per_epoch
        self.reset()
        
    def next_batch(self, batch_size):
        data = np.reshape(self._random.rand(self._datasize*batch_size), [batch_size]+self._shape)
        labels = np.expand_dims(self._random.random_integers(0, 1, batch_size),1)
        normweights = np.expand_dims(self._random.rand(batch_size),1)
        weights = normweights
        psr = labels
        
        #increase data counter and check if epoch finished
        self._data_index += batch_size
        if self._data_index >= self._samples_per_epoch:
            self._data_index = 0
            self._epochs_completed += 1
        
        return data, labels, normweights, weights, psr

def build_cnn_model(args):
    """Construct the HEP CNN model"""
    
    #datatype
    dtype=args["precision"]
    
    #find out which device to use:
    device='/cpu:0'
    if args['arch']=='gpu':
        device='/gpu:0'
    
    #define empty variables dict
    variables={}
    
    #rotate input shape depending on data format
    data_format=args['conv_params']['data_format']
    input_shape = args['input_shape']
    
    #create graph handle
    args['graph'] = tf.Graph()
    
    with args['graph'].as_default():
    
        #create placeholders
        batch_size = args['train_batch_size_per_node']
        variables['images_'] = tf.placeholder(dtype, shape=[batch_size]+input_shape)
        variables['keep_prob_'] = tf.placeholder(dtype)
    
        #empty network:
        network = []
    
        #input layer
        network.append(tf.reshape(variables['images_'], [-1]+input_shape, name='input'))
    
        #get all the conv-args stuff:
        activation = args['conv_params']['activation']
        initializer = args['conv_params']['initializer']
        ksize = args['conv_params']['filter_size']
        num_filters = args['conv_params']['num_filters']
        padding = str(args['conv_params']['padding'])
        
        #conv layers:
        prev_num_filters = args['input_shape'][0]
        if data_format == "NHWC":
            prev_num_filters = args['input_shape'][2]
        
        for layerid in range(1, args['num_layers'] + 1):
        
            #create weight-variable
            conv_weight_name = 'conv%i_w' % str(layerid)
            variables[conv_weight_name] = tf.Variable(
                initializer([ksize, ksize, prev_num_filters, num_filters], dtype=dtype),
                name=conv_weight_name, dtype=dtype)
            prev_num_filters = num_filters
        
            #conv unit
            network.append(tf.nn.conv2d(network[-1],
                                        filter=variables[conv_weight_name],
                                        strides=[1, 1, 1, 1], 
                                        padding=padding,
                                        data_format=data_format,
                                        name='conv'+str(layerid)))
        
            #batchnorm if desired
            outshape = network[-1].shape[1:]
            if args['batch_norm']:
                #add batchnorm
                #mu
                variables['bn'+str(layerid)+'_m'] = tf.Variable(
                    tf.zeros(outshape,dtype=dtype), name='bn'+str(layerid)+'_m', dtype=dtype)
                #sigma
                variables['bn'+str(layerid)+'_s'] = tf.Variable(
                    tf.ones(outshape,dtype=dtype), name='bn'+str(layerid)+'_s', dtype=dtype)
                #gamma
                variables['bn'+str(layerid)+'_g'] = tf.Variable(
                    tf.ones(outshape,dtype=dtype), name='bn'+str(layerid)+'_g', dtype=dtype)
                #beta
                variables['bn'+str(layerid)+'_b'] = tf.Variable(
                    tf.zeros(outshape,dtype=dtype), name='bn'+str(layerid)+'_b', dtype=dtype)
                #add batch norm layer
                network.append(tf.nn.batch_normalization(network[-1],
                               mean=variables['bn'+str(layerid)+'_m'],
                               variance=variables['bn'+str(layerid)+'_s'],
                               offset=variables['bn'+str(layerid)+'_b'],
                               scale=variables['bn'+str(layerid)+'_g'],
                               variance_epsilon=1.e-4,
                               name='bn'+str(layerid)))
            else:
                bshape = (variables['conv'+str(layerid)+'_w'].shape[3])
                variables['conv'+str(layerid)+'_b'] = tf.Variable(
                    tf.zeros(bshape,dtype=dtype), name='conv'+str(layerid)+'_b', dtype=dtype)
                #add bias
                if dtype!=tf.float16:
                    network.append(tf.nn.bias_add(network[-1],
                                                  variables['conv'+str(layerid)+'_b'],
                                                  data_format=data_format))
                else:
                    print("Warning: bias-add currently snot supported for fp16!")

            #add relu unit
            network.append(activation(network[-1]))
        
            #add maxpool
            kshape = [1,1,2,2]
            sshape = [1,1,2,2]
            if data_format == "NHWC":
                kshape=[1,2,2,1]
                sshape=[1,2,2,1]
            network.append(tf.nn.max_pool(network[-1],
                                          ksize=kshape,
                                          strides=sshape,
                                          padding=args['conv_params']['padding'],
                                          data_format=data_format,
                                          name='maxpool'+str(layerid)))
        
            #add dropout
            network.append(tf.nn.dropout(network[-1],
                                         keep_prob=variables['keep_prob_'],
                                         name='drop'+str(layerid)))

        #reshape
        outsize = np.prod(network[-1].shape[1:]).value
        #with tf.device(device):
        network.append(tf.reshape(network[-1], shape=[-1, outsize], name='flatten'))
    
        #now do the MLP
        #fc1
        variables['fc1_w'] = tf.Variable(
            initializer([outsize, args['num_fc_units']], dtype=dtype),
            name='fc1_w', dtype=dtype)
        variables['fc1_b'] = tf.Variable(
            tf.zeros([args['num_fc_units']], dtype=dtype),
            name='fc1_b', dtype=dtype)
        network.append(tf.matmul(network[-1], variables['fc1_w']) + variables['fc1_b'])
    
        #add relu unit
        network.append(activation(network[-1]))
    
        #add dropout
        network.append(tf.nn.dropout(network[-1],
                                     keep_prob=variables['keep_prob_'],
                                     name='drop'+str(layerid)))
        #fc2
        variables['fc2_w'] = tf.Variable(
            initializer([args['num_fc_units'], 2], dtype=dtype),
            name='fc2_w', dtype=dtype)
        variables['fc2_b'] = tf.Variable(
            tf.zeros([2], dtype=dtype), name='fc2_b', dtype=dtype)
        network.append(tf.matmul(network[-1], variables['fc2_w']) + variables['fc2_b'])
        
        #add softmax
        network.append(tf.nn.softmax(network[-1]))
    
    #return the network and variables
    return variables,network

def build_functions(args,variables,network):
    """Build Functions from the Network Output"""
    
    with args['graph'].as_default():
    
        #additional variables
        variables['labels_'] = tf.placeholder(
            tf.int32, shape=[args['train_batch_size_per_node'], 1])
        variables['weights_'] = tf.placeholder(
            args["precision"], shape=[args['train_batch_size_per_node'], 1])
    
        #loss function
        prediction = network[-1]
        tf.add_to_collection('prediction_op', prediction)
    
        #compute loss, important: use unscaled version!
        loss = tf.losses.sparse_softmax_cross_entropy(variables['labels_'],
                                                      network[-2],
                                                      weights=variables['weights_'])
    
        #compute accuracy
        accuracy = tf.metrics.accuracy(variables['labels_'],
                                       tf.round(prediction[:,1]),
                                       weights=variables['weights_'],
                                       name='accuracy')
    
        #compute AUC
        auc = tf.metrics.auc(variables['labels_'],
                             prediction[:,1],
                             weights=variables['weights_'],
                             num_thresholds=5000,
                             curve='ROC',
                             name='AUC')
    
    #return functions
    return variables, prediction, loss, accuracy, auc
