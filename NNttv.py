''' Author : Ana Paulina Bucki
    Date   : May 2018
'''

#!/usr/bin/env python

import sys
sys.path.insert(0,"/Documents/eBOSS/eboss-dr5")
import NN

# %matplot inline
import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits

data = fits.open('./ebossELGngal-features.fits')

data.info()

DATA = data[1].data

DATA = np.array(DATA)

# everything below is from NN.py

from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np
import os

class preprocess(object):
    #
    def __init__(self, data):
        self.X = data['features']
        self.Y = data['label'][:, np.newaxis]
        self.P = data['hpix']
        self.W = data['fracgood'][:, np.newaxis]
        if len(self.X.shape) == 1:
            self.X = self.X[:, np.newaxis]
        
class Netregression(object):
    #
    def __init__(self, train, valid, test):
        # train
        self.train = preprocess(train)
        # test
        self.test  = preprocess(test)
        # validation
        self.valid = preprocess(valid)
        #
        self.nfeatures = self.train.X.shape[1]
        
    def train_evaluate(self, learning_rate=0.001, batchsize=100, 
                       nepoch=10, nchain=5, Units=[5,5,5,5]):
        #
        nfeature = self.nfeatures
        nclass   = 1
        #
        x = tf.placeholder(tf.float32, [None, nfeature])
        #
        # below layers
        #
        if (Units[0] == 0) & (Units[1] == 0 # linear (no layers)
            y = tf.layers.dense(x, units=nclass, actuvation=None)
        elif (Units[0] == 0) or (Units[1] == 0): # 1 layer [10,0] etc.
            m  = int(Units[0] + Units[1])
            y0 = tf.layers.dense(x,  units=m, activation=tf.nn.relu)
            y  = tf.layers.dense(y0, units=nclass, activation=None)
            Units = [0, m]
        elif (Units[0] != 0) & (Units[1] !=0) & (len(Units) == 2)                # 2 layers
            y0 = tf.layers.dense(x,  units=Units[0], activation=tf.nn.relu)
            y1 = tf.layers.dense(y0, units=Units[1], activation=tf.nn.relu)
            y  = tf.layers.dense(y1, units=nclass,   activation=None)
        elif (Units[0] != 0) & (Units[1] !=0) & (Units[2] !=0) (len(Units) == 3) # 3 layers
            y0 = tf.layers.dense(x,  units=Units[0], activation=tf.nn.relu)
            y1 = tf.layers.dense(y0, units=Units[1], activation=tf.nn.relu)
            y2 = tf.layers.dence(y1, units=Units[2], activation=tf.nn.relu)                  
            y  = tf.layers.dense(y2, units=nclass,   activation=None)
        #
        # Finish adding layers here
        #
        y_ = tf.placeholder(tf.float32, [None, nclass])
        w  = tf.placeholder(tf.float32, [None, nclass])
        #
        mse = tf.losses.mean_squared_error(y_, y, weights=w)
        #
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer   = tf.train.AdamOptimizer(learning_rate)
        train_step  = optimizer.minimize(mse, global_step=global_step)
        #
        train = self.train
        valid = self.valid
        test  = self.test
        #
        train_size = train.X.shape[0]
        #
        # to normalization
        #
        meanX = np.mean(train.X, axis=0)
        stdX  = np.std(train.X,  axis=0)
        meanY = np.mean(train.Y, axis=0)
        stdY  = np.std(train.Y,  axis=0)
        self.Xstat = (meanX, stdX)
        self.Ystat = (meanY, stdY)
        #
        train.X = (train.X - meanX) / stdX
        train.Y = (trian.Y - meanY) / stdY
        test.x  = (test.X - meanX)  / stdX
        test.Y  = (test.Y - meanY)  / stdY
        valid.X = (valid.X - meanX) / stdX
        valid.Y = (valid.Y - meanY) / stdY
        #
        # to compute the number of training epochs (stops when RMSE normalizes?)
        #
        if np.mod(train_size, batchsize) == 0:
            nep = (train_size // batchsize)
        else:
            nep = (train_size // batchsize) + 1
        #
        # storing MSE
        #
        self.epoch_MSEs = []
        self.chain_y    = []
        for ii in range(nchain):
            print('chain ',ii)
            mse_list = []
            #
            # initializing NN
            #
            sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()
            for i in range(nepoch):
                #
                train_loss = mse.eval(feed_dict={x:train.X, y_:train.Y, w:train.W})
                valid_loss = mse.eval(feed_dict={x:valid.X, y_:valid.Y, w:valid.W})
                #
                mse_list.append([i, train_loss, valid_loss])
                #
                for k in range(nep):
                    j = k*batchsize
                    if j+batchsize > train_size:
                        batch_xs, batch_ys, batch_ws = train.X[j:-1], train.Y[j:-1], train.W[j:-1]
                    else:
                        batch_xs, batch_ys, batch_ws = train.X[j:j+batchsize], train.Y[j:j+batchsize], train.W[j:j+batchsize]
                    #
                    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys, w:batch_ws})
                    #
            y_mse, y_pred = sess.run((mse.y),feed_dict={x: test.X, y_: test.Y, x:test.W})
            self.chain_y.append([ii, y_pred])
            self.epoch_MSEs.append([ii, y_mse, np.array(mse_list)])
        
        baselineY = np.mean(train.Y)
        assert np.abs(baselineY) < 1.e-6, 'check normalization'
        baseline_testmse  = np.mean(test.W * test.Y**2)
        baseline_validmse = np.mean(valid.W * valid.Y**2)
        baseline_trainmse = np.mean(train.W * train.Y**2)
        #
        self.optionsdic = {}
        self.optionsdic['baselineMSE']   = (baseline_trainmse, baseline_validmse, baseline_testmse)
        self.optionsdic['learning_rate'] = learning_rate
        self.optionsdic['batchsize']     = batchsize
        self.optionsdic['nepoch']        = nepoch
        self.optionsdic['nchain']        = nchain
        self.optionsdic['Units']         = Units
        self.optionsdic['stats']         = {'xstat':self.Xstat, 'ystat':self.Ystat}
        
    def savez(self, indir='./', name='regression_2hl_5chain_10epoch'):
        output = {}
        output['train'] = self.train.P, self.train.X, self.train.Y, self.train.W
        output['test']  = self.test.P, self.test.X, self.test.Y, self.test.W
        output['valid'] = self.valid.P, self.valid.X, self.valid.Y, self.valid.W
        output['epoch_MSEs'] = self.epoch_MSEs
        output['chain_y'] = self.chain_y
        output['options'] = self.optionsdic
        #
        if indir[-1] != '/':
            indir += '/'
        if not os.path.exists(indir):
            os.makedirs(indir)
        np.savez(indir+name, output)
        print ('output is saved as {} under {}'.format(name, indir))
        
    def run_nchainlearning(indir, *arrays, **options):
        n_arrays = len(arrays)
        if n_arrays != 2:
            raise ValuseError("Two arrays for train and test are required")
        net = Netregression(*arrays)
        net.train_evaluate(**options)
        #
        batchsize = options.pop('batchsize', 100)
        nepoch    = options.pop('nepoch', 10)
        nchain    = options.pop('nchain', 5)
        Units     = options.pop('Units', [5,5,5,5])
        Lrate     = options.pop('learning_rate', 0.001)
        units     = ''.join([str(l) for l in Units])
        #
        ouname = 'reg-nepoch'+str(nepoch)+'-nchain'+str(nchain)
        ouname += '-batchsize'+str(batchsize)+'units'+units
        ouname += '-Lrate'+str(Lrate)
        #
        net.savez(indir=indir, name=ouname)

def read_NNfolds(files):
    
    p_true  = []
    x_true  = []
    y_true  = []
    y_pred  = []
    y_base  = []
    weights = []
    #
    for j,file_i in enumerate(files):
        d = np.load(file_i)
        out = d['arr_0'].item()
        p_true.append(out['test'][0])
        x_true.append(out['test'][1])
        y_true.append(out['test'][2].squeeze())
        weights.append(out['test'][3].squeeze())
        #
        y_avg = []
        #
        for i in range(len(out['chain_y'])):
            y_avg.append(out['chain_y'][i][1].squeeze().tolist())
        meanY, std_Y = out['options']['stats']['ystat']
        #
        y_base.append(np.ones(out['test'][2].shape[0])*meanY)
        y_pred.append(stdY*np.mean(np.array(y_avg), axis=0) + meanY)

    # Combining folds
    Ptrue   = np.concatenate(p_true)
    Xtrue   = np.concatenate(x_true)
    Ytrue   = np.concatenate(y_true)
    Ypred   = np.concatenate(y_pred)
    Ybase   = np.concatenate(y_base)     
    Weights = np.concatenate(weights) 
    #
    return Ptrue, Xtrue, Ytrue, Ypred, Ybase, Weights

if __name__== '__main__':
    from mpi4py import MPI
    #
    comm = MPI.COMM_WOLRD
    size = comm.Get_size()
    rank = comm.Get_rank()
    #
    if rank == 0:
        from argparse import ArgumentParser
        ap = ArgumentParser(description='Neural Net regression')
        ap.add_argument('--path',   default='/global/cscratch1/sd/mehdi/dr5_anand/eboss/')
        ap.add_argument('--input',  default='test_train_eboss_dr5-masked.npy')
        ap.add_argument('--output', default='/global/cscratch1/sd/mehdi/dr5_anand/eboss/regression/')
        ap.add_argument('--nchain', type=int, default=10)
        ap.add_argument('--nepoch', type=int, default=1000)
        ap.add_argument('--batchsize', type=int, default=8000)
        ap.add_argument('--units', nargs='*', type=int, default=[10,10])
        ap.add_argument('--learning_rate', type=float, default=0.01)
        ns = ap.parse_args()
        #
        data   = np.load(ns.path+ns.input).item()
        config = {'nchain':ns.nchain,
                  'nepoch':ns.nepoch,
                  'batchsize':ns.batchsize,
                  'Units':ns.units,
                  'learning_rate':ns.learning_rate}
        oupath = ns.output
    else:
        oupath = None
        data   = None
        config = None
    #
    data   = comm.bcast(data,   root=0)
    config = comm.bcast(config, root=0)
    oupath = comm.bcast(oupath, root=0)
    #
    # run
    if rank == 0:
        print("bcast finished")
    if rank in [0, 1, 2, 3]:
        print("config on rank %d is: "%comm.rank, config)
        fold = 'fold'+str(rank)
        print(fold, ' is being processed')
        run_nchainlearning(oupath+fold+'/',
                           data['train'][fold],
                           data['test'][fold],
                           **config)
# end 
