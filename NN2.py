''' On NERSC
    module load python/3.6-anaconda-4.4 
    salloc -N 1 -q interactive -C haswell -t 15:00
    srun -n 4 python NN.py 
'''
import tensorflow as tf                    # NN stuff
import numpy as np                         # numerical python 
import os

class preprocess(object):
    def __init__(self, data):
        self.X = data['features']
        self.Y = data['label'][:,np.newaxis]
        self.P = data['hpix']
        self.W = data['fracgood'][:, np.newaxis]
        if len(self.X.shape) == 1:
            self.X = self.X[:,np.newaxis]
        #self.Xs = None
        #self.Ys = None
        
    
class Netregression(object):
    """
        class for a general regression
    """
    def __init__(self, train, valid, test):
        # data (Traind, Testd) should have following attrs,
        #
        # features i.e.  X
        # label    i.e. Y = f(X) + noise ?
        # hpix     i.e. healpix indices to keep track of data
        # fracgood i.e. weight associated to each datapoint eg. pixel
        #
        # train
        self.train = preprocess(train)
        # test
        self.test  = preprocess(test)
        # validation
        self.valid = preprocess(valid)
        
        #
        # one feature or more
        self.nfeatures = self.train.X.shape[1]
        
        
    def train_evaluate(self, learning_rate=0.001,
                       batchsize=100, nepoch=10, nchain=5,
                      Units=[10,10]):
        #
        nfeature = self.nfeatures
        nclass   = 1            # for classification, you will have to change this
        #
        # set up the model x [input] -> y [output]
        x   = tf.placeholder(tf.float32, [None, nfeature])
        #
        # linear, one hidden layer or 2 hidden layers 
        # need to modify this if more layers are desired
        # tf.layers.dense works like f(aX+b) where f is activation
        if Units == 'Lin':    # linear
            y  = tf.layers.dense(x, units=nclass, activation=None)
        elif len(Units) == 1: # 1 hidden layer
            y0 = tf.layers.dense(x,  units=Units[0], activation=tf.nn.relu)
            y  = tf.layers.dense(y0, units=nclass, activation=None)
        elif len(Units) == 2:                                    # 2 hidden layers
            y0 = tf.layers.dense(x,  units=Units[0], activation=tf.nn.relu)
            y1 = tf.layers.dense(y0, units=Units[1], activation=tf.nn.relu)
            y  = tf.layers.dense(y1, units=nclass,   activation=None)
        elif len(Units) == 3:
            y0 = tf.layers.dense(x,  units=Units[0], activation=tf.nn.relu)
            y1 = tf.layers.dense(y0, units=Units[1], activation=tf.nn.relu)
            y2 = tf.layers.dense(y1, units=Units[2], activation=tf.nn.relu)
            y  = tf.layers.dense(y2, units=nclass,   activation=None)
        elif len(Units) == 4:
            y0 = tf.layers.dense(x,  units=Units[0], activation=tf.nn.relu)
            y1 = tf.layers.dense(y0, units=Units[1], activation=tf.nn.relu)
            y2 = tf.layers.dense(y1, units=Units[2], activation=tf.nn.relu)
            y3 = tf.layers.dense(y2, units=Units[3], activation=tf.nn.relu)
            y  = tf.layers.dense(y3, units=nclass,   activation=None)
        else:
            raise ValueError('Units should be either None, [M], [M,N] ...')
        #
        # placeholders for the input errorbar and label
        y_  = tf.placeholder(tf.float32, [None, nclass])
        w   = tf.placeholder(tf.float32, [None, nclass])
        
        #
        # objective function
        mse = tf.losses.mean_squared_error(y_, y, weights=w)
        
        #
        # see https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer   = tf.train.AdamOptimizer(learning_rate)
        train_step  = optimizer.minimize(mse, global_step=global_step)
        
        #
        #
        train = self.train
        valid = self.valid
        test  = self.test
        
        train_size = train.X.shape[0]
        #
        # using training label/feature mean and std
        # to normalize training/testing label/feature
        meanX = np.mean(train.X, axis=0)
        stdX  = np.std(train.X, axis=0)
        meanY = np.mean(train.Y, axis=0)
        stdY  = np.std(train.Y, axis=0)
        self.Xstat = (meanX, stdX)
        self.Ystat = (meanY, stdY)
        
        train.X = (train.X - meanX) / stdX
        train.Y = (train.Y - meanY) / stdY
        test.X = (test.X - meanX) / stdX
        test.Y = (test.Y - meanY) / stdY
        valid.X = (valid.X - meanX) / stdX
        valid.Y = (valid.Y - meanY) / stdY
        #
        # compute the number of training updates
        if np.mod(train_size, batchsize) == 0:
            nep = (train_size // batchsize)
        else:
            nep = (train_size // batchsize) + 1
        #
        # initialize empty lists to store MSE 
        # and prediction at each training epoch for each chain
        self.epoch_MSEs = []
        self.chain_y     = []
        for ii in range(nchain): # loop on chains
            print('chain ',ii)
            mse_list = []
            # 
            # initialize the NN
            sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()            
            for i in range(nepoch): # loop on training epochs
                #
                # save train & test MSE at each epoch
                train_loss = mse.eval(feed_dict={x:train.X, y_:train.Y, w:train.W}) # June 7th 2:20pm - change self.train.X to train.X
                valid_loss = mse.eval(feed_dict={x:valid.X, y_:valid.Y, w:valid.W})                                
                #test_loss = mse.eval(feed_dict={x:test.X, y_:test.Y, w:test.W}) # to evaluate test MSE
                mse_list.append([i, train_loss, valid_loss])
                #mse_list.append([i, train_loss, valid_loss, test_loss])  # to save test MSE
                #
                for k in range(nep): # loop on training unpdates
                    j = k*batchsize
                    if j+batchsize > train_size:
                        batch_xs, batch_ys, batch_ws = train.X[j:], train.Y[j:], train.W[j:]   # use up to the last element
                    else:
                        batch_xs, batch_ys, batch_ws = train.X[j:j+batchsize], train.Y[j:j+batchsize], train.W[j:j+batchsize]
                    # train NN at each update
                    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys, w:batch_ws})
            #
            # save the final test MSE and prediction for each chain 
            y_mse, y_pred  = sess.run((mse,y),feed_dict={x: test.X, y_: test.Y, w:test.W})
            self.chain_y.append([ii, y_pred])
            self.epoch_MSEs.append([ii, y_mse, np.array(mse_list)])

        # baseline model is the average of training label
        # baseline mse
        baselineY  = np.mean(train.Y)
        assert np.abs(baselineY) < 1.e-6, 'check normalization!'
        baseline_testmse  = np.mean(test.W  * test.Y**2)
        baseline_validmse  = np.mean(valid.W * valid.Y**2)
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
        output['train']      = self.train.P, self.train.X, self.train.Y, self.train.W 
        output['test']       = self.test.P, self.test.X, self.test.Y, self.test.W 
        output['valid']      = self.valid.P, self.valid.X, self.valid.Y, self.valid.W         
        output['epoch_MSEs'] = self.epoch_MSEs
        output['chain_y']    = self.chain_y
        output['options']    = self.optionsdic
        if indir[-1] != '/':
            indir += '/'
        if not os.path.exists(indir):
            os.makedirs(indir)
        #if not os.path.isfile(indir+name+'.npz'):   # write w a new name
        np.savez(indir+name, output)
        #else:
        #    print("there is already a file!")
        #    name = name+''.join(time.asctime().split(' '))
        #    np.savez(indir+name, output)
        print('output is saved as {} under {}'.format(name, indir))

def run_nchainlearning(indir, *arrays, **options):
    n_arrays = len(arrays)
    if n_arrays != 3:
        raise ValueError("Three arrays for train, validation  and test are required")
    net = Netregression(*arrays)
    net.train_evaluate(**options) #learning_rate=0.01, batchsize=100, nepoch=10, nchain=5
    #
    batchsize = options.pop('batchsize', 100)
    nepoch = options.pop('nepoch', 10)
    nchain = options.pop('nchain', 5)
    Units  = options.pop('Units', [10,10])
    Lrate  = options.pop('learning_rate', 0.001)
    units  = ''.join([str(l) for l in Units])
    ouname = 'reg-nepoch'+str(nepoch)+'-nchain'+str(nchain)
    ouname += '-batchsize'+str(batchsize)+'units'+units
    ouname += '-Lrate'+str(Lrate)
    #
    net.savez(indir=indir, name=ouname)        
    
def read_NNfolds(files):
    """
        Reading different folds results, 
        `files` is a list holding the paths to different folds
        uses the mean and std of training label
        to scale back the prediction 
    """
    p_true = []
    x_true = []
    y_true = []
    y_pred = []
    y_base = []
    weights = []
    for j,file_i in enumerate(files):
        d = np.load(file_i)              # read the file
        out = d['arr_0'].item()          # 
        p_true.append(out['test'][0])    # read the true pixel id
        x_true.append(out['test'][1])    # features ie. X
        y_true.append(out['test'][2].squeeze())  # true label ie. Y
        weights.append(out['test'][3].squeeze()) # weights
        #
        # loop over predictions from chains 
        # take the average of them
        # and use the mean & std of the training label to scale back
        y_avg = []
        for i in range(len(out['chain_y'])):
            y_avg.append(out['chain_y'][i][1].squeeze().tolist())    
        meanY, stdY = out['options']['stats']['ystat']
        #print(np.mean(out['train'][2]))
        #print(meanY, stdY)
        # mean of training label as baseline model
        y_base.append(np.ones(out['test'][2].shape[0])*meanY)        
        y_pred.append(stdY*np.mean(np.array(y_avg), axis=0) + meanY)
    
    # combine different folds 
    Ptrue = np.concatenate(p_true)
    Xtrue = np.concatenate(x_true)
    Ytrue = np.concatenate(y_true)
    Ypred = np.concatenate(y_pred)
    Ybase = np.concatenate(y_base)
    Weights = np.concatenate(weights)
    #print(Xtrue.shape, Ytrue.shape, Ypred.shape, Ybase.shape, Weights.shape)
    return Ptrue, Xtrue, Ytrue, Ypred, Ybase, Weights
    
if __name__ == '__main__':    
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    #
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
    # bcast
    data   = comm.bcast(data, root=0)
    config = comm.bcast(config, root=0)
    oupath = comm.bcast(oupath, root=0)
    
    #
    # run
    if rank == 0:
        print("bcast finished")
    if rank in [0, 1, 2, 3, 4]:
        print("config on rank %d is: "%rank, config)        
        fold = 'fold'+str(rank)
        print(fold, ' is being processed')
        run_nchainlearning(oupath+fold+'/',
                       data['train'][fold],
                       data['validation'][fold], 
                       data['test'][fold],
                      **config)