
import numpy as np
import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.updates import *
from lasagne.objectives import *
from lasagne.nonlinearities import *
from dataset import load_dataset
import cPickle as pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.io import imsave

def build_generator():
    # input: 100dim
    inp = InputLayer(shape=(None, 100))
    y_input = InputLayer((None, 10))
    layer = ConcatLayer((inp, y_input))
    # fully-connected layer
    layer = batch_norm(DenseLayer(layer, 1024))
    # project and reshape
    layer = batch_norm(DenseLayer(layer, 128*7*7))
    layer = ReshapeLayer(layer, ([0], 128, 7, 7))
    # two fractional-stride convolutions
    layer = batch_norm(Deconv2DLayer(layer, 64, 5, stride=2, crop='same',
                                     output_size=14))
    layer = Deconv2DLayer(layer, 1, 5, stride=2, crop='same', output_size=28,
                          nonlinearity=sigmoid)
    print ("Generator output:", layer.output_shape)
    return {"z_in":inp, "y_in": y_input, "out": layer}

def build_critic():
    from lasagne.layers import (InputLayer, Conv2DLayer, ReshapeLayer,
                                DenseLayer)
    from lasagne.layers import batch_norm
    from lasagne.nonlinearities import LeakyRectify, softmax
    lrelu = LeakyRectify(0.2)
    # input: (None, 1, 28, 28)
    layer = InputLayer(shape=(None, 1, 28, 28))
    # two convolutions
    layer = batch_norm(Conv2DLayer(layer, 64, 5, stride=2, pad='same',
                                   nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 128, 5, stride=2, pad='same',
                                   nonlinearity=lrelu))
    # fully-connected layer
    layer = batch_norm(DenseLayer(layer, 1024, nonlinearity=lrelu))
    # output layer (linear)
    layer = DenseLayer(layer, 10, nonlinearity=softmax)
    print ("critic output:", layer.output_shape)
    return layer

def print_network(l_out):
    for layer in get_all_layers(l_out):
        print layer.__class__, layer.output_shape, "" if not hasattr(layer, 'nonlinearity') else layer.nonlinearity
    print "# params:", count_params(layer)

def sample_yfake(batch_size):
    tmp = np.zeros((batch_size, 10))
    for i in range(tmp.shape[0]):
        k = np.random.randint(1, tmp.shape[1])
        tmp[i,k] = 1.
    return tmp.astype("float32")

def sample_noise(batch_size):
    return np.random.uniform(0, 1, size=(batch_size, 100)).astype("float32")

def iterator(X_full, y_full, bs):
    for b in range(0, X_full.shape[0] // bs):
        yield X_full[b*bs:(b+1)*bs], y_full[b*bs:(b+1)*bs]
    
if __name__ == '__main__':

    X_train, y_train, _, _ , _, _ = load_dataset()
    X_train_zeros = X_train[ y_train == 0 ]
    X_train = X_train[ y_train != 0 ]
    y_train = y_train[ y_train != 0 ]
    y_train = np.eye(10)[y_train.tolist()].astype("float32")

    generator = build_generator()
    discriminator = build_critic()

    print_network(generator["out"])

    print_network(discriminator)

    z = T.fmatrix('z')
    yfake = T.fmatrix('yfake')
    y = T.fmatrix('y')
    x = T.tensor4('x')

    batch_size = 128
    y_idk = np.zeros((batch_size, 10), dtype="float32")
    y_idk[:,0] += 1
    y_idk = theano.shared(y_idk)

    generator_out = get_output(
        generator['out'], 
        {generator['y_in']: yfake, generator['z_in']: z}
    )

    disc_out_real = get_output(discriminator, x)
    disc_out_real_det = get_output(discriminator, x, deterministic=True)
    disc_out_gen = get_output(discriminator, generator_out)

    disc_loss = squared_error(disc_out_real, y).mean() + squared_error(disc_out_gen, y_idk).mean()
    gen_loss = squared_error(disc_out_gen, yfake).mean()

    gen_params = get_all_params(generator['out'], trainable=True)
    
    disc_params = get_all_params(discriminator, trainable=True)

    updates = rmsprop(gen_loss, gen_params, learning_rate=1e-4)
    updates.update(rmsprop(disc_loss, disc_params, learning_rate=1e-4))

    train_fn = theano.function([z, yfake, x, y], [gen_loss, disc_loss], updates=updates)
    gen_fn = theano.function([z, yfake], generator_out)
    disc_fn = theano.function([x], disc_out_real)
    disc_fn_det = theano.function([x], disc_out_real_det)

    mode = "morph"
    assert mode in ['train', 'test', 'morph']
    if mode == "train":
        try:
            for epoch in range(100):
                gen_losses = []
                disc_losses = []
                for X_batch, y_batch in iterator(X_train, y_train, batch_size):
                    yfake_batch = sample_yfake(batch_size)
                    z_batch = sample_noise(batch_size)        
                    g, d = train_fn(z_batch, yfake_batch, X_batch, y_batch)
                    gen_losses.append(g)
                    disc_losses.append(d)
                gen_out = gen_fn(z_batch, yfake_batch)
                grid = np.zeros((28*10, 28*10))
                ctr = 0
                for i in range(10):
                    for j in range(10):
                        grid[i*28:(i+1)*28, j*28:(j+1)*28] = gen_out[ctr][0]
                        ctr += 1
                imsave(arr=grid, fname="%i.png" % (epoch+1))
                print epoch+1, np.mean(gen_losses), np.mean(disc_losses)
                wts = [ get_all_param_values(generator['out']), get_all_param_values(discriminator) ]
                with open("weights.pkl","wb") as f:
                    pickle.dump(wts, f, pickle.HIGHEST_PROTOCOL)       
        except KeyboardInterrupt:
            import pdb
            pdb.set_trace()
    elif mode == 'test':
        with open("weights.pkl") as f:
            dat = pickle.load(f)
            set_all_param_values(generator['out'], dat[0])
            set_all_param_values(discriminator, dat[1])
        
        import pdb
        pdb.set_trace()
    else:
        X_inp = theano.shared(np.zeros((1,1,28,28)).astype("float32"))
        #X_inp.set_value( X_train[0:1] )
        disc_out_shared = get_output(discriminator, X_inp, deterministic=True)
        inp_fn = theano.function([], disc_out_shared)
        y_fixed = np.zeros((1,10)).astype("float32")
        y_fixed[0,1] = 1.
        # min distance between '1' digit vector and disc out for shared
        style_loss = squared_error(disc_out_shared, y_fixed).mean()
        style_updates = rmsprop(style_loss, [X_inp], learning_rate=1e-2)
        style_fn = theano.function([], style_loss, updates=style_updates)
        grid = np.zeros((28*10, 28*10))
        for i in range(10):
            for j in range(10):
                std_img = X_inp.get_value()[0][0]
                std_img = (std_img - np.min(std_img)) / (np.max(std_img) - np.min(std_img))
                grid[i*28:(i+1)*28,j*28:(j+1)*28] = std_img
                style_fn()
                # classify the image
                print np.argmax(disc_fn_det(std_img[np.newaxis][np.newaxis]),axis=1)
                
        imsave(arr=grid,fname="style.png")
