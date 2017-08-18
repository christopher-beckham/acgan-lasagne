
# coding: utf-8

# In[ ]:




# In[50]:

import numpy as np
import theano
from theano import tensor as T
import lasagne
from dataset import load_dataset


# In[2]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# --------

# Import datasets

# In[3]:

X_train, y_train, _, _ , _, _ = load_dataset()


# In[113]:

y_train = y_train.astype("float32")


# In[134]:

y_train = np.eye(10)[y_train.tolist()].astype("float32")


# ---------

# Define architectures

# In[37]:

# ##################### Build the neural network model #######################
# We create two models: The generator and the critic network.
# The models are the same as in the Lasagne DCGAN example, except that the
# discriminator is now a critic with linear output instead of sigmoid output.

def build_generator():
    from lasagne.layers import InputLayer, ReshapeLayer, DenseLayer, ConcatLayer
    from lasagne.layers import TransposedConv2DLayer as Deconv2DLayer
    from lasagne.layers import batch_norm
    from lasagne.nonlinearities import sigmoid
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


# In[41]:

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


# In[143]:

generator = build_generator()
discriminator = build_critic()


# In[89]:

from lasagne.layers import get_all_layers, count_params, get_output, get_all_params


# In[44]:

def print_network(l_out):
    for layer in get_all_layers(l_out):
        print layer.__class__, layer.output_shape, "" if not hasattr(layer, 'nonlinearity') else layer.nonlinearity
    print "# params:", count_params(layer)


# In[47]:

print_network(generator["out"])


# In[48]:

print_network(discriminator)


# In[144]:

z = T.fmatrix('z')
yfake = T.fmatrix('yfake')
y = T.fmatrix('y')
x = T.tensor4('x')


# In[145]:

batch_size = 32
y_idk = np.zeros((batch_size, 10), dtype="float32")
y_idk[:,0] += 1
y_idk = theano.shared(y_idk)


# In[146]:

generator


# In[147]:

generator_out = get_output(
    generator['out'], 
    {generator['y_in']: yfake, generator['z_in']: z}
)


# In[148]:

disc_out_real = get_output(discriminator, x)
disc_out_gen = get_output(discriminator, generator_out)


# In[149]:

from lasagne.objectives import squared_error


# In[150]:

disc_loss = squared_error(disc_out_real, y).mean() + squared_error(disc_out_gen, y_idk).mean()


# In[151]:

gen_loss = squared_error(disc_out_gen, yfake).mean()


# In[152]:

gen_params = get_all_params(generator['out'], trainable=True)
gen_params


# In[153]:

disc_params = get_all_params(discriminator, trainable=True)
disc_params


# In[154]:

from lasagne.updates import rmsprop


# In[155]:

updates = rmsprop(gen_loss, gen_params, learning_rate=1e-4)
updates.update(rmsprop(disc_loss, disc_params, learning_rate=1e-4))


# --------

# In[156]:

train_fn = theano.function([z, yfake, x, y], [gen_loss, disc_loss], updates=updates)


# In[157]:

def sample_yfake(batch_size):
    tmp = np.zeros((batch_size, 10))
    for i in range(tmp.shape[0]):
        k = np.random.randint(1, tmp.shape[1])
        tmp[i,k] = 1.
    return tmp.astype("float32")


# In[158]:

def sample_noise(batch_size):
    return np.random.uniform(0, 1, size=(batch_size, 100)).astype("float32")


# In[159]:

def iterator(X_full, y_full, bs):
    for b in range(0, X_full.shape[0] // bs):
        yield X_full[b*bs:(b+1)*bs], y_full[b*bs:(b+1)*bs]


# In[160]:

for epoch in range(100):
    gen_losses = []
    disc_losses = []
    for X_batch, y_batch in iterator(X_train, y_train, batch_size):
        yfake_batch = sample_yfake(batch_size)
        z_batch = sample_noise(batch_size)        
        g, d = train_fn(z_batch, yfake_batch, X_batch, y_batch)
        gen_losses.append(g)
        disc_losses.append(d)
    print epoch+1, np.mean(gen_losses), np.mean(disc_losses)


# In[ ]:



