import theano
from theano import tensor as T
import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import *
from lasagne.init import *
from lasagne.updates import *
from lasagne.objectives import *
from keras.preprocessing.image import ImageDataGenerator
import os
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time
import nolearn
#from keras_ports import ReduceLROnPlateau
import pickle
import sys
import gzip
from skimage.io import imsave
from util import convert_to_rgb, compose_imgs, plot_grid
from scipy.stats import norm

class ACGAN():
    def _print_network(self,l_out):
        for layer in get_all_layers(l_out):
            print layer, layer.output_shape, "" if not hasattr(layer, 'nonlinearity') else layer.nonlinearity
        print "# learnable params:", count_params(layer, trainable=True)
    def _get_idk(self, batch_size):
        y_idk = floatX(np.zeros((batch_size, self.num_classes)))
        y_idk[:,0] += 1
        return y_idk
    def __init__(self,
                 gen_fn, 
                 gen_params,
                 disc_fn,
                 disc_params,
                 latent_dim,
                 num_classes,
                 in_shp, is_grayscale,
                 cls_lambda=1.,
                 opt=adam, opt_args={'learning_rate':theano.shared(floatX(1e-3))},
                 lsgan=False, abc=(1,0,1), verbose=True):
        self.is_grayscale = is_grayscale
        self.in_shp = in_shp
        self.latent_dim = latent_dim
        self.verbose = verbose
        self.num_classes = num_classes
        generator = gen_fn(latent_dim, is_grayscale, num_classes, **gen_params)
        discriminator = disc_fn(in_shp, is_grayscale, num_classes, **disc_params)
        self.generator = generator
        self.discriminator = discriminator
        if verbose:
            self._print_network(generator['out'])
            self._print_network(discriminator['out_dummy'])
        if lsgan:
            adv_loss = squared_error
        else:
            adv_loss = categorical_crossentropy
        z = T.fmatrix('z')
        yfake = T.fmatrix('yfake') # this gets fed into the generator
        y = T.fmatrix('y') # this gets fed into the discriminator
        x = T.tensor4('x')
        generator_out = get_output(
            generator['out'], {generator['y_in']: yfake, generator['z_in']: z}
        )
        generator_out_det = get_output(
            generator['out'], {generator['y_in']: yfake, generator['z_in']: z}, deterministic=True
        )
        disc_out_real = get_output(discriminator['out_disc'], x)
        disc_out_real_det = get_output(discriminator['out_disc'], x, deterministic=True)
        cls_out = get_output(discriminator['out_cls'], x)
        cls_out_det = get_output(discriminator['out_cls'], x, deterministic=True)
        cls_out_fake = get_output(discriminator['out_cls'], generator_out)
        accuracy_cls_det = T.eq(T.argmax(cls_out_det,axis=1), T.argmax(y, axis=1)).mean()
        disc_out_gen = get_output(discriminator['out_disc'], generator_out)
        disc_out_gen_det = get_output(discriminator['out_disc'], generator_out_det, deterministic=True)
        accuracy_disc_det = T.eq(disc_out_real_det >= 0.5, y).mean()
        # auxiliary
        disc_latent_out_det = get_output(discriminator['out_disc'].input_layer, x, deterministic=True)
        # distinguish between real and fake, and also classify the data correctly
        disc_loss = adv_loss(disc_out_real, abc[0]).mean() + adv_loss(disc_out_gen, abc[1]).mean()
        if cls_lambda > 0.:
            disc_loss += cls_lambda*categorical_crossentropy(cls_out, y).mean()
        # fool disc into thinking it's real, and also try and fool the classifier
        gen_loss = adv_loss(disc_out_gen, abc[2]).mean()
        if cls_lambda > 0.:
            gen_loss += cls_lambda*categorical_crossentropy(cls_out_fake, yfake).mean()
        # updates
        gen_params = get_all_params(generator['out'], trainable=True)
        disc_params = get_all_params(discriminator['out_dummy'] if cls_lambda > 0. else discriminator['out_disc'], trainable=True)
        updates = opt(gen_loss, gen_params, **opt_args)
        updates.update(opt(disc_loss, disc_params, **opt_args))
        # functions
        if self.verbose:
            print "creating fns..."
        fn_keys = [gen_loss, disc_loss, accuracy_cls_det]
        self.train_keys = ['gen_loss', 'disc_loss', 'accuracy_cls']
        self.train_fn = theano.function([z, yfake, x, y], fn_keys, updates=updates)
        self.loss_fn = theano.function([z, yfake, x, y], fn_keys)
        self.gen_fn = theano.function([z, yfake], generator_out)
        self.gen_fn_det = theano.function([z, yfake], generator_out_det)
        self.disc_fn = theano.function([x], disc_out_real)
        self.disc_fn_det = theano.function([x], disc_out_real_det)
        self.disc_acc = theano.function([x,y], accuracy_disc_det)
        self.disc_latent_fn_det = theano.function([x], disc_latent_out_det)
        self.lr = opt_args['learning_rate']
    def save_model(self, filename):
        with gzip.open(filename, "wb") as g:
            pickle.dump({
                'gen': get_all_param_values(self.generator['out']), 'disc': get_all_param_values(self.discriminator['out_dummy']),
            }, g, pickle.HIGHEST_PROTOCOL )
    def load_model(self, filename):
        """
        filename:
        mode: what weights should we load? E.g. `both` = load
          weights for both p2p and dcgan.
        """
        with gzip.open(filename) as g:
            dd = pickle.load(g)
            set_all_param_values(self.generator['out'], dd['gen'])
            set_all_param_values(self.discriminator['out_dummy'], dd['disc'])
    def sample_yfake(self,batch_size):
        tmp = np.zeros((batch_size, self.num_classes))
        for i in range(tmp.shape[0]):
            k = np.random.randint(0, tmp.shape[1])
            tmp[i,k] = 1.
        return tmp.astype("float32")
    def sample_noise(self,batch_size):
        return np.random.uniform(0, 1, size=(batch_size, self.latent_dim)).astype("float32")
    def dump_latent(self, itr, out_file):
        """
        saves an npy file
        """
        accum = []
        for b in range(itr.N // itr.bs):
            X_batch, y_batch = itr.next()
            val = self.disc_latent_fn_det(X_batch)
            val = np.hstack((val,y_batch))
            accum.append(val)
        accum = np.asarray(accum)
        np.save(out_file, accum)
            
    def train(self, it_train, it_val,
              it_train_binary, it_val_binary,
              num_epochs,
              out_dir,
              model_dir=None, save_every=10,
              resume=False, reduce_on_plateau=False, schedule={}, quick_run=False):
        """
        it_train: training iterator which only returns in-distribution examples
          (and their corresponding classes)
        it_val:
        it_train_binary: training iterator which returns in and out-of-distribution examples
          (and binary class, so 0 = OOD, 1 = ID)
        num_epochs:
        out_dir:
        model_dir:
        save_every:
        resume:
        reduce_on_plateau:
        schedule:
        quick_run:
        """
        def _loop(fn, itr):
            rec = [ [] for i in range(len(self.train_keys)) ]
            for b in range(itr.N // itr.bs):
                # update idk vector
                A_batch, B_batch = it_train.next()
                yfake_batch = self.sample_yfake(itr.bs)
                z_batch = self.sample_noise(itr.bs)
                results = fn(z_batch, yfake_batch, A_batch, B_batch)
                for i in range(len(results)):
                    rec[i].append(results[i])
                if quick_run:
                    break
            return tuple( [ np.mean(elem) for elem in rec ] )
        header = ["epoch"]
        for key in self.train_keys:
            header.append("train_%s" % key)
        for key in self.train_keys:
            header.append("valid_%s" % key)
        for key in ["train_score_0", "train_score_1", "train_score_noise", "valid_score_0", "valid_score_1", "valid_score_noise"]:
            header.append(key)
            header.append(key + "_std")
        header.append("lr")
        header.append("time")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if model_dir != None and not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if self.verbose:
            try:
                from nolearn.lasagne.visualize import draw_to_file
                draw_to_file(get_all_layers(self.generator['out']), "%s/gen.png" % out_dir, verbose=True)
                draw_to_file(get_all_layers(self.discriminator), "%s/disc.png" % out_dir, verbose=True)
            except:
                pass
        f = open("%s/results.txt" % out_dir, "a" if resume else "wb")
        if not resume:
            f.write(",".join(header)+"\n"); f.flush()
            print ",".join(header)
        #cb = ReduceLROnPlateau(self.lr,verbose=self.verbose)
        if self.verbose:
            print "training..."
        for e in range(num_epochs):
            try:
                if e+1 in schedule:
                    self.lr.set_value( schedule[e+1] )
                out_str = []
                out_str.append(str(e+1))
                t0 = time()
                # training
                results = _loop(self.train_fn, it_train)
                for i in range(len(results)):
                    out_str.append(str(results[i]))
                if reduce_on_plateau:
                    cb.on_epoch_end(np.mean(recon_losses), e+1)
                # validation
                results = _loop(self.loss_fn, it_val)
                for i in range(len(results)):
                    out_str.append(str(results[i]))
                # binary validation
                # TODO make cleaner
                def _loop_binary(itr):
                    zero_scores, one_scores, noise_scores = [], [], []
                    for b in range(itr.N // itr.bs):
                        this_x, this_y = itr.next()
                        assert np.min(this_y) == 0 and np.max(this_y) == 1
                        this_pred = self.disc_fn_det(this_x)
                        preds_with_gt_0 = this_pred[ this_y == 0 ] ## what if empty??
                        preds_with_gt_1 = this_pred[ this_y == 1 ] ## ??
                        zero_scores += preds_with_gt_0.flatten().tolist()
                        one_scores += preds_with_gt_1.flatten().tolist()
                        noise = floatX(np.random.normal(0,0.1,size=(itr.bs, 1 if self.is_grayscale else 3, self.in_shp, self.in_shp)))
                        noise_pred = self.disc_fn_det(noise)
                        noise_scores += noise_pred.flatten().tolist()
                    return np.mean(zero_scores), np.std(zero_scores), np.mean(one_scores), np.std(one_scores), np.mean(noise_scores), np.std(noise_scores)
                for elem in _loop_binary(it_train_binary):
                    out_str.append(str(elem))
                for elem in _loop_binary(it_val_binary):
                    out_str.append(str(elem))
                out_str.append(str(self.lr.get_value()))
                out_str.append(str(time()-t0))
                out_str = ",".join(out_str)
                print out_str
                f.write("%s\n" % out_str); f.flush()
                # PLOT SAMPLES
                dump_grid = "%s/grid" % out_dir
                for path in [dump_grid]:
                    if not os.path.exists(path):
                        os.makedirs(path)
                yfake_batch = self.sample_yfake(100)
                z_batch = self.sample_noise(100)
                gen_out = self.gen_fn_det(z_batch, yfake_batch)
                grid = np.zeros((self.in_shp*10, self.in_shp*10, 3))
                ctr = 0
                for i in range(10):
                    for j in range(10):
                        grid[i*28:(i+1)*28, j*28:(j+1)*28, :] = convert_to_rgb(gen_out[ctr], self.is_grayscale)
                        ctr += 1
                imsave(arr=grid, fname="%s/%i.png" % (dump_grid, e+1))
                if model_dir != None and (e+1) % save_every == 0:
                    self.save_model("%s/%i.model" % (model_dir, e+1))
            except KeyboardInterrupt:
                import pdb
                pdb.set_trace()
            



def build_generator(latent_dim, is_grayscale, num_classes):
    # input: 100dim
    inp = InputLayer(shape=(None, 100))
    y_input = InputLayer((None, num_classes))
    layer = ConcatLayer((inp, y_input))
    # fully-connected layer
    layer = batch_norm(DenseLayer(layer, 1024))
    # project and reshape
    layer = batch_norm(DenseLayer(layer, 128*7*7))
    layer = ReshapeLayer(layer, ([0], 128, 7, 7))
    # two fractional-stride convolutions
    layer = batch_norm(Deconv2DLayer(layer, 64, 5, stride=2, crop='same',
                                     output_size=14))
    layer = Deconv2DLayer(layer, 1 if is_grayscale else 3, 5, stride=2, crop='same', output_size=28,
                          nonlinearity=sigmoid if is_grayscale else tanh)
    return {"z_in":inp, "y_in": y_input, "out": layer}

def build_critic(in_shp, is_grayscale, num_classes, out_nonlinearity):
    lrelu = LeakyRectify(0.2)
    # input: (None, 1, 28, 28)
    layer = InputLayer(shape=(None, 1 if is_grayscale else 3, in_shp, in_shp))
    # two convolutions
    layer = batch_norm(Conv2DLayer(layer, 64, 5, stride=2, pad='same',
                                   nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 128, 5, stride=2, pad='same',
                                   nonlinearity=lrelu))
    # fully-connected layer
    layer = batch_norm(DenseLayer(layer, 1024, nonlinearity=lrelu))
    # output layer for real/not real
    l_out_disc = DenseLayer(layer, 1, nonlinearity=out_nonlinearity)
    l_out_cls = DenseLayer(layer, num_classes, nonlinearity=softmax)
    l_out_dummy = ConcatLayer([l_out_disc, l_out_cls])
    return {"out_disc":l_out_disc, "out_cls":l_out_cls, "out_dummy":l_out_dummy}

def build_critic2(in_shp, is_grayscale, num_classes, out_nonlinearity):
    lrelu = LeakyRectify(0.2)
    # input: (None, 1, 28, 28)
    layer = InputLayer(shape=(None, 1 if is_grayscale else 3, in_shp, in_shp))
    # two convolutions
    layer = batch_norm(Conv2DLayer(layer, 64, 3, stride=2, pad='same',
                                   nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 128, 3, stride=2, pad='same',
                                   nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 256, 3, stride=2, pad='same',
                                   nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 512, 3, stride=2, pad='same',
                                   nonlinearity=lrelu))
    # fully-connected layer
    layer = batch_norm(DenseLayer(layer, 512, nonlinearity=lrelu))
    # output layer for real/not real
    l_out_disc = DenseLayer(layer, 1, nonlinearity=out_nonlinearity)
    l_out_cls = DenseLayer(layer, num_classes, nonlinearity=softmax)
    l_out_dummy = ConcatLayer([l_out_disc, l_out_cls])
    return {"out_disc":l_out_disc, "out_cls":l_out_cls, "out_dummy":l_out_dummy}


def build_critic3(in_shp, is_grayscale, num_classes, out_nonlinearity):
    lrelu = LeakyRectify(0.2)
    # input: (None, 1, 28, 28)
    layer = InputLayer(shape=(None, 1 if is_grayscale else 3, in_shp, in_shp))
    # two convolutions
    layer = batch_norm(Conv2DLayer(layer, 64, 5, stride=2, pad='same',
                                   nonlinearity=lrelu))
    layer = batch_norm(Conv2DLayer(layer, 128, 5, stride=2, pad='same',
                                   nonlinearity=lrelu))
    # fully-connected layer
    layer = batch_norm(DenseLayer(layer, 512, nonlinearity=lrelu))
    # output layer for real/not real
    l_out_disc = DenseLayer(layer, 1, nonlinearity=out_nonlinearity)
    l_out_cls = DenseLayer(layer, num_classes, nonlinearity=softmax)
    l_out_dummy = ConcatLayer([l_out_disc, l_out_cls])
    return {"out_disc":l_out_disc, "out_cls":l_out_cls, "out_dummy":l_out_dummy}



class MnistIterator():
    def __init__(self, dataset, bs, set1=(0,), set2=(1,2,3,4,5,6,7,8,9,), mode='set1_mc' ):
        """
        dataset: train, valid, or test?
        bs: batch size
        set1: which examples are out-of-distribution?
        set2: which examples are in-distribution?
        which_set: yield set 0 (only anom. examples) or set 1 (only in-dist examples),
          or set 2 (yield both sets)
        make_binary: if True then all classes in set1 will be class 0, and all classes
          in set2 will be class 1
        returns: an iterator
        """
        assert dataset in ['train', 'valid', 'test']
        assert mode in ['set1_mc', 'set2_mc', 'set12_binary'] # TODO: this is fugly
        from load_mnist import load_dataset
        X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset()
        if dataset == 'train':
            self.X_dat, self.y_dat = X_train, y_train
        elif dataset == 'valid':
            self.X_dat, self.y_dat = X_valid, y_valid
        else:
            self.X_dat, self.y_dat = X_test, y_test
        make_binary = False
        if mode == 'set1_mc':
            target_set = set1
        elif mode == 'set2_mc':
            target_set = set2
        else:
            target_set = tuple(list(set1) + list(set2))
            make_binary = True
        # X_buf, y_buf will accumulate the digits corresponding
        # to the classes we're interested in
        X_buf, y_buf = [], []
        for idx, cls in enumerate(target_set):
            if idx==0:
                X_buf = self.X_dat[ self.y_dat == cls ]
                y_buf = self.y_dat[ self.y_dat == cls ]
            else:
                X_buf = np.vstack((X_buf, self.X_dat[ self.y_dat == cls ]))
                y_buf = np.hstack((y_buf, self.y_dat[ self.y_dat == cls ]))
        X_buf = np.asarray(X_buf, dtype=self.X_dat.dtype)
        y_buf = np.asarray(y_buf, dtype=self.y_dat.dtype)
        # we must remap the classes now, e.g. [2,5,6] -> [0,1,2]
        dd = {}
        for i in range(len(target_set)):
            dd[ target_set[i] ] = i
        if make_binary:
            # if cls in set0 then set -> 0, else set -> 1
            for i in range(len(y_buf)):
                y_buf[i] = 1 if y_buf[i] in set1 else 0
        else:
            for i in range(len(y_buf)):
                y_buf[i] = dd[y_buf[i]]
        self.X_dat = X_buf
        self.y_dat = y_buf
        self.N = self.X_dat.shape[0]
        self.bs = bs
        self.make_binary = make_binary
        self.itr = self._iterate()
        self.num_classes = len(target_set)
        self.target_set = target_set
        print "X, y shapes =", self.X_dat.shape, self.y_dat.shape
    def _iterate(self):
        while True:
            idxs = [x for x in range(self.X_dat.shape[0])]
            np.random.shuffle(idxs)
            self.X_dat = self.X_dat[idxs]
            self.y_dat = self.y_dat[idxs]
            for b in range(self.X_dat.shape[0] // self.bs):
                this_y = self.y_dat[b*self.bs:(b+1)*self.bs]
                if self.make_binary or self.num_classes==1:
                    this_y = floatX(this_y.reshape((len(this_y),1)))
                else:
                    this_y = floatX(np.eye(self.num_classes)[this_y])
                    
                this_x = floatX(self.X_dat[b*self.bs:(b+1)*self.bs])
                yield this_x, this_y
    def __iter__(self):
        return self
    def next(self):
        return self.itr.next()

if __name__ == '__main__':
    


    def test_iterator(mode):
        itr_valid_disc = MnistIterator(
            mode='valid',
            bs=32,
            set1=(0,1,2),
            set2=(3,4,5,),
            which_set=3)
        for xx, yy in itr_valid_disc:
            print xx.shape, yy.shape
            print yy
            break


    def test1b(mode):
        assert mode in ['train', 'dump']
        md = ACGAN(build_generator, {}, build_critic, {'out_nonlinearity':linear},
                   latent_dim=100, num_classes=9, in_shp=28, is_grayscale=True, lsgan=True, cls_lambda=0.1)
        itr_train = MnistIterator('train', 128) # 1..9 train
        itr_valid = MnistIterator('valid', 128) # 1..9 valid
        itr_train_disc = MnistIterator('train', 128, make_binary=True) # 0/1 train
        itr_valid_disc = MnistIterator('valid', 128, make_binary=True) # 0/1 valid
        name = "test1b_fixd.shuffle.repeat5"
        if mode == 'train':
            md.train(itr_train, itr_valid, itr_train_disc, itr_valid_disc, num_epochs=100, out_dir="output/%s" % name, model_dir="models/%s" % name)
        else:
            pass


    def test1b_1vsall(mode):
        """
        1 vs rest
        """
        assert mode in ['train', 'dump']
        id_ = (1,)
        ood_ = (0,2,3,4,5,6,7,8,9,)
        md = ACGAN(build_generator, {}, build_critic, {'out_nonlinearity':linear},
                   latent_dim=100, num_classes=len(id_), in_shp=28, is_grayscale=True, lsgan=True,
                   cls_lambda=0.)
        itr_train = MnistIterator('train', 128, set1=id_, set2=ood_, mode='set1_mc') # 1 vs rest
        itr_valid = MnistIterator('valid', 128, set1=id_, set2=ood_, mode='set1_mc') # 1 vs rest
        itr_train_disc = MnistIterator('train', 128, set1=id_, set2=ood_, mode='set12_binary') # 0/1 train
        itr_valid_disc = MnistIterator('valid', 128, set1=id_, set2=ood_, mode='set12_binary') # 0/1 valid
        name = "test1b_1vsall"
        if mode == 'train':
            md.train(itr_train, itr_valid, itr_train_disc, itr_valid_disc, num_epochs=100, out_dir="output/%s" % name, model_dir="models/%s" % name)
        else:
            pass



        
    def test1b_margin(mode):
        assert mode in ['train', 'dump']
        md = ACGAN(build_generator, {}, build_critic, {'out_nonlinearity':linear},
                   latent_dim=100, num_classes=9, in_shp=28, is_grayscale=True, lsgan=True, cls_lambda=0.1, abc=(1,-1,1))
        itr_train = MnistIterator('train', 128) # 1..9 train
        itr_valid = MnistIterator('valid', 128) # 1..9 valid
        itr_train_disc = MnistIterator('train', 128, binary=True) # 0/1 train
        itr_valid_disc = MnistIterator('valid', 128, binary=True) # 0/1 valid
        name = "test1b_fixd.shuffle.repeat5.margin"
        if mode == 'train':
            md.train(itr_train, itr_valid, itr_train_disc, itr_valid_disc, num_epochs=100, out_dir="output/%s" % name, model_dir="models/%s" % name)
        else:
            pass

    def test1b_margin2(mode):
        assert mode in ['train', 'dump']
        md = ACGAN(build_generator, {}, build_critic, {'out_nonlinearity':linear},
                   latent_dim=100, num_classes=9, in_shp=28, is_grayscale=True, lsgan=True, cls_lambda=0.1, abc=(3,-3,3))
        itr_train = MnistIterator('train', 128) # 1..9 train
        itr_valid = MnistIterator('valid', 128) # 1..9 valid
        itr_train_disc = MnistIterator('train', 128, binary=True) # 0/1 train
        itr_valid_disc = MnistIterator('valid', 128, binary=True) # 0/1 valid
        name = "test1b_fixd.shuffle.repeat5.margin2"
        if mode == 'train':
            md.train(itr_train, itr_valid, itr_train_disc, itr_valid_disc, num_epochs=100, out_dir="output/%s" % name, model_dir="models/%s" % name)
        else:
            pass


        
    def test1b_c2(mode):
        assert mode in ['train', 'dump']
        md = ACGAN(build_generator, {}, build_critic2, {'out_nonlinearity':linear},
                   latent_dim=100, num_classes=9, in_shp=28, is_grayscale=True, lsgan=True, cls_lambda=0.1)
        itr_train = MnistIterator('train', 128) # 1..9 train
        itr_valid = MnistIterator('valid', 128) # 1..9 valid
        itr_train_disc = MnistIterator('train', 128, binary=True) # 0/1 train
        itr_valid_disc = MnistIterator('valid', 128, binary=True) # 0/1 valid
        name = "test1b_fixd.shuffle.repeat3.c2"
        if mode == 'train':
            md.train(itr_train, itr_valid, itr_train_disc, itr_valid_disc, num_epochs=100, out_dir="output/%s" % name, model_dir="models/%s" % name)
        else:
            pass

    def test1b_c3(mode):
        assert mode in ['train', 'dump']
        md = ACGAN(build_generator, {}, build_critic3, {'out_nonlinearity':linear},
                   latent_dim=100, num_classes=9, in_shp=28, is_grayscale=True, lsgan=True, cls_lambda=0.1)
        itr_train = MnistIterator('train', 128) # 1..9 train
        itr_valid = MnistIterator('valid', 128) # 1..9 valid
        itr_train_disc = MnistIterator('train', 128, binary=True) # 0/1 train
        itr_valid_disc = MnistIterator('valid', 128, binary=True) # 0/1 valid
        name = "test1b_fixd.shuffle.repeat3.c3"
        if mode == 'train':
            md.train(itr_train, itr_valid, itr_train_disc, itr_valid_disc, num_epochs=100, out_dir="output/%s" % name, model_dir="models/%s" % name)
        else:
            pass

        
    def test1b_nocls(mode):
        assert mode in ['train', 'dump']
        md = ACGAN(build_generator, {}, build_critic, {'out_nonlinearity':linear},
                   latent_dim=100, num_classes=9, in_shp=28, is_grayscale=True, lsgan=True, cls_lambda=0.0)
        itr_train = MnistIterator('train', 128)
        itr_valid = MnistIterator('valid', 128)
        itr_train_disc = MnistIterator('train', 128, binary=True) # 0/1 train        
        itr_valid_disc = MnistIterator('valid', 128, binary=True)
        name = "test1b_fixd.shuffle.repeat3.nocls"
        if mode == 'train':
            md.train(itr_train, itr_valid, itr_train_disc, itr_valid_disc, num_epochs=100, out_dir="output/%s" % name, model_dir="models/%s" % name)

        
    #test1b('train')

    locals()[ sys.argv[1] ]( sys.argv[2] )
    
    
    
