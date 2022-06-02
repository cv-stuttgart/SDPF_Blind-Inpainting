from tensorflow.keras import callbacks
from tensorflow.nn import moments
from tensorflow.image import psnr, ssim_multiscale,ssim
from tensorflow import reduce_sum, scalar_mul, ones, divide, \
    subtract, multiply, reduce_mean, reduce_max, reduce_min, sqrt, abs
import numpy as np
from tensorflow import shape, ones
import tensorflow as tf

from os import path, makedirs
from numpy import savez_compressed

from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export('keras.metrics.own_mse',
           'keras.losses.own_mse')
def own_mse(y_true, y_pred):
    err = y_pred - y_true
    return K.mean(math_ops.multiply(err, err), axis=-1)


@tf_export('keras.metrics.own_mse_variance',
           'keras.losses.own_mse_variance')
def own_mse_variance(y_true, y_pred):
    sq = math_ops.multiply(y_pred - y_true, y_pred - y_true)
    mean_per_img = reduce_mean(sq, axis=[1,2,3])
    mean = K.mean(mean_per_img, axis=-1)

    onesvec = tf.ones_like(tf.shape(mean_per_img), dtype=tf.float32)
    means = math_ops.scalar_mul(mean, onesvec)
    powdiff = math_ops.multiply(mean_per_img - means, mean_per_img - means)

    return K.mean(powdiff, axis=-1)


@tf_export('keras.metrics.l1',
           'keras.losses.l1')
def l1(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


@tf_export('keras.metrics.l1_variance',
           'keras.losses.l1_variance')
def l1_variance(y_true, y_pred):
    mean_per_img = reduce_mean(K.abs(y_pred - y_true), axis=[1,2,3])
    mean = K.mean(mean_per_img, axis=-1)

    onesvec = tf.ones_like(tf.shape(mean_per_img), dtype=tf.float32)
    means = math_ops.scalar_mul(mean, onesvec)
    powdiff = math_ops.multiply(mean_per_img - means, mean_per_img - means)

    return K.mean(powdiff, axis=-1)


@tf_export('keras.metrics.l2',
           'keras.losses.l2')
def l2(y_true, y_pred):
    err = y_pred - y_true
    l2_per_img = math_ops.sqrt(reduce_mean(math_ops.multiply(err, err), axis=[1, 2, 3]))
    return K.mean(l2_per_img, axis=-1)


@tf_export('keras.metrics.l2_variance',
           'keras.losses.l2_variance')
def l2_variance(y_true, y_pred):
    err = y_pred - y_true
    l2_per_img = math_ops.sqrt(reduce_mean(math_ops.multiply(err, err), axis=[1, 2, 3]))
    mean = K.mean(l2_per_img, axis=-1)

    onesvec = tf.ones_like(tf.shape(l2_per_img), dtype=tf.float32)
    means = math_ops.scalar_mul(mean, onesvec)
    powdiff = math_ops.multiply(l2_per_img - means, l2_per_img - means)

    return K.mean(powdiff, axis=-1)


@tf_export('keras.metrics.psnr',
           'keras.losses.psnr')
def peak_signal_noise_ratio(y_true, y_pred):
    p = psnr(y_pred, y_true, max_val=1.0)
    return K.mean(p, axis=-1)


@tf_export('keras.metrics.psnr_variance',
           'keras.losses.psnr_variance')
def peak_signal_noise_ratio_variance(y_true, y_pred):
    p = psnr(y_pred, y_true, max_val=1.0)
    mean = K.mean(p, axis=-1)

    onesvec = tf.ones_like(tf.shape(p), dtype=tf.float32)
    means = math_ops.scalar_mul(mean, onesvec)
    powdiff = math_ops.multiply(p - means, p - means)

    return K.mean(powdiff, axis=-1)

@tf_export('keras.metrics.ssim_avg',
           'keras.losses.ssim_avg')
def ssim_avg(y_true, y_pred):
    return tf.reduce_mean(ssim(y_true, y_pred, max_val=1.0))


@tf_export('keras.metrics.ssim_multiscale',
           'keras.losses.ssim_multiscale')
def ssim_multiscale_own(y_true, y_pred):
    p = ssim_multiscale(y_pred, y_true, max_val=1.0)
    print("SSIM:")
    print(p)
    return K.mean(p, axis=-1)


def l1_per_img(err_pp):

    absol = abs(err_pp)
    l1 = reduce_mean(absol, axis=[1, 2, 3])

    return l1


def mse_per_img(err_pp, l2=False):
    err_pp_squared = multiply(err_pp, err_pp)
    pred_mse_image = reduce_mean(err_pp_squared, axis=[1, 2, 3])

    if l2:
        l2_image = sqrt(pred_mse_image)
        return pred_mse_image, l2_image
    else:
        return pred_mse_image



class PredictionSaver(callbacks.Callback):

    def __init__(self, prediction_data, validation_steps, outputfolder, batch_size):
        self.prediction_data = prediction_data
        self.validation_steps = validation_steps
        self.batch_size = batch_size

        self.outputfolder = outputfolder
        self.predicitionfiles = []

    def getPredictionFiles(self):
        return self.predicitionfiles

    def on_train_end(self, epoch, logs={}):

        predfolder = path.join(self.outputfolder, "endpredictions")
        makedirs(predfolder, exist_ok=True)

        print('PredictionSaver Callback: Saving %d Predictions to %s' % (self.validation_steps, predfolder))

        maxsteps = self.validation_steps * self.batch_size
        step = 0
        predictions = []
        for element in self.prediction_data:
            if step < maxsteps:
                pr = self.model.predict(element)
                print("PRED-SHAPE:")
                # print(pr)
                print(pr.size)
                print(pr.shape)
                if step % self.batch_size == 0:
                    predictions = pr
                else:
                    predictions.append(pr)
                step += 1
            else:
                break

        for i in range(self.validation_steps):
            pr_data = self.prediction_data.take(self.batch_size)
            print("PR_DATA:")
            print(pr_data)
            pr = self.model.predict(pr_data, batch_size=self.batch_size)
            print("PRED-SHAPE:")
            print(pr.size)
            print(pr.shape)
            pred_path = path.join(predfolder, 'endpredictions_%d.npz' % i)
            self.predicitionfiles.append(pred_path)
            print("    ...", pred_path)

            savez_compressed(pred_path, inputs=dat, groundtruths=lab, predictions=pr[i])

        print()


def create_and_save_predictions(model, test_data, batchsize, steps_per_test, outputfolder):
    predfolder = path.join(outputfolder, "endpredictions")
    makedirs(predfolder, exist_ok=True)

    print("Starting prediction")
    pr = model.predict(test_data, batch_size=batchsize, steps=steps_per_test)
    print(pr.shape)
    print("Prediction done")
    print('Saving Predictions to %s' % (predfolder))

    pred_path = path.join(predfolder, 'endpredictions.npz')
    savez_compressed(pred_path, predictions=pr)

    return [pred_path]



class FilterSaver(callbacks.Callback):

    def __init__(self, model, outputfolder):
        self.model = model

        self.outputfolder = outputfolder
        self.filterfiles = []

    def getFilterFiles(self):
        return self.filterfiles

    def on_train_end(self, epoch, logs={}):

        filterfolder = path.join(self.outputfolder, "layerfilters")
        makedirs(filterfolder, exist_ok=True)

        print('FilterSaver Callback: Saving Filters to ', filterfolder)

        filterctr = 0
        for layer in self.model.layers:
            layername = layer.name
            print("    ...", layername)
            mixlayer = False

            if "conv2d_lc" in layername: #if the layer is a sparse layer:
                filters = tf.einsum('ijklm,klm->ijkl', layer.fixed_kernel_tensor,
                                    layer.LC_coeffs).numpy()
            elif "conv2d" in layername: #if the layer is a conv layer:
                filters = layer.kernel.numpy()
            elif "mixed_layer" in layername: #if the layer consists of a mix:
                sparselayer = layer.get_sparselayer()
                convlayer = layer.get_convlayer()
                filters = tf.einsum('ijklm,klm->ijkl', sparselayer.fixed_kernel_tensor, sparselayer.LC_coeffs).numpy()
                filters2 = convlayer.kernel.numpy()
                mixlayer = True
            else:
                try:
                    filters = layer.kernel.numpy()
                except AttributeError as e:
                    print(e)
                    print("    ...Saving filters for layer %s not possible. See error above." % (layername))
                    filters = None

            filterfilename = path.join(filterfolder, "Filter-" + str(filterctr) + "_" + layername + ".npy")
            filterctr += 1
            np.save(filterfilename, filters)
            layerfiledict = {layername: filterfilename}
            self.filterfiles.append(layerfiledict)

            if mixlayer:
                filterfilename = path.join(filterfolder, "Filter-" + str(filterctr) + "_" + layername + ".npy")
                filterctr += 1
                np.save(filterfilename, filters2)
                layerfiledict = {layername: filterfilename}
                self.filterfiles.append(layerfiledict)

        print()



class Metrics(callbacks.Callback):

    def __init__(self, validation_generator, validation_steps, session, period=1):
        self.validation_generator = validation_generator
        self.validation_steps = validation_steps
        self.session = session

        self.called_on_epochs = []
        self.own_mse_mean = []
        self.own_mse_variance = []
        self.l2_normalized_mean = []
        self.l2_normalized_variance = []
        self.l1_normalized_mean = []
        self.l1_normalized_variance = []
        self.psnr_mean = []
        self.psnr_variance = []
        self.ssim = []

        self.epochs_passed = 0
        self.period = period

    def asdict(self):
        return self.__dict__

    def on_train_begin(self, logs={}):
        self.called_on_epochs = []
        self.own_mse_mean = []
        self.own_mse_variance = []
        self.l2_normalized_mean = []
        self.l2_normalized_variance = []
        self.l1_normalized_mean = []
        self.l1_normalized_variance = []
        self.psnr_mean = []
        self.psnr_variance = []
        self.ssim = []

        self.epochs_passed = 0

    #@profile
    def on_epoch_end(self, epoch, logs={}):

        if self.epochs_passed % self.period == 0:

            gt = []
            pred = []
            sess = self.session

            for i in range(self.validation_steps):
                (dat, lab) = sess.run(self.validation_generator)
                pr = self.model.predict(dat, steps=1)
                if i == 0:
                    gt = lab
                    pred = pr
                else:
                    gt = np.concatenate((gt, lab), axis=0)
                    pred = np.concatenate((pred, pr), axis=0)

            err_pp = subtract(pred, gt)

            mse_image, l2_image = mse_per_img(err_pp, l2=True)

            mean_mse, variance_mse = moments(mse_image, axes=[0])
            self.own_mse_mean.append(sess.run(mean_mse))
            self.own_mse_variance.append(sess.run(variance_mse))
            mean_l2, variance_l2 = moments(l2_image, axes=[0])
            self.l2_normalized_mean.append(sess.run(mean_l2))
            self.l2_normalized_variance.append(sess.run(variance_l2))

            l1_image = l1_per_img(err_pp)
            mean_l1, variance_l1 = moments(l1_image, axes=[0])
            self.l1_normalized_mean.append(sess.run(mean_l1))
            self.l1_normalized_variance.append(sess.run(variance_l1))

            psnr_image = psnr(pred, gt, max_val=1.0)
            psnr_mean, psnr_var = moments(psnr_image, axes=[0])
            self.psnr_mean.append(sess.run(psnr_mean))
            self.psnr_variance.append(sess.run(psnr_var))

            ssim_mean = tf.reduce_mean(ssim(pred, gt, max_val=1.0))
            self.ssim.append(sess.run(ssim_mean))

            self.epochs_passed += 1
            self.called_on_epochs.append(self.epochs_passed)
        else:
            self.epochs_passed += 1

        return
