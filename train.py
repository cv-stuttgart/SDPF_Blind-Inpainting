import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import losses
from SPFD_Network import SPFDModel
from time import time
import sys
from tfrecords_handling import input_fn, in_out_size
from data_processing import readJSONLayer, createDateFolder, createLayerFilterFolder, \
     nameFromLayers, pretty, simpleCNNOutputCorrection, \
    createCheckpointFolder, removefloat32_dict
from data_processing import DataLogger, loadJsonString, removefloat32_dict
from custom_callbacks import l1, l1_variance, l2, l2_variance, own_mse, own_mse_variance, \
    peak_signal_noise_ratio, peak_signal_noise_ratio_variance, ssim_avg, FilterSaver, create_and_save_predictions

from os import environ, path
import numpy as np

import argparse



def check_value(value, lower_validbound=0):
    v = int(value)
    if v < lower_validbound:
        raise argparse.ArgumentTypeError("%s is an invalid int value, "
                                         "supposed to be > %d" % (value, lower_validbound))
    return v

parser = argparse.ArgumentParser(description="Train a prespecified CNN.")
parser.add_argument("layerjson",
                    help=".json file with the layer specification (full path).")

parser.add_argument("traindat",
                    help=".tfrecords file containing the training data (full path).")
parser.add_argument("trainsamples", type=lambda x: check_value(x, lower_validbound=1),
                    help="Number of training samples in .tfrecords file, should be >= 1.")
parser.add_argument("-e", "--testdat", action='append',
                    help=".tfrecords file containing the test data, if different from training data (full path).")
parser.add_argument("-t", "--testsamples", type=lambda x: check_value(x, lower_validbound=0), action="append",
                    help="Number of test samples in .tfrecords file. "
                         "If set to 0, tests are performed on trfrom Network_Implementations import data_processing, custom_callbacksaining set")

parser.add_argument("epochs", type=lambda x: check_value(x, lower_validbound=0),
                    help="Number of epochs to train the model. Only accepts values >= 0.")
parser.add_argument("batchsize", type=lambda x: check_value(x, lower_validbound=1),
                    help="Batch size for model training. Only accepts values >=1 .")

parser.add_argument("outpath",
                    help="Folder path. A folder for the logged output will be created there.")

parser.add_argument("-s", "--shuffle", action='store_true',
                    required='-b' in sys.argv or '--shufflebuffer' in sys.argv,
                    help="Shuffles the training set.")
parser.add_argument("-b", "--shufflebuffer", type=lambda x: check_value(x, lower_validbound=2), default=0,
                    required='-s' in sys.argv or "--shuffle" in sys.argv,
                    help="The shuffle buffer size for dataset shuffling. Required to be >=2."
                         "This argument is required, if the shuffle option was set.")
parser.add_argument("-m", "--maxshards", type=lambda x: check_value(x, lower_validbound=1), default=None,
                    help="If this argument is given, maximal maxshards tfrecordsshards from a folder "
                         "will be used in the training dataset. If the argument is not set, by default all available "
                         "shards are used. Also, the argument is meaningless if a single .tfrecords file "
                         "is given as training samples.")
parser.add_argument("-n", "--sorttfrecordsshardsbynumber", action='store_true',
                    help="This argument is only relevant if trainsamples or testsample is a folder containing "
                         ".tfrecords files. If set, the files are assumed to have the structure dir/name_1234.tfrecords"
                         " and are ordered by their number that comes before the .tfrecords extension, which must"
                         " be preceeded by an underscore '_'. If the file name structure is different, "
                         " sorting will not work.")
parser.add_argument("-p", "--savepredictions", action='store_true',
                    help='If this flag is set, the endpredictions will be saved as .npz files.')


print(sys.argv)
print(tf.__version__)
print(keras.__version__)


#Argument processing
args = parser.parse_args()


#Read json layer description, store file paths and create output folder
layers = readJSONLayer(args.layerjson)
tfrecords_pathtra = args.traindat
tfrecords_pathtes = args.traindat
tfrecords_allpathtes = None

num_pathtes = 0
num_pathtestdat = 0


if args.testdat is None:
    raise ValueError("No testdata given, please provide test data via the -e argument.")
else:
    # if multiple testfiles are given, choose the first one for the testing during training
    tfrecords_pathtes = args.testdat[0]
    tfrecords_allpathtes = args.testdat
    num_pathtes = len(tfrecords_allpathtes)

if args.testsamples is not None:
    num_pathtestdat = len(args.testsamples)

if not num_pathtes == num_pathtestdat:
    raise ValueError("Number of folders for test data (%d) and number of datapoints per folder (%d) do not match!" % (num_pathtes, num_pathtestdat) )


maxshards = args.maxshards
sorttfrecordsshardsbynumber = args.sorttfrecordsshardsbynumber

# correctly assign the suffle buffer for test dataset
shubutrain = 0
if args.shuffle:
    shubutrain = args.shufflebuffer

print("shufflebuffer train=%d" % (shubutrain))


(x_dat, y_dat, z_dat), (x_lab, y_lab, z_lab) = in_out_size(nomask=True, greyscale=True)
# now correct layer definintion, in case the output channels do not match:
layers = simpleCNNOutputCorrection(layers, z_lab)

custom_name = "CNN_"+nameFromLayers(layers)
(outpath, outpath_foldname, outpath_datestr) = createDateFolder(args.outpath, custom_extension=custom_name)




# prepare checkpoint saving
checkpoint_fold = createCheckpointFolder(outpath)
checkpoint_path = path.join(checkpoint_fold, "cp-{epoch:04d}.ckpt")


# prepare Model
# make model input independent of it's x/y dimensions
mod_inp_shape = (None, None, z_dat)
model = SPFDModel(layers)


model_train_vars = model.trainable_variables

# Calculate data-related parameters for training
num_img_train = args.trainsamples
# if multiple testfiles are given, choose the first one for the training testing. If none are given, test on training data
if not num_pathtestdat == 0:
    num_img_test = args.testsamples[0]
else:
    raise ValueError("The number of testsamples is 0! Please provide tests samples.")


epochs = args.epochs
batchsize = args.batchsize

steps_per_epoch = int(num_img_train/batchsize)
steps_per_test = int(num_img_test/batchsize)
#steps_per_epoch = 1
#steps_per_test = 1
print("steps per epoch: " + str(steps_per_epoch) + " epochs: " + str(epochs) + " product: " +
      str(steps_per_epoch*epochs) + " real_batches: ")
print()


testdata = []

print("Training tfrecords: %s" % tfrecords_pathtra)
train_data = input_fn(tfrecords_pathtra, batch_size=batchsize, nomask=True,
                              grayscale=True, test=False, traintest=False, shuffle_buffer=shubutrain,
                              maxshards=maxshards, sorttfrecordsshardsbynumber=sorttfrecordsshardsbynumber)

for i in range(num_pathtes):
    tfrec_path_test = tfrecords_allpathtes[i]
    print("Testing tfrecords: %s" % tfrec_path_test)
    test_data = input_fn(tfrec_path_test, batch_size=batchsize, nomask=True,
                                grayscale=True, test=True, traintest=False, shuffle_buffer=0,
                                sorttfrecordsshardsbynumber=sorttfrecordsshardsbynumber)
    testdata.append(test_data)

# Create Validation data
num_img_val = num_img_test
tfrecords_pathval = tfrecords_pathtes


# update tfrecords test files and number of samples
if tfrecords_allpathtes is None:
    tfrecords_allpathtes = [tfrecords_pathtra]
if args.testsamples is None:
    all_testsamples = [num_img_test]
else:
    all_testsamples = args.testsamples





# Initialize data logger

logger = DataLogger(custom_name, outpath)
logger.addScriptCall(str(sys.argv))
logger.addTime(outpath_datestr)
logger.addNetworkStructure(args.layerjson, layers)
logger.addLoadedFromCheckpoint(False, None)
logger.addEpochsAndBatchSize(epochs, batchsize)
logger.addLossDescription("tensorflow.keras.losses.mean_squared_error")
logger.addTrainingData(tfrecords_pathtra, num_img_train, (x_dat, y_dat, z_dat), (x_lab, y_lab, z_lab),
                       greyscale=True, shuffle=False if shubutrain == 0 else True, shufflebuffer=shubutrain)
ctr = 0
for (tfrecords_pathtes_, num_img_test_) in zip(tfrecords_allpathtes, all_testsamples):
    if ctr == 0:
        logger.addTestingData(tfrecords_pathtes_, num_img_test_, (x_dat, y_dat, z_dat), (x_lab, y_lab, z_lab),
                              greyscale=True, shuffle=False, shufflebuffer=0)
    else:
        logger.addFreeData("test_"+ tfrecords_pathtes_.split("/")[-1], tfrecords_pathtes_, num_img_test_,
                           (x_dat, y_dat, z_dat), (x_lab, y_lab, z_lab),
                            greyscale=True, shuffle=False, shufflebuffer=0)
    ctr += 1
logger.addValidationData(tfrecords_pathval, num_img_val, (x_dat, y_dat, z_dat), (x_lab, y_lab, z_lab),
                       greyscale=True, shuffle=False, shufflebuffer=0)
logger.addCheckpointPath(checkpoint_fold)



# Define Callback to save model weights during training
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1, save_freq=10*steps_per_epoch)

filtersaver = FilterSaver(model, outpath)


model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=losses.mean_squared_error,
              metrics=['mse', own_mse, own_mse_variance, l1, l1_variance, l2, l2_variance,
                       peak_signal_noise_ratio, peak_signal_noise_ratio_variance, ssim_avg])
print('Starting Training')
trainstart = time()

hist = model.fit(train_data,
                 epochs=epochs,
                 steps_per_epoch=steps_per_epoch,
                 validation_data=test_data,
                 validation_steps=steps_per_test,
                 shuffle=False,
                 callbacks=[cp_callback, filtersaver])

trainend = time()
traintime = trainend - trainstart
logger.addTrainingTime(traintime)

model.summary(line_length=120)

logger.writeToJson()

print('Training end')
print()

print(list(hist.history.keys()))
logger.addFilterFiles(filtersaver.getFilterFiles())

# All standard measurements from history to logger
measured_epochs = [x+1 for x in range(len(hist.history["loss"]))]
for measurement in list(hist.history.keys()):

    name = measurement
    if measurement.split("_")[0] == "val":
        set = "validation"
    else:
        set = "training"

    if measurement.split("_")[-1] == "variance":
        name = "_".join(name.split("_")[:-1])
        logger.addMeasuresTraining(name, set, measurement_variance=hist.history[measurement],
                                   measured_epochs=measured_epochs)
    else:
        logger.addMeasuresTraining(name, set, measurement=hist.history[measurement],
                                   measured_epochs=measured_epochs)


logger.writeToJson()

if args.savepredictions:
    predictionfiles = create_and_save_predictions(model, test_data, batchsize, steps_per_test, outpath)
    logger.addEndPredictions(predictionfiles)

logger.writeToJson()

ctr = 0
for (dat_test, dataset, numtestimages) in zip(testdata, tfrecords_allpathtes, all_testsamples):
    print('Starting Model evaluation for %s' % dataset)
    steps_per_test = int(numtestimages/batchsize)

    predictstart = time()

    hist_eval = model.evaluate(dat_test, verbose=0, steps=steps_per_test)

    predictend = time()

    if ctr == 0:
        testingdata_set = "testing"
    else:
        testingdata_set = "test_" + dataset.split("/")[-1]
    ctr += 1
    measured_epochs = [0]
    logger.addMeasuresTraining('loss', testingdata_set, measurement=[hist_eval[0]], measured_epochs=measured_epochs)
    logger.addMeasuresTraining('mse', testingdata_set, measurement=[hist_eval[1]], measured_epochs=measured_epochs)
    logger.addMeasuresTraining('mse_own', testingdata_set, measurement=[hist_eval[2]], measured_epochs=measured_epochs)
    logger.addMeasuresTraining('mse_own', testingdata_set, measurement_variance=[hist_eval[3]], measured_epochs=measured_epochs)
    logger.addMeasuresTraining('l1', testingdata_set, measurement=[hist_eval[4]], measured_epochs=measured_epochs)
    logger.addMeasuresTraining('l1', testingdata_set, measurement_variance=[hist_eval[5]], measured_epochs=measured_epochs)
    logger.addMeasuresTraining('l2', testingdata_set, measurement=[hist_eval[6]], measured_epochs=measured_epochs)
    logger.addMeasuresTraining('l2', testingdata_set, measurement_variance=[hist_eval[7]], measured_epochs=measured_epochs)
    logger.addMeasuresTraining('peak_signal_noise_ratio', testingdata_set, measurement=[hist_eval[8]], measured_epochs=measured_epochs)
    logger.addMeasuresTraining('peak_signal_noise_ratio', testingdata_set, measurement_variance=[hist_eval[9]], measured_epochs=measured_epochs)
    logger.addMeasuresTraining('ssim_avg', testingdata_set, measurement=[hist_eval[10]], measured_epochs=measured_epochs)

    print('   ... Model evaluation done.\n' )
    predicttime = predictend - predictstart
    logger.addPredictionTime(predicttime)

    pretty(logger.loggertodict())
    logger.writeToJson()

    print(sys.argv)
    print("Testing set Mean Abs Error: ${:7.4f}".format(hist_eval[1] * 1))
