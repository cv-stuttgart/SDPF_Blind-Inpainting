from tqdm import tqdm
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from os.path import join, isdir, isfile
from os import listdir, makedirs, walk, rename
from skimage.transform import rotate

def readJPGorPNG(filepath):
    """
    Reads the filepath to a numpy array that is scaled between [0,1] with dtype=np.float32
    1 = white
    0 = black
    :param filepath:
    :return: a numpy array representing the images with values between [0,1] and dtype=np.float32
    """
    ext = filepath.split(".")[-1]
    if ext == "png" or ext == 'PNG' or ext == "Png":
        img = mpimg.imread(filepath)
    elif ext == "jpg" or ext=="JPG":
        img = np.array(plt.imread(filepath)/255., dtype=np.float32)
    else:
        raise ValueError('Image "%s" is neither .jpg nor .png. Please use other method to read this file.'
                         % filepath)
    return img



def getFileList(file_path):

    dat = np.load(file_path, allow_pickle=True)
    train = dat["trainingsfiles"]
    test = dat["testfiles"]

    return train, test


def convert_to_grayscale(img):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    r = tf.math.scalar_mul(0.299, r)
    g = tf.math.scalar_mul(0.587, g)
    b = tf.math.scalar_mul(0.114, b)

    gray = tf.math.add(r,g)
    gray = tf.math.add(gray, b)

    #gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return tf.expand_dims(gray, axis=2)


def convert_to_grayscaleNP(img):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def collectTFRecords_fromFolder(folder, sortedbynumber=True):
    """
    Currently traverses all subfolders on it's search for .tfrecords files! Files that are deeper in
    the directory tree are sorted to the back. Files on the same level are sorted by their folder names.
    :param folder: folder to traverse for .tfrecordsfiles
    :param sortedbynumber: if true, the files are sorted by their tailing number.
    :return: a list of found .tfrecords files, potentially sorted
    """

    tfrecordslist = []

    fol = folder
    filelist = list()
    for (dirpath, dirnames, filenames) in walk(fol):
        filelist += [join(dirpath, f) for f in filenames]

    for filepath in filelist:
        if filepath.endswith(".tfrecords"):
            tfrecordslist.append(filepath)

    if len(tfrecordslist) == 0:
        raise ValueError("No .tfrecords files found in folder %s" % folder)
    print("Added %d .tfrecords files from %s" % (len(tfrecordslist), folder))

    if sortedbynumber:
        getNumber = lambda fname: ("/".join(fname.split("/")[:-1]), int(fname.split('.')[-2].split("_")[-1]))
        tfrecordslist.sort(key=getNumber)

    print("TFRecords-List:")
    for f in tfrecordslist:
        print("\t%s" % f)
    print()

    return tfrecordslist


def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



def ensure3channels(image, notification=False):
    """
    Avoids dimension problems for grayscale input .png files.
    For those files, the dimensions returned by imread are (x,y), and not (as required) (x,y,3).

    In this case, function returns a three channel rgb image of the original grayscale image.
    The new image is produced by extending the 2d array to 3d (x,y,1), and stacking the same
    layers three times.
    For images with three channels, the function returns the input.
    """

    dims = len(image.shape)
    if dims == 2:
        #image = np.expand_dims(image, axis=2)
        image = tf.expand_dims(image, axis=2)
        if notification:
            print('expanded channels by 1')
    if dims == 1:
        #image = np.expand_dims(image, axis=1)
        #image = np.expand_dims(image, axis=2)
        image = tf.expand_dims(image, axis=1)
        image = tf.expand_dims(image, axis=2)

        if notification:
            print('expanded channels by 2')

    return image

def ensure3channelsNP(image, notification=False):
    """
    Avoids dimension problems for grayscale input .png files.
    For those files, the dimensions returned by imread are (x,y), and not (as required) (x,y,3).

    In this case, function returns a three channel rgb image of the original grayscale image.
    The new image is produced by extending the 2d array to 3d (x,y,1), and stacking the same
    layers three times.
    For images with three channels, the function returns the input.
    """

    dims = len(image.shape)


    if dims == 2:
        image = np.expand_dims(image, axis=2)
        #image = tf.expand_dims(image, axis=2)
        if notification:
            print('expanded channels by 1')
    if dims == 1:
        image = np.expand_dims(image, axis=1)
        image = np.expand_dims(image, axis=2)
        #image = tf.expand_dims(image, axis=1)
        #image = tf.expand_dims(image, axis=2)

        if notification:
            print('expanded channels by 2')

    return image

def removeAlpha(image, notification=False, name="image"):
    dims = len(image.shape)
    if dims == 3:
        if image.shape[2] == 4:
            image = image[:,:,:3]
            if notification:
                print("Removed alpha channel for %s" % name)
    return image


def ensureRGB(image, notification=False):
    numdims = len(image.shape)
    if numdims < 3:
        image = ensure3channels(image, notification=notification)

    dims = np.array(list(image.shape))

    #print(dims[2])
    #print(tf.shape(image))
    if dims[2] == 1:
        #print(image.shape)
        #print(image)
        #image = np.concatenate((image, image, image), axis=2)
        image = tf.concat([image, image, image], 2)
        #print(image.shape)
        if notification:
            print('Extended greyscale to RGB')
    #print("ensured RGB")

    return image


def convertShardedRotateMask(imagepaths, maskpaths, outpath="", numshards=4, tfrecordsbasename="sharded_records",
                             startshardidx=0, endshardidx=None, image_base="", color_type="plain", color=1):
    '''

    :param imagepaths:
    :param maskpaths:
    :param outpath:
    :param numshards:
    :param tfrecordsbasename:
    :param startshardidx:
    :param endshardidx:
    :param image_base:
    :param color_type: ["plain", "gauss"]; the color information that is included for a mask. Plain colors the mask
    in one color, while gauss adds gaussian noise
    :param color: [0,1] or None; only applicable if color_type="plain", then it sets the color which is used,
     where 1=white and 0=black. If None, the color is set by a gaussian distribution around 0.5.
    :return:
    '''

    makedirs(outpath, exist_ok=True)

    rotation = np.tile(np.array(range(4)), (len(maskpaths),1))
    rotation = np.transpose(rotation)
    rotation = rotation.flatten()
    maskpathsx4 = np.tile(maskpaths, 4)

    shardsize = int(np.ceil(float(len(imagepaths))/numshards))
    #print(imagepaths.shape)
    #print(maskpaths.shape)
    #print(maskpathsx4.shape)
    #print(shardsize)

    lowerbound = 0

    sum_px = 0
    sum_px_val = 0
    for i in range(numshards):
        #print(i)
        if lowerbound >= len(imagepaths):
            break
        #print(len(imagepaths))
        #print(lowerbound+shardsize)
        upperbound = min(lowerbound+shardsize, len(imagepaths))

        continuecondition = endshardidx is None or i < endshardidx

        if i >= startshardidx and continuecondition:
            img_red = imagepaths[lowerbound:upperbound]
            msk_red = maskpathsx4[lowerbound:upperbound]
            rot_red = rotation[lowerbound:upperbound]

            shardimgs = upperbound - lowerbound

            shardfilename = tfrecordsbasename + "_%d_%d.tfrecords" % (shardimgs, i)
            shardfile = join(outpath, shardfilename)
            px, px_val = convert(img_red, msk_red, shardfile, rotations=rot_red, image_base=image_base,
                                 color_type=color_type, color=color)
            sum_px += px
            sum_px_val += px_val

        lowerbound = upperbound

    print("Total:\n\t%d pixels\n\t%d values" % (sum_px,sum_px_val))

def rotateimage(image, rotationcode):
    """
    Rotates an image by 0, 90, 180 or 270° counter clockwise depending on the rotation code
    :param image: image to rotate
    :param rotationcode: 0=0°, 1=90°, 2=180°, 3=270°
    :return: the rotated image
    """
    rotationcode = int(rotationcode)
    angle = 0.
    if rotationcode == 0:
        angle = 0.
    elif rotationcode == 1:
        angle = 90.
    elif rotationcode == 2:
        angle = 180.
    elif rotationcode == 3:
        angle = 270.
    else:
        print("WARNING: Rotation code %d not recognized, using 0 instead. "
              "Please use 0, 1, 2 or 3 to rotate the image by 0°, 90°, 180° or 270°." % rotationcode)
    return np.array(rotate(image, angle, mode='reflect', preserve_range=True), dtype="float32")


def convert(image_paths, mask_paths, out_path, rotations=None, image_base="", color_type="plain", color=1):
    # Args:
    # image_paths   List of file-paths for the images.
    # labels        Class-labels for the images.
    # out_path      File-path for the TFRecords output file.

    print("Converting: " + out_path)
    if rotations is None:
        rotations = np.zeros((len(mask_paths)))

    # Number of images. Used when printing the progress.
    num_images = len(image_paths)

    sum_px = 0
    sum_px_val = 0

    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(out_path) as writer:
        # Iterate over all the image-paths and class-labels.
        for i, (p_i, p_m, rot) in tqdm(enumerate(zip(image_paths, mask_paths, rotations))):
            # Print the percentage-progress.

            # Load the image-file using matplotlib's imread function.
            if not image_base == "":
                p_i = image_base + p_i.split("Documents")[-1]
            img = readJPGorPNG(p_i)
            msk = readJPGorPNG(p_m)


            # Rotate mask
            msk = rotateimage(msk, rot)[:,:,0]
            # Remove alpha channel and unnecessary color channels from image
            if not len(img.shape) == 2:
                img = removeAlpha(img)
                img = convert_to_grayscaleNP(img)
            # Both mask an image have shape (xdim, ydim) now

            # Make sure images and masked images have three channels.
            img = ensure3channelsNP(img, notification=False)
            msk = ensure3channelsNP(msk, notification=False)

            (d_1,d_2,d_3) = img.shape
            sum_px += d_1*d_2*d_3
            sum_px_val += np.sum(img)

            # parameters for gaussian distribution
            g_mu = 0.436350171562004 # mu for the places dataset (selection of own_places_full)
            g_sigma = 0.2737717918458937 # standard deviation sigma for the places dataset (selection of own_places_full)

            # Mask image with mask
            imgmsk = img.copy()
            for channel in range(img.shape[-1]):
                if color_type=="plain":
                    if color is not None:
                        # 1 = white masking, 0 = black masking
                        imgmsk[np.where(1-msk)] = color
                    else:
                        colo = np.clip(np.random.normal(loc=g_mu,scale=g_sigma), 0., 1.)
                        imgmsk[np.where(1 - msk)] = colo

                if color_type=="gauss":
                    imgnse = np.clip(np.random.normal(loc=g_mu,scale=g_sigma, size=imgmsk.shape), 0., 1.)
                    imgmsk[np.where(1 - msk)] = imgnse[np.where(1 - msk)]
            dimensions = np.array(list(img.shape))

            # more storage efficiency: Store mask as integer
            msk = np.array(msk, dtype=int)

            #if i==0:
            #    print(msk.shape)
            #    print(msk.dtype)
            #    print(img.shape)
            #    print(img.dtype)
            #    print(imgmsk.shape)
            #    print(imgmsk.dtype)
            #    print(dimensions.dtype)
            #    print(dimensions.shape)

            #    fig, ax = plt.subplots(1, 3)
            #    ax[0].imshow(img[:,:,0])
            #    ax[1].imshow(imgmsk[:,:,0])
            #    ax[2].imshow(msk[:,:,0])
            #    plt.show()
            #    plt.close(fig)


            # Convert the image to raw bytes.
            img_bytes = img.tostring()
            msk_bytes = msk.tostring()
            imgmsk_bytes = imgmsk.tostring()

            # Create a dict with the data we want to save in the
            # TFRecords file. You can add more relevant data here.
            data = \
                {
                    'image': wrap_bytes(tf.compat.as_bytes(img_bytes)),
                    'mask': wrap_bytes(tf.compat.as_bytes(msk_bytes)),
                    'masked_image': wrap_bytes(tf.compat.as_bytes(imgmsk_bytes)),
                    'xdim': _int64_feature(dimensions[0]),
                    'ydim': _int64_feature(dimensions[1]),
                    'zdim': _int64_feature(dimensions[2])
                }

            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)

            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)

            # Serialize the data.
            serialized = example.SerializeToString()

            # Write the serialized data to the TFRecords file.
            writer.write(serialized)

    return sum_px, sum_px_val


def parse(serialized, grayscale=False, nomask=False):
    # Define a dict with the data-names and types we expect to
    # find in the TFRecords file.
    # It is a bit awkward that this needs to be specified again,
    # because it could have been written in the header of the
    # TFRecords file instead.
    features = \
        {
            'image': tf.io.FixedLenFeature([], tf.string),
            'mask': tf.io.FixedLenFeature([], tf.string),
            'masked_image': tf.io.FixedLenFeature([], tf.string),
            'xdim': tf.io.FixedLenFeature([], tf.int64),
            'ydim': tf.io.FixedLenFeature([], tf.int64),
            'zdim': tf.io.FixedLenFeature([], tf.int64)
        }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.io.parse_single_example(serialized=serialized,
                                             features=features)

    # Get the image as raw bytes.
    raw_image = parsed_example['image']
    raw_image_mask = parsed_example['masked_image']
    # Get image dimensions
    x = parsed_example['xdim']
    y = parsed_example['ydim']
    z = parsed_example['zdim']

    # Decode the raw bytes so it becomes a tensor with type.
    image = tf.io.decode_raw(raw_image, tf.float32)
    image_mask = tf.io.decode_raw(raw_image_mask, tf.float32)

    # The type is now uint8 but we need it to be float.
    #image = tf.reshape(tf.cast(image, tf.float32), (x, y, z))
    #image_mask = tf.reshape(tf.cast(image_mask, tf.float32), (x, y, z))
    image = tf.reshape(tf.cast(image, tf.float32), (x, y, 1))
    image_mask = tf.reshape(tf.cast(image_mask, tf.float32), (x, y, 1))

    # Correct Dimensions to rgb in case the image was grayscale but RGB is expected
    image = ensureRGB(image)
    image_mask = ensureRGB(image_mask)

    # Generate the label associated with the image.
    # preprocessing

    if grayscale:
        # Explicitly convert the images to gray scale if required to do so
        image = convert_to_grayscale(image)
        image_mask = convert_to_grayscale(image_mask)

    if not nomask:
        raw_mask = parsed_example['mask']
        mask = tf.io.decode_raw(raw_mask, tf.int64)
        #mask = tf.reshape(tf.cast(mask, tf.float32), (x, y, z))
        mask = tf.reshape(tf.cast(mask, tf.float32), (x, y, 1))

        label = tf.concat([image, mask], 2)
    else:
        label = image

    # The image and label are now correct TensorFlow types.
    return image_mask, label


def datasetPreprocessing(dataset, batch_size, nomask, grayscale, repeat=None, shuffle_buffer=0):
    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(lambda x: parse(x, grayscale=grayscale, nomask=nomask))

    # shuffle the datasets parsed data chuncks
    if not shuffle_buffer == 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)  # , seed = 1)

    # Repeat the dataset the given number of times.
    dataset = dataset.repeat(repeat)

    # Get a batch of data with the given size.
    dataset = dataset.batch(batch_size)
    # dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE) ## used in Tensorflow 1
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  ## Tensorflow 2.3.2

    return dataset


def input_fn(filenames, batch_size=1, nomask=False, grayscale=False, test=False, traintest=False, shuffle_buffer=0,
             num_test=1000, maxshards=None, sorttfrecordsshardsbynumber=True, forcedrepeat=None):
    '''
    :param traintest:
    :param shuffle_buffer: int, if >0 dataset is shuffled with this shuffle buffersize
    :param filenames: Filenames for the TFRecords files.
    :param batch_size: Return batches of this size.
    :param nomask: if True, the y-batch only consists of the clean image,
                   without the additional corruption mask
    :param grayscale: if True, the corrupted and clean image are transformed to
                      grayscale (no effect to mask, as it is black-and-white)
    :param test: if True, shuffling will be suppressed no matter what the shuffle buffer holds
    :param traintest: if True, the dataset will be split in a training and test set (see num_test parameter)
    :param num_test: number of datapoints for test set (only required when traintest=True)
    :param maxshards: maximum number of shards to use, if there are several. If None, all available shards are used.
    :param sorttfrecordsshardsbynumber: if True, the tfrecordsshards will be sorted by their number.
    This only will work, if the filenames have the structure /dir1/dir2/tfrecordsfilename_12345.tfrecords
    Here, it is important that the number is the last thing before the .tfrecords extension and has a leading "_".
    If False, the tfrecordsshards will be added as os.listdir lists them.
    :param forcedrepeat: if set to a number (and not None), this number will be set to the repeat for all datasets, regardless of
    the standard repeat values.
    :return: returns an iterator over batches of x and y,
             where x are |batch_size| corrupted images, and y the corresponding clean images
             (eventually concatenated with the belonging corruption mask).
    '''

    if isdir(filenames):
        filenames = collectTFRecords_fromFolder(filenames, sortedbynumber=sorttfrecordsshardsbynumber)
        print("Collected tfrecordsfiles from folder")
        if maxshards is not None:
            availshards = len(filenames)
            if availshards > maxshards:
                filenames = filenames[:maxshards]
                print("Only used %d out of %d available shards." % (maxshards, availshards))
                for shard in filenames:
                    print("    %s" % shard)

    # Create a TensorFlow Dataset-object which has functionality
    # for reading and shuffling data from TFRecords files.
    dataset = tf.data.TFRecordDataset(filenames=filenames)

    # Prevent shuffling of testing data
    if test:
        shuffle_buffer = 0
    print("shuffle buffer in input_fn: %d" % shuffle_buffer)

    # optional shuffling of dataset (here: first shuffling of shards that create the dataset)
    if not shuffle_buffer == 0:
        print("input_fn: shuffling dataset")
        #dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=shuffle_seed, reshuffle_each_iteration=False)
        shardbuffer = 1
        # If dataset only composed of one shard (aka .tfrecords file), we have nothing to shuffle yet.
        # If composed of several shards, set shuffle buffer to number of shards (in order to mix them all)
        if type(filenames) == list:
            shardbuffer = len(filenames)
        dataset = dataset.shuffle(buffer_size=shardbuffer, reshuffle_each_iteration=False)#, seed = 1)
        dataset = dataset.prefetch(buffer_size=batch_size)

    if traintest:
        dataset_test = dataset.take(num_test)
        dataset_train = dataset.skip(num_test)

        train_dataset = datasetPreprocessing(dataset_train, batch_size, nomask,
                                                grayscale, repeat=forcedrepeat, shuffle_buffer=shuffle_buffer)
        test_dataset = datasetPreprocessing(dataset_test, batch_size, nomask,
                                              grayscale, repeat=forcedrepeat, shuffle_buffer=shuffle_buffer)
        return train_dataset,test_dataset#(x_train, y_train), (x_test, y_test)

    else:
        dataset = datasetPreprocessing(dataset, batch_size, nomask, grayscale, repeat=forcedrepeat, shuffle_buffer=shuffle_buffer)
        return dataset


def in_out_size(nomask=False, greyscale=False):
    """
    Returns the output dimensions for (data, label) produced by input_fn, if input_fn is called
    with the following flags

    :param nomask: if True, the y-batch (label) only consists of the clean image,
                   without the additional corruption mask
    :param greyscale: if True, the corrupted and clean image are transformed to
                      grayscale (no effect to mask, as it is black-and-white)
    :return: two 3-tupel with the x-, y- and z-dimensions for data and label produced by input_fn
             with the same configuration
    """
    x_dim = None
    y_dim = None
    z_dat = 3
    z_lab = 4

    # if no mask flag is set, output only consists of GT image (gray-val mask appended)
    if nomask:
        z_lab -= 1
    # if grayscale flag is set, input and output loose 2 channels (rgb: 3 channel, gray: 1 channel)
    if greyscale:
        z_dat -= 2
        z_lab -= 2

    return (x_dim, y_dim, z_dat), (x_dim, y_dim, z_lab)




def get_img_pair(pairnumber, mask_file_list, image_file_list, image_mask_file_list):
    mask_lst = getFileList(mask_file_list)
    imge_lst = getFileList(image_file_list)
    imsk_lst = getFileList(image_mask_file_list)

    img = readJPGorPNG(imge_lst[pairnumber])
    msk = readJPGorPNG(mask_lst[pairnumber])
    imgmsk = readJPGorPNG(imsk_lst[pairnumber])

    img = ensure3channels(img, notification=False)
    imgmsk = ensure3channels(imgmsk, notification=False)

    msk = np.expand_dims(msk, axis=2)
    label = np.concatenate((img, msk), axis=2)
    print(label.shape)

    imgmsk = np.expand_dims(imgmsk, axis=0)
    label = np.expand_dims(label, axis=0)
    print('label pre' + str(label.shape))

    print(imgmsk.dtype)
    print(np.max(imgmsk))
    print(np.min(imgmsk))
    print(label.dtype)
    print(np.max(label[0,:,:,:3]))
    print(np.min(label[0,:,:,:3]))
    print(np.max(label[0,:,:,3]))
    print(np.min(label[0,:,:,3]))


    return imgmsk, label


def printshapecorrection(image):
    if len(image.shape)==3:
        if list(image.shape)[2]==1:
            image = image[:,:,0]
    return image

def visualizeOrSaveOutput(image, image_mask, save=False, savepath='', suffix='img', imageformat='png'):

    #print(image.shape)
    #print(image_mask.shape)
    #print(image[0, :, :, 0].shape)

    orig = printshapecorrection(image[0, :, :, :])
    masked = printshapecorrection(image_mask[0, :, :, :-1])
    mask = printshapecorrection(image_mask[0, :, :, -1])

    c_o = None
    c_i = None
    if len(np.array(orig.shape)) == 2:
        c_o = plt.get_cmap("binary_r")
    if len(np.array(masked.shape)) == 2:
        c_i = plt.get_cmap("binary_r")

    if not save:
        fig, ax = plt.subplots(1, 3)

        ax[0].imshow(orig)
        #print(image_mask[0, :, :, :-1].shape)
        ax[1].imshow(masked)
        ax[2].imshow(mask)

        plt.show()
        plt.close(fig)

    else:
        for imgdata, prefix, colormap in zip([orig, masked, mask], ["o", "i", "m"], [c_o, c_i, plt.get_cmap("binary_r")]):

            filename = "%s_%s.%s" % (suffix, prefix, imageformat)
            filepath = join(savepath, filename)

            if imageformat in ['png', 'jpg']:
                mpimg.imsave(filepath, imgdata, vmin=0.0, vmax=1.0, cmap=colormap, format=None, origin=None, dpi=200)

            elif imageformat in ['npy']:
                np.save(filepath, imgdata)


def saveAllImages(tfrecordsfilefolder, savefolder, imageformat="png", nomask=False, grayscale=False, maxshards=None):

    print("------------------------------------------------------------------------------------------")
    print("Saving images from %s to %s as .%s" % (tfrecordsfilefolder, savefolder, imageformat))
    print()

    makedirs(savefolder, exist_ok=True)

    #!!! important to set forcedrepeat=0, as the iteration end will indicate the end of the dataset
    x, y = input_fn_old(tfrecordsfilefolder, batch_size=1, grayscale=grayscale, nomask=nomask,
                    shuffle_buffer=0, maxshards=maxshards, forcedrepeat=1)

    counter = 0
    sess = tf.Session()
    terminated = False
    while not terminated:
        try:
            image, image_mask = sess.run([x, y])
            visualizeOrSaveOutput(image, image_mask, save=True, savepath=savefolder,
                                  suffix=str(counter), imageformat=imageformat)
            if (counter+1) % 10 == 0:
                print("\t...saved %s images" % str(counter+1))
            counter += 1


        except tf.errors.OutOfRangeError as e:
            terminated = True

    print()
    print("A total of %s images were saved." % str(counter+1))
    print("------------------------------------------------------------------------------------------")

