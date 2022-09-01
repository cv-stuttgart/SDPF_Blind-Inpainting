# Blind Inpainting with Sparse Directional Parseval Frame Dictionary CNNs

This repository contains the code for our [paper](https://arxiv.org/abs/2205.06597):

> **Blind Image Inpainting with Sparse Directional Filter Dictionaries for Lightweight CNNs**,<br>
> J. Schmalfuss, E. Scheurer, H. Zeng, N. Karantzas, A. Bruhn and D. Labate<br>
> Journal of Mathematical Imaging and Vision (JMIV), 2022.

```
@Article{Schmalfuss2022BlindImageInpainting,
  author  = {Schmalfuss, Jenny and Scheurer, Erik and Zhao, Heng and Karantzas, Nikolaos and Bruhn, Andr\'{e}s and Labate, Demetrio},
  title   = {Blind image inpainting with sparse directional filter dictionaries for lightweight {CNNs}},
  journal = {Journal of Mathematical Imaging and Vision},
  year    = {2022},
  doi     = {https://doi.org/10.1007/s10851-022-01119-6}
}
``

If you use the code or parts of it, please cite the above publication.


## Installation

### Software

The code was tested with `Python 3.7.7` and the following packages

```
tensorflow   2.2.0
scipy        1.6.0
matplotlib   3.2.2
scikit-image 0.16.2
tqdm         4.47.0
```

### Dataset

To recreate the experiments from the paper, please download the [Handwriting Inpainting Dataset(s)](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2886).
It is sufficient to **only download** the zip archives containing the `.tfrecords` files (not the `png` versions; saves memory). More precisely, the relevant datasets are called `D1_White.zip`, `D2_Uniform.zip` and `D3_Gauss.zip`, while archives with a `_png.zip`-extension contain the same data as png (which is not necessary to run the code).
Each dataset has a size of about 240GB.



## Training a (Sparse) CNN


### On the Full Dataset

After downloading the dataset, a previously specified network can be trained. The available networks are:

* **GBCNN** `CNN_Specifications/GBCNN.json`
* **GBCNN-L** `CNN_Specifications/GBCNNL.json`
* **IRCNN** `CNN_Specifications/IRCNN.json`

See "CNN Architecture Modifications" for constructing further architectures.
Training a network is done via `train.py`:

```
python train.py architecture.json DS/train trainsamples -e DS/test -t testsamples epochs batchsize outpath/ -s -b shufflebuffer -n -p
```

Replace the arguments above by the following values:

* `architecture.json`: Use the path to a network-json specification (see above, GBCNN, GBCNN-L or IRCNN).
* `DS/train`: Training data, composed of path to dataset `DS`, where the subfolder `train` contains (folders that contain) training .tfrecords files.
* `trainsamples`: The number of samples in the training data; `221000` for the full training datasets.
* `-e DS/test`: *Optional* test data, composed of path to dataset `DS`, where the subfolder `test` contains (folders that contain) testing .tfrecords files.
* `-t testsamples`: *Optional (but required when test data specified)* number of samples in the test data; `4100` for the full test dataset.
* `epochs`: Number of epochs, the paper always uses `100`.
* `batchsize`: Number samples per batch, the paper uses `10`, but may require adapation according to available memory.
* `outpath/`: Folder in which the run outputs will be saved. Note that each run creates an own folder in the specified outpath.
* `-s`: *Optional*, this option shuffles the training data.
* `-b shufflebuffer`: *Optional (only required when -s is specified)*, the size of the suffle buffer. `6000` was used in the paper.
* `-n`: *Optional (highly recommended)*, sorts the datafiles by their number. This yields the same order of tfrecords files for the test dataset, and improves the comparability of test results.
* `-p`: *Optional*, saves the inpainting predictions for the test data after the final training epoch as .npz archives.


-----------------------------

**GBCNN full training** (replace dataset paths)

```
python train.py CNN_Specifications/GBCNN.json DS/train 221000 -e DS/test -t 4100 100 10 Results/ -s -b 6000 -n -p
```

**GBCNN-L full training** (replace dataset paths)

```
python train.py CNN_Specifications/GBCNNL.json DS/train 221000 -e DS/test -t 4100 100 10 Results/ -s -b 6000 -n -p
```

**IRCNN full training** (replace dataset paths)

```
python train.py CNN_Specifications/IRCNN.json DS/train 221000 -e DS/test -t 4100 100 10 Results/ -s -b 6000 -n -p
```
-----------------------------

### Reduced Dataset Sizes

To reproduce the performance evaluation over the training set size on different occlusion splits (Fig. 7 in Paper), the above evaluations can be limited to a certain amount of training data via the maximal number of used tfrecords shards `-m` or `--maxshards`.
Here it is mandatory to also specify the `-n` (sort shards) option, because sorting them will make sure that the right fraction of the shards are taken.
It is also necessary to adapt the `trainsamples` according to the number of shards.
The table below summarizes which number of shards leads to which number of training samples:


| `trainsamples`  | 10 | 100 | 1.000 | 10.000 |
|:--------------- | --:| ---:| -----:| ------:|
| `--maxshards`   |  1 |  10 |    19 |     28 |


These values can then be added to the training call:

```
python train.py architecture.json DS/train/train00 trainsamples -e DS/test/test00 -t testsamples epochs batchsize outpath/ -s -b shufflebuffer -n -p -m maxshards
```


As datasets, do not use the generic `DS/train` folders, but the more specific `DS/train/train00`, `DS/train/train05`, `DS/train/train10`, `DS/train/train15` or `DS/train/train20` subfolders, which contain only images with a certain coverage range. In the range 20-25% (`DS/train/train20`), only 1.000 samples are available, hence evaluating with 28 shards is not possible.



### CNN Architecture Modifications

This repository contains the architectures for GBCNN, GBCNN-L and IRCNN in the `CNN_Specifications` folder.
To recreate the experiments from the publication that also used other architectures, a new `json` file for this network must be created. Note that the implementation only supports architectures with 6 layers.

To create the architectures from the paper, start with the architecture `CNN_Specifications/IRCNN.json`, and replace the convolutional layers (`"type": "conv"`) by SDPF layers (`"type": "sdpf"`) whenever required.
Note, that the `sdpf` layers only support 5x5 filters, hence all layers except the third one (which has a 1x1 conv) can be replaced.


## Output Data

Each time `train.py` is called, a new experiment folder is created within the specified `outpath`.
These folders follow the naming scheme `<time>_CNN_<architecture>`, where the architecture may be `B5r-C5r-C1r-C5r-C5r-C5s` (GBCNN), `B5r-B5r-C1r-C5r-C5r-C5s` (GBCNN-L) or `C5r-C5r-C1r-C5r-C5r-C5s` (IRCNN).

Each experiment folder contains a `dataLogger.json`, containing data that was collected during the training, and the folders `checkpoints` for model checkpoints, `layerfilters` which contains the filters for each layer after the training and optionally `endpredictions` with the final predictions (if the `-p` option was specified):

```
outpath
    ⮡ <time>_CNN_<architecture>
        dataLogger.json
        ⮡ checkpoints
              cp-0010.ckpt.index
              cp-0020.ckpt.index
              ....
        ⮡ layerfilters
              Filter-0_<type>.npy
              Filter-1_<type>.npy
              Filter-2_<type>.npy
              Filter-3_<type>.npy
              Filter-4_<type>.npy
              Filter-5_<type>.npy
        ⮡ endpredictions [only if -p was specified]
              endpredictions.npz
```

### Checkpoints

By default, checkpoints are saved after every 10 epochs.

### Layerfilters

Contains .npy files with the learned filters (sparse or fully convolutional) after the full training.
Sparse directional Parseval frame filters are called `Filter-X_conv2d_lc.npy`, convolutional filters are `Filter-X_conv2d.npy`

### Endpredictions

If the final predictions were saved (`-p` option specified), the `endpredictions` folder is created and contains all final predictions in `endpredictions.npz`.



## Acknowledgements

We thank kazemSafari for the `conv2d_LC_layer.py` implementation. 
If you are interested to use sparse filter combinations with your pytorch implementation, check out this [Github repository](https://github.com/kazemSafari/convrf/tree/master/convrf).
