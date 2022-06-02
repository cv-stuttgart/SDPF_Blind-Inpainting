import json
from datetime import datetime
from os import makedirs, path, listdir
# !!! contains import from tfrecords_32x32 around line 385
# !!! contains import from tfrecords_script_triangles around line 385



class DataLogger:

    def __init__(self, name, output_folder):
        # Initialize all previously given parameters
        self.parameters = {}
        self.parameters["name"] = name
        self.parameters["output_folder"] = output_folder
        json_path = path.join(output_folder, "dataLogger.json")
        self.parameters["json_path"] = json_path
        self.parameters["script_call"] = ""

        self.parameters["time"] = None
        self.parameters["network_structure"] = []
        self.parameters["epochs"] = None
        self.parameters["batch_size"] = None
        self.parameters["loss_description"] = None

        self.parameters["datasets"] = {}
        self.__datasetplaceholder("training")
        self.__datasetplaceholder("testing")
        self.__datasetplaceholder("validation")

        self.modelconfig = {}
        self.modelconfig["model"] = None
        self.modelconfig["loadedfromcheckpoint"] = {}
        self.modelconfig["modelparameters"] = {}
        self.modelconfig["modelparameters"]["total"] = None
        self.modelconfig["modelparameters"]["layers"] = {}

        # Initialize all outputs (generated during model training)
        self.outputs = {}
        self.outputs["endpredictionsfile"] = None
        self.outputs["layerfilters"] = None
        self.outputs["training_measures"] = {}

        self.outputs["image_evolution"] = None      # Not used yet
        self.outputs["images_end"] = None           # Not used yet
        self.outputs["stored_session"] = None       # Not used yet
        self.outputs["checkpoint_path_model_weights"] = None
        self.outputs["times"] = {}
        self.outputs["times"]["trainingtime"] = None
        self.outputs["times"]["predictiontime"] = None

    def __datasetplaceholder(self, datasetname):
        self.parameters["datasets"][datasetname] = {}
        self.parameters["datasets"][datasetname]["tfrecords_path"] = None
        self.parameters["datasets"][datasetname]["number_of_images"] = 0
        self.parameters["datasets"][datasetname]["greyscale"] = False
        self.parameters["datasets"][datasetname]["data_dimensions"] = None
        self.parameters["datasets"][datasetname]["label_dimensions"] = None
        self.parameters["datasets"][datasetname]["shuffle"] = False
        self.parameters["datasets"][datasetname]["shufflebuffer"] = None

    def __filldatasetplaceholder(self, datasetname, tfrecordspath, numimg, datadim,
                                 labeldim, greyscale=False, shuffle=False, shufflebuffer=None):
        self.parameters["datasets"][datasetname]["tfrecords_path"] = tfrecordspath
        self.parameters["datasets"][datasetname]["number_of_images"] = numimg
        self.parameters["datasets"][datasetname]["greyscale"] = greyscale
        self.parameters["datasets"][datasetname]["data_dimensions"] = datadim
        self.parameters["datasets"][datasetname]["label_dimensions"] = labeldim
        self.parameters["datasets"][datasetname]["shuffle"] = shuffle
        self.parameters["datasets"][datasetname]["shufflebuffer"] = shufflebuffer

    def __trainingmeasureplaceholder(self, name):
        self.outputs["training_measures"][name] = {}
        #self.outputs["training_measures"][name]["evaluation_set"] = None
        #self.outputs["training_measures"][name]["measurements"] = []
        #self.outputs["training_measures"][name]["measurements_variance"] = []
        #self.outputs["training_measures"][name]["measured_epochs"] = []

    def ___filltrainingmeasureplaceholder(self, name, evalset, measurement=None, measured_epochs=None, measurement_variance=None):
        #if not ( evalset == "training" or evalset == "test" or evalset == "validation"):
        #    raise ValueError("The evaluation set for a training measure needs to be either 'training', 'test' or "
        #                     "'validation'.")
        if not evalset in self.outputs["training_measures"][name].keys():
            self.outputs["training_measures"][name][evalset] = {}
            self.outputs["training_measures"][name][evalset]["measurements"] = []
            self.outputs["training_measures"][name][evalset]["measurements_variance"] = []
            self.outputs["training_measures"][name][evalset]["measured_epochs"] = []

        if type(measurement) == list:
            self.outputs["training_measures"][name][evalset]["measurements"] = replaceNumpyfloat(measurement)
        if type(measurement_variance) == list:
            self.outputs["training_measures"][name][evalset]["measurements_variance"] = replaceNumpyfloat(measurement_variance)
        if type(measured_epochs) == list:
            self.outputs["training_measures"][name][evalset]["measured_epochs"] = replaceNumpyfloat(measured_epochs)

    def __modelparameterlayerplaceholder(self, layername, params=None):
        v = 0
        if not params == None:
            v = params
        self.modelconfig["modelparameters"]["layers"][layername] = v

    def __addtolayerparams(self, layername, params):
        self.modelconfig["modelparameters"]["layers"][layername] += params

    def __updatetotalmodelparams(self):
        sum = None
        for layer in list(self.modelconfig["modelparameters"]["layers"].keys()):
            if sum == None:
                sum = 0
            sum += self.modelconfig["modelparameters"]["layers"][layer]
        self.modelconfig["modelparameters"]["total"] = sum








    def loggertodict(self):
        return self.__dict__

    def addValidationMeasure(self, name, epochs, validation_measure):
        self.outputs["validation_measures_training"][name] = {}
        self.outputs["validation_measures_training"][name]["epochs"] = epochs
        self.outputs["validation_measures_training"][name]["values"] = validation_measure

    def addScriptCall(self, callstring):
        self.parameters["script_call"] = callstring

    def addNetworkStructure(self, path_to_json, structure_tuples):
        """
        Adds a given network structure to the Data logger under parameters->network_structure

        :param path_to_json: path to json file, specifying the networks layer structure
        :param structure_tuples: a list of tuples specifying IRCNN layers
        :return:
        """
        structure_dict = layersToDict(structure_tuples, unifykerneldims=True)

        network = {}
        network["path"] = path_to_json
        network["dict"] = structure_dict
        self.parameters["network_structure"].append(network)

    def addLoadedFromCheckpoint(self, loaded, checkpointpath=None):
        self.modelconfig["loadedfromcheckpoint"]["loaded"] = loaded
        if loaded:
            self.modelconfig["loadedfromcheckpoint"]["checkpointpath"] = checkpointpath
        else:
            self.modelconfig["loadedfromcheckpoint"]["checkpointpath"] = ""

    def addTime(self, time):
        self.parameters["time"] = time

    def addEpochsAndBatchSize(self, epochs, batchsize):
        self.parameters["epochs"] = epochs
        self.parameters["batch_size"] = batchsize

    def addLossDescription(self, lossdescription):
        self.parameters["loss_description"] = lossdescription

    def addFreeData(self, datasetname, tfrecordspath, numimg, datadim,
                        labeldim, greyscale=False, shuffle=False, shufflebuffer=None):
        if datasetname not in self.parameters["datasets"].keys():
            self.__datasetplaceholder(datasetname)
        self.__filldatasetplaceholder(datasetname, tfrecordspath, numimg, datadim,
                                      labeldim, greyscale=greyscale, shuffle=shuffle,
                                      shufflebuffer=shufflebuffer)

    def addTrainingData(self, tfrecordspath, numimg, datadim,
                        labeldim, greyscale=False, shuffle=False, shufflebuffer=None):
        self.__filldatasetplaceholder("training", tfrecordspath, numimg, datadim,
                                      labeldim, greyscale=greyscale, shuffle=shuffle,
                                      shufflebuffer=shufflebuffer)

    def addTestingData(self, tfrecordspath, numimg, datadim,
                        labeldim, greyscale=False, shuffle=False, shufflebuffer=None):
        self.__filldatasetplaceholder("testing", tfrecordspath, numimg, datadim,
                                      labeldim, greyscale=greyscale, shuffle=shuffle,
                                      shufflebuffer=shufflebuffer)

    def addValidationData(self, tfrecordspath, numimg, datadim,
                        labeldim, greyscale=False, shuffle=False, shufflebuffer=None):
        self.__filldatasetplaceholder("validation", tfrecordspath, numimg, datadim,
                                      labeldim, greyscale=greyscale, shuffle=shuffle,
                                      shufflebuffer=shufflebuffer)

    def addModelJson(self, modeljson, modelvars=None):
        self.modelconfig["model"] = modeljson
        for layer in modeljson["config"]["layers"]:
            self.__modelparameterlayerplaceholder(layer["name"])

        if not modelvars == None:
            for mv in modelvars:
                layername = mv.name.split("/")[-2]
                prod = 0
                for num in mv.shape:
                    if prod == 0:
                        prod = 1
                    prod *= int(num)
                self.__addtolayerparams(layername, prod)

        self.__updatetotalmodelparams()

    def addEndPredictions(self, predicitionsarray):
        self.outputs["endpredictionsfile"] = predicitionsarray

    def addFilterFiles(self, filterfiles):
        self.outputs["layerfilters"] = filterfiles

    def addMeasuresTraining(self, measurename, eval_set, measurement=None, measured_epochs=None, measurement_variance=None):
        """
        Note: eval_set, measurement and measured_epochs will NOT overwrite their existing values if set to None. However,
        their values in the logger will be replaced by [] if set to [].

        :param measurename: A unique string name for the measurement. Note that calling the method twice
        using the same string will overwrite the measures from the first call.
        :param eval_set: One of 'training', 'test' or 'validation'
        :param measurement: a list of measured values
        :param measured_epochs: a list of epoch-numbers, corresponding to the epochs when the measurements were taken.
        :return:
        """
        # if measurement not existing yet, make a placeholder for it. Otherwise fill it directly.
        if measurename not in self.outputs["training_measures"].keys():
            self.__trainingmeasureplaceholder(measurename)
        self.___filltrainingmeasureplaceholder(measurename, eval_set, measurement=measurement,
                                               measured_epochs=measured_epochs, measurement_variance=measurement_variance)

    def addCheckpointPath(self, checkpointpath):
        self.outputs["checkpoint_path_model_weights"] = checkpointpath

    def addTrainingTime(self, traintime):
        self.outputs["times"]["trainingtime"] = traintime

    def addPredictionTime(self, predicitiontime):
        self.outputs["times"]["predictiontime"] = predicitiontime

    def writeToJson(self):
        with open(self.parameters["json_path"], 'w') as outfile:
            jsondict = self.loggertodict()
            json.dump(jsondict, outfile, indent=2)




def replaceNumpyfloat(lis):
    if isinstance(lis, list):
        return [float(x) for x in lis]
    elif lis == None:
        return None
    else:
        raise ValueError("replaceNumpyFloat should only be applied to lists or Nonetype objects.")


def removefloat32_dict(d):
    for key, value in d.items():
        if isinstance(value, dict):
            removefloat32_dict(value)
        elif isinstance(value, list):
            removefloat32_list(value)
        else:
            print(str(type(value)) + ": %s" % str(value))

def removefloat32_list(l):
    for item in l:
        if isinstance(item, dict):
            removefloat32_dict(item)
        elif isinstance(item, list):
            removefloat32_list(item)
        else:
            print(str(type(item)) + ": %s" % str(item))


def pretty(d, indent=0):
    """
    Dictionary pretty printing, original from
    https://stackoverflow.com/questions/3229419/how-to-pretty-print-nested-dictionaries
    Modified to allow pretty printing of lists in dictionaries
    :param d: dict
    :param indent: at line beginning
    :return:
    """

    for key, value in d.items():
        print('\t' * indent + str(key))

        if isinstance(value, dict):
            pretty(value, indent+1)
        elif isinstance(value, list):
            try:
                if isinstance(value[0], dict):
                    for obj in value:
                        print('\t' * (indent + 1) + "[...]")
                        pretty(obj, indent + 2)
                else:
                    print('\t' * (indent + 1) + str(value))
            except IndexError:
                print('\t' * (indent + 1) + str(value))

        else:
            print('\t' * (indent+1) + str(value))


def correctJSONs(jsonFolder, unifykerneldims=True):
    for filename in listdir(jsonFolder):
        if filename.endswith(".json"):
            filepath = path.join(jsonFolder, filename)
            print("Working on %s" % filepath)
            lays = readJSONLayer(filepath)
            lays_new = []
            corrected = False
            for lay in lays:
                (a1, a2, a3, a4, alph) = lay
                if alph == "null":
                    alph = None
                    if not corrected:
                        print("    -> Corrected Dataset")
                    corrected = True
                lays_new.append((a1, a2, a3, a4, alph))

            writeJSONlayer(filepath, lays_new)




def readJSONLayer(jsonPath, unifykerneldims=True):
    """
    Reads an IRCNN layer specification from a json file.

    :param jsonPath: path to json with layer specification
    :param unifykerneldims: bool, if True the kernel dimensions (x,y) will be reduced to x alone.
    Useful for kernels where x and y are the same.
    :return: python list with tuples of the form (layertype, kernelsize, outputchannels, activation, alpha)
    """

    lays = []

    with open(jsonPath) as file:
        data = json.load(file)

        for layer in data["layers"]:
            type = layer["type"]
            ksiz = layer["kernelsize"]
            outc = layer["outputchannels"]
            acti = layer["activation"]
            alph = layer["alpha"]

            if unifykerneldims:
                ksiz = ksiz[0]

            lays.append((type, ksiz, outc, acti, alph))

    return lays


def loadJsonString(string):
    jsonobj = json.loads(string)
    return jsonobj


def writeJSONlayer(jsonPath, lays, unifykerneldims=True):
    """
    Writes a layer tuple into a json file. It is the "inverse operation" to readJSONlayer()

    :param jsonPath: Path to .json file to be written
    :param lays: list of layer tuples to be converted into .json file
    :param unifykerneldims: if True, it is assumed that the layer tuples only contain one kernel dimension
    due to kernel symmetrie.

    :return:
    """

    with open(jsonPath, 'w') as outfile:
        jsondict = layersToDict(lays, unifykerneldims=unifykerneldims)
        json.dump(jsondict, outfile, indent=2)



def layersToDict(lays, unifykerneldims=True):
    """
    Transforms the tuple specification for IRCNN layers into a dict.

    :param lays: the layer tuple
    :param unifykerneldims: If true, the method assumes that only a scalar kerneldimension is given (opposed to
    an array with x- and y-kerneldimensions
    :return: the dict for the specified layer tuples
    """

    dict = {}
    dict["layers"] = []

    for lay in lays:
        (typ, ksiz, outc, acti, alph) = lay

        if unifykerneldims:
            ksiz = [ksiz, ksiz]

        dict["layers"].append({
            "type": typ,
            "kernelsize": ksiz,
            "outputchannels": outc,
            "activation": acti,
            "alpha": alph
        })

    return dict




def createDateFolder(parent_path, custom_extension=""):
    """
    Creates a new folder with the current time as name in the parent-directory.
    If the parent directory doesn't exit, it is also created.

    :param parent_path: path to parent folder

    :return: a tuple containing the full path to the new folder (no tailing "/")
    and the raw date string that is used as name.
    """

    time = datetime.now()
    datestr = time.strftime("%Y-%m-%d_%H:%M:%S:%f")
    if not custom_extension == "":
        foldername = "%s_%s" % (datestr, custom_extension)
    else:
        foldername = datestr

    folderpath = path.join(parent_path, foldername)
    makedirs(folderpath, exist_ok=True)

    return (folderpath, foldername, datestr)


def createCheckpointFolder(outputfolder):

    folderpath = path.join(outputfolder, "checkpoints")
    makedirs(folderpath, exist_ok=True)

    return folderpath


def createLayerFilterFolder(outputfolder):

    folderpath = path.join(outputfolder, "layerfilters")
    makedirs(folderpath, exist_ok=True)

    return folderpath



def simpleCNNOutputCorrection(layers, labeloutputchannels):
    (typ, ksiz, outc, acti, alph) = layers[-1]
    if not outc == labeloutputchannels:
        outc = labeloutputchannels
    layers[-1] = (typ, ksiz, outc, acti, alph)
    return layers


def nameFromLayers(layers):

    name = []
    for layer in layers:
        (typ, ksiz, outc, acti, alph) = layer
        n = ""
        if typ == "sdpf":
            n += "B%d" % (ksiz)
        elif typ == "conv":
            n += "C%d" % (ksiz)
        else:
            n += "U%d" % (ksiz)

        if acti == "relu":
            n += "r"
        elif acti == "leakyrelu":
            n += "l"
        elif acti == "sigmoid":
            n += "s"
        elif acti == None:
            n += "_"
        else:
            n += "a"

        name.append(n)
    return '-'.join(name)

