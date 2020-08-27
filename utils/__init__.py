from . import web as web
import logging

def generate_logger(MODEL_SAVE_FOLDER, LOGGER_SAVE_NAME):
    logger = logging.getLogger(MODEL_SAVE_FOLDER)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(LOGGER_SAVE_NAME)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s-%(msecs)d %(message)s',datefmt="%H:%M:%S")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    cs = logging.StreamHandler()
    cs.setLevel(logging.DEBUG)
    cs.setFormatter(logging.Formatter('%(asctime)s-%(msecs)d %(message)s',datefmt="%H:%M:%S"))
    logger.addHandler(cs)
    return logger

def generate_save_names(cfg):
    MODEL_SAVE_NAME = "%s-v%i"%(cfg.get("SAVE.MODEL_CORE_NAME"), cfg.get("SAVE.MODEL_VERSION"))
    MODEL_SAVE_FOLDER = "%s-v%i-%s-%s"%(cfg.get("SAVE.MODEL_CORE_NAME"), cfg.get("SAVE.MODEL_VERSION"), cfg.get("SAVE.MODEL_BACKBONE"), cfg.get("SAVE.MODEL_QUALIFIER"))
    LOGGER_SAVE_NAME = "%s-v%i-%s-%s-logger.log"%(cfg.get("SAVE.MODEL_CORE_NAME"), cfg.get("SAVE.MODEL_VERSION"), cfg.get("SAVE.MODEL_BACKBONE"), cfg.get("SAVE.MODEL_QUALIFIER"))
    if cfg.get("SAVE.DRIVE_BACKUP"):
        CHECKPOINT_DIRECTORY = "./drive/My Drive/Vehicles/Models/" + MODEL_SAVE_FOLDER
    else:
        CHECKPOINT_DIRECTORY = ''
    return MODEL_SAVE_NAME, MODEL_SAVE_FOLDER, LOGGER_SAVE_NAME, CHECKPOINT_DIRECTORY

def fix_generator_arguments(cfg):
    if type(cfg.get("TRANSFORMATION.NORMALIZATION_MEAN")) is int or type(cfg.get("TRANSFORMATION.NORMALIZATION_MEAN")) is float:
        NORMALIZATION_MEAN = [cfg.get("TRANSFORMATION.NORMALIZATION_MEAN")]*cfg.get("TRANSFORMATION.CHANNELS")
    if type(cfg.get("TRANSFORMATION.NORMALIZATION_STD")) is int or type(cfg.get("TRANSFORMATION.NORMALIZATION_STD")) is float:
        NORMALIZATION_STD = [cfg.get("TRANSFORMATION.NORMALIZATION_STD")]*cfg.get("TRANSFORMATION.CHANNELS")
    if type(cfg.get("TRANSFORMATION.RANDOM_ERASE_VALUE")) is int or type(cfg.get("TRANSFORMATION.RANDOM_ERASE_VALUE")) is float:
        RANDOM_ERASE_VALUE = [cfg.get("TRANSFORMATION.RANDOM_ERASE_VALUE")]*cfg.get("TRANSFORMATION.CHANNELS")
    return NORMALIZATION_MEAN, NORMALIZATION_STD, RANDOM_ERASE_VALUE

model_weights = {
    "resnet18":["https://download.pytorch.org/models/resnet18-5c106cde.pth", "resnet18-5c106cde.pth"],
    "resnet34":["https://download.pytorch.org/models/resnet34-333f7ec4.pth", "resnet34-333f7ec4.pth"],
    "resnet50":["https://download.pytorch.org/models/resnet50-19c8e357.pth", "resnet50-19c8e357.pth"],
    "resnet101":["https://download.pytorch.org/models/resnet101-5d3b4d8f.pth", "resnet101-5d3b4d8f.pth"],
    "resnet152":["https://download.pytorch.org/models/resnet152-b121ed2d.pth", "resnet152-b121ed2d.pth"],
    "resnext50_32x4d":["https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth", "resnext50_32x4d-7cdf4587.pth"],
    "resnext101_32x8d":["https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth", "resnext101_32x8d-8ba56ff5.pth"],
    "wide_resnet50_2":["https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth", "wide_resnet50_2-95faca4d.pth"],
    "wide_resnet101_2":["https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth", "wide_resnet50_2-95faca4d.pth"],
    "resnet18_cbam":["https://download.pytorch.org/models/resnet18-5c106cde.pth", "resnet18-5c106cde_cbam.pth"],
    "resnet34_cbam":["https://download.pytorch.org/models/resnet34-333f7ec4.pth", "resnet34-333f7ec4_cbam.pth"],
    "resnet50_cbam":["https://download.pytorch.org/models/resnet50-19c8e357.pth", "resnet50-19c8e357_cbam.pth"],
    "resnet101_cbam":["https://download.pytorch.org/models/resnet101-5d3b4d8f.pth", "resnet101-5d3b4d8f_cbam.pth"],
    "resnet152_cbam":["https://download.pytorch.org/models/resnet152-b121ed2d.pth", "resnet152-b121ed2d_cbam.pth"]
    }