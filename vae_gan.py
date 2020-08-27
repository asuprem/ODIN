import os, shutil, glob, re, pdb, json
import kaptan
import click
import utils
import torch, torchsummary, torchvision


@click.command()
@click.argument('config')
@click.option('--mode', default="train", help="Execution mode: [train/test]")
@click.option('--weights', default=".", help="Path to weights if mode is test")
def main(config, mode, weights):
    # Generate configuration
    cfg = kaptan.Kaptan(handler='yaml')
    config = cfg.import_config(config)

    # Generate logger
    MODEL_SAVE_NAME, MODEL_SAVE_FOLDER, LOGGER_SAVE_NAME, CHECKPOINT_DIRECTORY = utils.generate_save_names(config)
    logger = utils.generate_logger(MODEL_SAVE_FOLDER, LOGGER_SAVE_NAME)

    logger.info("*"*40);logger.info("");logger.info("")
    logger.info("Using the following configuration:")
    logger.info(config.export("yaml", indent=4))
    logger.info("");logger.info("");logger.info("*"*40)

    """ SETUP IMPORTS """
    #from crawlers import ReidDataCrawler
    #from generators import SequencedGenerator
    

    #from loss import LossBuilder
    

    NORMALIZATION_MEAN, NORMALIZATION_STD, RANDOM_ERASE_VALUE = utils.fix_generator_arguments(config)
    TRAINDATA_KWARGS = {"rea_value": config.get("TRANSFORMATION.RANDOM_ERASE_VALUE")}


    """ Load previousely saved logger, if it exists """
    DRIVE_BACKUP = config.get("SAVE.DRIVE_BACKUP")
    if DRIVE_BACKUP:
        backup_logger = os.path.join(CHECKPOINT_DIRECTORY, LOGGER_SAVE_NAME)
        if os.path.exists(backup_logger):
            shutil.copy2(backup_logger, ".")
    else:
        backup_logger = None

    NUM_GPUS = torch.cuda.device_count()
    if NUM_GPUS > 1:
        raise RuntimeError("Not built for multi-GPU. Please start with single-GPU.")
    logger.info("Found %i GPUs"%NUM_GPUS)


    # --------------------- BUILD GENERATORS ------------------------
    # Supported integrated data sources --> MNIST, CIFAR
    # For BDD or others need a crawler and stuff...but we;ll deal with it later
    from generators import ClassedGenerator

    load_dataset = config.get("EXECUTION.DATASET_PRELOAD")
    if load_dataset in ["MNIST", "CIFAR10", "CIFAR100"]:
        crawler = load_dataset
        #dataset = torchvision.datasets.MNIST(root="./MNIST", train=True,)
        logger.info("No crawler necessary when using %s dataset"%crawler)
    else:
        raise NotImplementedError()

    
    train_generator = ClassedGenerator( gpus=NUM_GPUS, i_shape=config.get("DATASET.SHAPE"), \
                                        normalization_mean=NORMALIZATION_MEAN, normalization_std=NORMALIZATION_STD, normalization_scale=1./config.get("TRANSFORMATION.NORMALIZATION_SCALE"), \
                                        h_flip = config.get("TRANSFORMATION.H_FLIP"), t_crop=config.get("TRANSFORMATION.T_CROP"), rea=config.get("TRANSFORMATION.RANDOM_ERASE"), 
                                        **TRAINDATA_KWARGS)    
    train_generator.setup(  crawler, preload_classes = config.get("EXECUTION.DATASET_PRELOAD_CLASS"), \
                            mode='train',batch_size=config.get("TRANSFORMATION.BATCH_SIZE"), \
                            workers = config.get("TRANSFORMATION.WORKERS"))
    logger.info("Generated training data generator")

    test_generator = ClassedGenerator( gpus=NUM_GPUS, i_shape=config.get("DATASET.SHAPE"), \
                                        normalization_mean=NORMALIZATION_MEAN, normalization_std=NORMALIZATION_STD, normalization_scale=1./config.get("TRANSFORMATION.NORMALIZATION_SCALE"), \
                                        h_flip = config.get("TRANSFORMATION.H_FLIP"), t_crop=config.get("TRANSFORMATION.T_CROP"), rea=config.get("TRANSFORMATION.RANDOM_ERASE"), 
                                        **TRAINDATA_KWARGS)    
    test_generator.setup(  crawler, preload_classes = config.get("EXECUTION.DATASET_TEST_PRELOAD_CLASS"), \
                            mode='test',batch_size=config.get("TRANSFORMATION.BATCH_SIZE"), \
                            workers = config.get("TRANSFORMATION.WORKERS"))
    logger.info("Generated testing data generator")


    # --------------------- INSTANTIATE MODEL ------------------------
    model_builder = __import__("models", fromlist=["*"])
    model_builder = getattr(model_builder, config.get("EXECUTION.MODEL_BUILDER"))
    logger.info("Loaded {} from {} to build VAEGAN model".format(config.get("EXECUTION.MODEL_BUILDER"), "models"))

    vaegan_model = model_builder(   arch=config.get("MODEL.ARCH"), base=config.get("MODEL.BASE"), \
                                    latent_dimensions = config.get("MODEL.LATENT_DIMENSIONS"), \
                                    **json.loads(config.get("MODEL.MODEL_KWARGS")))
    logger.info("Finished instantiating model")

    if mode == "test":
        vaegan_model.load_state_dict(torch.load(weights))
        vaegan_model.cuda()
        vaegan_model.eval()
    else:
        vaegan_model.cuda()
        #logger.info(torchsummary.summary(vaegan_model, input_size=(config.get("TRANSFORMATION.CHANNELS"), *config.get("DATASET.SHAPE"))))
        logger.info(torchsummary.summary(vaegan_model.Encoder, input_size=(config.get("TRANSFORMATION.CHANNELS"), *config.get("DATASET.SHAPE"))))
        logger.info(torchsummary.summary(vaegan_model.Decoder, input_size=(config.get("MODEL.LATENT_DIMENSIONS"), 1)))
        logger.info(torchsummary.summary(vaegan_model.LatentDiscriminator, input_size=(config.get("MODEL.LATENT_DIMENSIONS"), 1)))
        logger.info(torchsummary.summary(vaegan_model.Discriminator, input_size=(config.get("TRANSFORMATION.CHANNELS"), *config.get("DATASET.SHAPE"))))
    

    # --------------------- INSTANTIATE LOSS ------------------------
    # ----------- NOT NEEDED. VAEGAN WILL USE BCE LOSS THROUGHOUT 
    # loss_function = LossBuilder(loss_functions=config.get("LOSS.LOSSES"), loss_lambda=config.get("LOSS.LOSS_LAMBDAS"), loss_kwargs=config.get("LOSS.LOSS_KWARGS"), **{"logger":logger})
    # logger.info("Built loss function")
    # --------------------- INSTANTIATE OPTIMIZER ------------------------
    optimizer_builder = __import__("optimizer", fromlist=["*"])
    optimizer_builder = getattr(optimizer_builder, config.get("EXECUTION.OPTIMIZER_BUILDER"))
    logger.info("Loaded {} from {} to build VAEGAN model".format(config.get("EXECUTION.OPTIMIZER_BUILDER"), "optimizer"))

    OPT = optimizer_builder(base_lr=config.get("OPTIMIZER.BASE_LR"))
    optimizer = OPT.build(vaegan_model, config.get("OPTIMIZER.OPTIMIZER_NAME"), **json.loads(config.get("OPTIMIZER.OPTIMIZER_KWARGS")))
    logger.info("Built optimizer")
    # --------------------- INSTANTIATE SCHEDULER ------------------------
    try:
        scheduler = __import__('torch.optim.lr_scheduler', fromlist=['lr_scheduler'])
        scheduler_ = getattr(scheduler, config.get("SCHEDULER.LR_SCHEDULER"))
    except (ModuleNotFoundError, AttributeError):
        scheduler_ = config.get("SCHEDULER.LR_SCHEDULER")
        scheduler = __import__("scheduler."+scheduler_, fromlist=[scheduler_])
        scheduler_ = getattr(scheduler, scheduler_)
    scheduler = {}
    for base_model in ["Encoder", "Decoder", "Discriminator", "Autoencoder", "LatentDiscriminator"]:
        scheduler[base_model] = scheduler_(optimizer[base_model], last_epoch = -1, **json.loads(config.get("SCHEDULER.LR_KWARGS")))
        logger.info("Built scheduler for {}".format(base_model))
    

    # --------------------- SETUP CONTINUATION  ------------------------
    if DRIVE_BACKUP:
        fl_list = glob.glob(os.path.join(CHECKPOINT_DIRECTORY, "*.pth"))
    else:
        fl_list = glob.glob(os.path.join(MODEL_SAVE_FOLDER, "*.pth"))
    _re = re.compile(r'.*epoch([0-9]+)\.pth')
    previous_stop = [int(item[1]) for item in [_re.search(item) for item in fl_list] if item is not None]
    if len(previous_stop) == 0:
        previous_stop = 0
    else:
        previous_stop = max(previous_stop) + 1
        logger.info("Previous stop detected. Will attempt to resume from epoch %i"%previous_stop)

    # --------------------- INSTANTIATE TRAINER  ------------------------
    Trainer = __import__("trainer", fromlist=["*"])
    Trainer = getattr(Trainer, config.get("EXECUTION.TRAINER"))
    logger.info("Loaded {} from {} to build VAEGAN model".format(config.get("EXECUTION.TRAINER"), "trainer"))

    loss_stepper = Trainer(model=vaegan_model, loss_fn = None, optimizer = optimizer, scheduler = scheduler, train_loader = train_generator.dataloader, test_loader = test_generator.dataloader, epochs = config.get("EXECUTION.EPOCHS"), batch_size = config.get("TRANSFORMATION.BATCH_SIZE"), latent_size = config.get("MODEL.LATENT_DIMENSIONS"), logger = logger)
    loss_stepper.setup(step_verbose = config.get("LOGGING.STEP_VERBOSE"), save_frequency=config.get("SAVE.SAVE_FREQUENCY"), test_frequency = config.get("EXECUTION.TEST_FREQUENCY"), save_directory = MODEL_SAVE_FOLDER, save_backup = DRIVE_BACKUP, backup_directory = CHECKPOINT_DIRECTORY, gpus=NUM_GPUS, fp16 = config.get("OPTIMIZER.FP16"), model_save_name = MODEL_SAVE_NAME, logger_file = LOGGER_SAVE_NAME)
    if mode == 'train':
      loss_stepper.train(continue_epoch=previous_stop)
    elif mode == 'test':
      loss_stepper.evaluate()
    else:
      raise NotImplementedError()






if __name__ == "__main__":
    main()