import pdb
import tqdm
from collections import defaultdict
#from sklearn.metrics import average_precision_score
import shutil
import os
import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

#from scipy.spatial.distance import cdist



class VAEGANTrainer:
    def __init__(self, model, loss_fn, optimizer, scheduler, train_loader, test_loader, epochs, batch_size, latent_size, logger):
        # NEED TO HANDLE ANOMALIES
        self.model = model
        
        if loss_fn is None:
            self.loss_fn = nn.BCELoss()
        else:
            self.loss_fn = loss_fn
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.epochs = epochs
        self.logger = logger

        self.global_batch = 0
        self.global_epoch = 0


        self.batch_size = batch_size
        self.latent_size = latent_size
        self.e_loss, self.de_loss, self.d_loss, self.z_loss, self.ae_loss = [],[],[],[],[]

    def setup(self, step_verbose = 5, save_frequency = 5, test_frequency = 5, \
                save_directory = './checkpoint/', save_backup = False, backup_directory = None, gpus=1,\
                fp16 = False, model_save_name = None, logger_file = None):
        self.step_verbose = step_verbose
        self.save_frequency = save_frequency
        self.test_frequency = test_frequency
        self.save_directory = save_directory
        self.backup_directory = None
        self.model_save_name = model_save_name
        self.logger_file = logger_file
        self.save_backup = save_backup
        if self.save_backup:
            self.backup_directory = backup_directory
            os.makedirs(self.backup_directory, exist_ok=True)
        os.makedirs(self.save_directory, exist_ok=True)

        self.gpus = gpus

        if self.gpus != 1:
            raise NotImplementedError()
        
        self.model.cuda()
        
        self.fp16 = False   # fp16
        if self.fp16 and self.apex is not None:
            self.model, self.optimizer = self.apex.amp.initialize(self.model, self.optimizer, opt_level='O1')
        
    # TODO
    def step(self,batch):
        if batch[0].shape[0] < self.batch_size:
            return
        e_loss, de_loss, d_loss, z_loss, ae_loss = 0,0,0,0,0
        ones = torch.ones(self.batch_size).cuda()
        zeros = torch.zeros(self.batch_size).cuda()
        self.model.train()
        
        self.optimizer["Encoder"].zero_grad()
        self.optimizer["Decoder"].zero_grad()
        self.optimizer["Autoencoder"].zero_grad()
        self.optimizer["Discriminator"].zero_grad()
        self.optimizer["LatentDiscriminator"].zero_grad()
        
        
        img, labels = batch
        img = img.cuda()
        # logits, features, labels

        self.model.Encoder.train()
        self.model.Decoder.train()
        self.model.Discriminator.train()
        self.model.LatentDiscriminator.train()

        self.model.Discriminator.zero_grad()
        real_ = self.model.Discriminator(img).squeeze()
        r_loss = self.loss_fn(real_, ones)
        varz = torch.autograd.Variable(torch.randn((self.batch_size, self.latent_size)).unsqueeze(-1)).cuda()
        generated = self.model.Decoder(varz)
        fake_ = self.model.Discriminator(generated).squeeze()
        f_loss = self.loss_fn(fake_, zeros)
        rf_loss = r_loss + f_loss
        rf_loss.backward()
        self.optimizer["Discriminator"].step()
        d_loss = rf_loss.item()


        self.model.Decoder.zero_grad()
        varzd = torch.autograd.Variable(torch.randn((self.batch_size, self.latent_size)).unsqueeze(-1)).cuda()
        fake_generated = self.model.Decoder(varzd)
        discri_fake = self.model.Discriminator(fake_generated).squeeze()
        df_loss = self.loss_fn(discri_fake, ones)
        df_loss.backward()
        self.optimizer["Decoder"].step()
        de_loss = df_loss.item()


        self.model.LatentDiscriminator.zero_grad()
        varzl = torch.autograd.Variable(torch.randn((self.batch_size, self.latent_size)).unsqueeze(-1)).cuda()
        z_real = self.model.LatentDiscriminator(varzl).squeeze()
        z_real_loss = self.loss_fn(z_real, ones)
        z_fake = self.model.LatentDiscriminator(self.model.Encoder(img).squeeze()).squeeze()
        z_fake_loss = self.loss_fn(z_fake, zeros)
        zf_loss = z_real_loss + z_fake_loss
        zf_loss.backward()
        self.optimizer["LatentDiscriminator"].step()
        z_loss = zf_loss.item()


        self.model.Encoder.zero_grad()
        self.model.Decoder.zero_grad()
        latent = self.model.Encoder(img)
        regen = self.model.Decoder(latent.unsqueeze(-1))
        latent_d = self.model.LatentDiscriminator(latent.squeeze()).squeeze()
        ef_loss = 2.*self.loss_fn(latent_d, ones)
        #pdb.set_trace()
        reconstruction = F.binary_cross_entropy(regen, img)
        ae_loss = ef_loss + reconstruction
        ae_loss.backward()
        self.optimizer["Autoencoder"].step()
        e_loss = ef_loss.item()
        ae_loss = reconstruction.item()

        self.e_loss.append(e_loss)
        self.de_loss.append(de_loss)
        self.d_loss.append(d_loss)
        self.z_loss.append(z_loss)
        self.ae_loss.append(ae_loss)

    def train(self,continue_epoch = 0):    
        self.logger.info("Starting training")
        self.logger.info("Logging to:\t%s"%self.logger_file)
        self.logger.info("Models will be saved to local directory:\t%s"%self.save_directory)
        if self.save_backup:
            self.logger.info("Models will be backed up to drive directory:\t%s"%self.backup_directory)
        self.logger.info("Models will be saved with base name:\t%s_epoch[].pth"%self.model_save_name)
        self.logger.info("Optimizers will be saved with base name:\t%s_epoch[]_optimizer.pth"%self.model_save_name)
        self.logger.info("Schedulers will be saved with base name:\t%s_epoch[]_scheduler.pth"%self.model_save_name)
        

        if continue_epoch > 0:
            load_epoch = continue_epoch - 1
            self.load(load_epoch)

        for epoch in range(self.epochs):
            if epoch >= continue_epoch:
                for batch in self.train_loader:
                    if not self.global_batch:
                        lrs = self.scheduler["Encoder"].get_lr(); lrs = sum(lrs)/float(len(lrs))
                        self.logger.info("Starting epoch {0} with {1} steps and learning rate {2:2.5E}".format(epoch, len(self.train_loader) - (len(self.train_loader)%10), lrs))
                    # TODO 
                    self.step(batch)
                    
                    self.global_batch += 1
                    if (self.global_batch + 1) % self.step_verbose == 0:
                        self.logger.info('Epoch{0}.{1}\tEncoder: {2:.3f} Decoder: {3:.3f} AE: {4:.3f} Discriminator: {5:.3f} Latent: {6:.3f}'.format(self.global_epoch, self.global_batch, np.mean(self.e_loss), np.mean(self.de_loss), np.mean(self.ae_loss), np.mean(self.d_loss), np.mean(self.z_loss)))
                        self.e_loss, self.de_loss, self.d_loss, self.z_loss, self.ae_loss = [],[],[],[],[]
                self.global_batch = 0
                # TODO 
                self.scheduler["Encoder"].step()
                self.scheduler["Decoder"].step()
                self.scheduler["Autoencoder"].step()
                self.scheduler["Discriminator"].step()
                self.scheduler["LatentDiscriminator"].step()
                
                self.logger.info('{0} Completed epoch {1} {2}'.format('*'*10, self.global_epoch, '*'*10))
                if self.global_epoch % self.test_frequency == 0:
                    #self.evaluate()     # TODO 
                    pass
                if self.global_epoch % self.save_frequency == 0:
                    self.save()
                self.global_epoch += 1
            else:
                self.global_epoch = epoch+1

    def evaluate(self,):
        

        pass

    def save(self):
        self.logger.info("Saving model, optimizer, and scheduler.")
        MODEL_SAVE = self.model_save_name + '_epoch%i'%self.global_epoch + '.pth'
        
        OPTIM_ENCODER_SAVE = self.model_save_name + '_epoch%i'%self.global_epoch + '_optimizer_encoder.pth'
        OPTIM_DECODER_SAVE = self.model_save_name + '_epoch%i'%self.global_epoch + '_optimizer_decoder.pth'
        OPTIM_DISCRIMINATOR_SAVE = self.model_save_name + '_epoch%i'%self.global_epoch + '_optimizer_discriminator.pth'
        OPTIM_AUTOENCODER_SAVE = self.model_save_name + '_epoch%i'%self.global_epoch + '_optimizer_autoencoder.pth'
        OPTIM_LATENT_SAVE = self.model_save_name + '_epoch%i'%self.global_epoch + '_optimizer_latent.pth'

        SCHEDULER_ENCODER_SAVE = self.model_save_name + '_epoch%i'%self.global_epoch + '_scheduler_encoder.pth'
        SCHEDULER_DECODER_SAVE = self.model_save_name + '_epoch%i'%self.global_epoch + '_scheduler_decoder.pth'
        SCHEDULER_DISCRIMINATOR_SAVE = self.model_save_name + '_epoch%i'%self.global_epoch + '_scheduler_discriminator.pth'
        SCHEDULER_AUTOENCODER_SAVE = self.model_save_name + '_epoch%i'%self.global_epoch + '_scheduler_autoencoder.pth'
        SCHEDULER_LATENT_SAVE = self.model_save_name + '_epoch%i'%self.global_epoch + '_scheduler_latent.pth'

        torch.save(self.model.state_dict(), os.path.join(self.save_directory, MODEL_SAVE))
        torch.save(self.optimizer["Encoder"].state_dict(), os.path.join(self.save_directory, OPTIM_ENCODER_SAVE))
        torch.save(self.optimizer["Decoder"].state_dict(), os.path.join(self.save_directory, OPTIM_DECODER_SAVE))
        torch.save(self.optimizer["Discriminator"].state_dict(), os.path.join(self.save_directory, OPTIM_DISCRIMINATOR_SAVE))
        torch.save(self.optimizer["Autoencoder"].state_dict(), os.path.join(self.save_directory, OPTIM_AUTOENCODER_SAVE))
        torch.save(self.optimizer["LatentDiscriminator"].state_dict(), os.path.join(self.save_directory, OPTIM_LATENT_SAVE))

        torch.save(self.scheduler["Encoder"].state_dict(), os.path.join(self.save_directory, SCHEDULER_ENCODER_SAVE))
        torch.save(self.scheduler["Decoder"].state_dict(), os.path.join(self.save_directory, SCHEDULER_DECODER_SAVE))
        torch.save(self.scheduler["Discriminator"].state_dict(), os.path.join(self.save_directory, SCHEDULER_DISCRIMINATOR_SAVE))
        torch.save(self.scheduler["Autoencoder"].state_dict(), os.path.join(self.save_directory, SCHEDULER_AUTOENCODER_SAVE))
        torch.save(self.scheduler["LatentDiscriminator"].state_dict(), os.path.join(self.save_directory, SCHEDULER_LATENT_SAVE))

        if self.save_backup:
            shutil.copy2(os.path.join(self.save_directory, MODEL_SAVE), self.backup_directory)

            shutil.copy2(os.path.join(self.save_directory, OPTIM_ENCODER_SAVE), self.backup_directory)
            shutil.copy2(os.path.join(self.save_directory, OPTIM_DECODER_SAVE), self.backup_directory)
            shutil.copy2(os.path.join(self.save_directory, OPTIM_AUTOENCODER_SAVE), self.backup_directory)
            shutil.copy2(os.path.join(self.save_directory, OPTIM_DISCRIMINATOR_SAVE), self.backup_directory)
            shutil.copy2(os.path.join(self.save_directory, OPTIM_LATENT_SAVE), self.backup_directory)

            shutil.copy2(os.path.join(self.save_directory, SCHEDULER_ENCODER_SAVE), self.backup_directory)
            shutil.copy2(os.path.join(self.save_directory, SCHEDULER_DECODER_SAVE), self.backup_directory)
            shutil.copy2(os.path.join(self.save_directory, SCHEDULER_AUTOENCODER_SAVE), self.backup_directory)
            shutil.copy2(os.path.join(self.save_directory, SCHEDULER_DISCRIMINATOR_SAVE), self.backup_directory)
            shutil.copy2(os.path.join(self.save_directory, SCHEDULER_LATENT_SAVE), self.backup_directory)

            self.logger.info("Performing drive backup of model, optimizer, and scheduler.")
            
            LOGGER_SAVE = os.path.join(self.backup_directory, self.logger_file)
            if os.path.exists(LOGGER_SAVE):
                os.remove(LOGGER_SAVE)
            shutil.copy2(self.logger_file, LOGGER_SAVE)
    
    def load(self, load_epoch):
        self.logger.info("Resuming training from epoch %i. Loading saved state from %i"%(load_epoch+1,load_epoch))
        model_load = self.model_save_name + '_epoch%i'%load_epoch + '.pth'
        optim_load = self.model_save_name + '_epoch%i'%load_epoch + '_optimizer.pth'
        scheduler_load = self.model_save_name + '_epoch%i'%load_epoch + '_scheduler.pth'

        OPTIM_ENCODER_SAVE = self.model_save_name + '_epoch%i'%self.global_epoch + '_optimizer_encoder.pth'
        OPTIM_DECODER_SAVE = self.model_save_name + '_epoch%i'%self.global_epoch + '_optimizer_decoder.pth'
        OPTIM_DISCRIMINATOR_SAVE = self.model_save_name + '_epoch%i'%self.global_epoch + '_optimizer_discriminator.pth'
        OPTIM_AUTOENCODER_SAVE = self.model_save_name + '_epoch%i'%self.global_epoch + '_optimizer_autoencoder.pth'
        OPTIM_LATENT_SAVE = self.model_save_name + '_epoch%i'%self.global_epoch + '_optimizer_latent.pth'

        SCHEDULER_ENCODER_SAVE = self.model_save_name + '_epoch%i'%self.global_epoch + '_scheduler_encoder.pth'
        SCHEDULER_DECODER_SAVE = self.model_save_name + '_epoch%i'%self.global_epoch + '_scheduler_decoder.pth'
        SCHEDULER_DISCRIMINATOR_SAVE = self.model_save_name + '_epoch%i'%self.global_epoch + '_scheduler_discriminator.pth'
        SCHEDULER_AUTOENCODER_SAVE = self.model_save_name + '_epoch%i'%self.global_epoch + '_scheduler_autoencoder.pth'
        SCHEDULER_LATENT_SAVE = self.model_save_name + '_epoch%i'%self.global_epoch + '_scheduler_latent.pth'

        if self.save_backup:
            self.logger.info("Loading model, optimizer, and scheduler from drive backup.")
            model_load_path = os.path.join(self.backup_directory, model_load)
            
            optim_load_path_e = os.path.join(self.backup_directory, OPTIM_ENCODER_SAVE)
            optim_load_path_de = os.path.join(self.backup_directory, OPTIM_DECODER_SAVE)
            optim_load_path_ae = os.path.join(self.backup_directory, OPTIM_AUTOENCODER_SAVE)
            optim_load_path_d = os.path.join(self.backup_directory, OPTIM_DISCRIMINATOR_SAVE)
            optim_load_path_z = os.path.join(self.backup_directory, OPTIM_LATENT_SAVE)

            scheduler_load_path_e = os.path.join(self.backup_directory, SCHEDULER_ENCODER_SAVE)
            scheduler_load_path_de = os.path.join(self.backup_directory, SCHEDULER_DECODER_SAVE)
            scheduler_load_path_ae = os.path.join(self.backup_directory, SCHEDULER_AUTOENCODER_SAVE)
            scheduler_load_path_d = os.path.join(self.backup_directory, SCHEDULER_DISCRIMINATOR_SAVE)
            scheduler_load_path_z = os.path.join(self.backup_directory, SCHEDULER_LATENT_SAVE)
        else:
            self.logger.info("Loading model, optimizer, and scheduler from local backup.")
            model_load_path = os.path.join(self.save_directory, model_load)
            
            optim_load_path_e = os.path.join(self.save_directory, OPTIM_ENCODER_SAVE)
            optim_load_path_de = os.path.join(self.save_directory, OPTIM_DECODER_SAVE)
            optim_load_path_ae = os.path.join(self.save_directory, OPTIM_AUTOENCODER_SAVE)
            optim_load_path_d = os.path.join(self.save_directory, OPTIM_DISCRIMINATOR_SAVE)
            optim_load_path_z = os.path.join(self.save_directory, OPTIM_LATENT_SAVE)

            scheduler_load_path_e = os.path.join(self.save_directory, SCHEDULER_ENCODER_SAVE)
            scheduler_load_path_de = os.path.join(self.save_directory, SCHEDULER_DECODER_SAVE)
            scheduler_load_path_ae = os.path.join(self.save_directory, SCHEDULER_AUTOENCODER_SAVE)
            scheduler_load_path_d = os.path.join(self.save_directory, SCHEDULER_DISCRIMINATOR_SAVE)
            scheduler_load_path_z = os.path.join(self.save_directory, SCHEDULER_LATENT_SAVE)

        self.model.load_state_dict(torch.load(model_load_path))
        self.logger.info("Finished loading model state_dict from %s"%model_load_path)
        self.optimizer["Encoder"].load_state_dict(torch.load(optim_load_path_e))
        self.optimizer["Decoder"].load_state_dict(torch.load(optim_load_path_de))
        self.optimizer["Autoencoder"].load_state_dict(torch.load(optim_load_path_ae))
        self.optimizer["Discriminator"].load_state_dict(torch.load(optim_load_path_d))
        self.optimizer["LatentDiscriminator"].load_state_dict(torch.load(optim_load_path_z))
        self.logger.info("Finished loading optimizer state_dict from %s"%OPTIM_ENCODER_SAVE)
        
        self.scheduler["Encoder"].load_state_dict(torch.load(scheduler_load_path_e))
        self.scheduler["Decoder"].load_state_dict(torch.load(scheduler_load_path_de))
        self.scheduler["Autoencoder"].load_state_dict(torch.load(scheduler_load_path_ae))
        self.scheduler["Discriminator"].load_state_dict(torch.load(scheduler_load_path_d))
        self.scheduler["LatentDiscriminator"].load_state_dict(torch.load(scheduler_load_path_z))
        self.logger.info("Finished loading scheduler state_dict from %s"%SCHEDULER_ENCODER_SAVE)