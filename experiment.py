import torch
from torch import optim
from models import BaseCNN
from archs.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from datasets.ASVSpoof import ASVSpoof
from datasets.ASVSpoofH import ASVSpoofH

class ImglistToTensor(torch.nn.Module):
        """
        Converts a list of PIL images in the range [0,255] to a torch.FloatTensor
        of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1].
        Can be used as first transform for ``VideoFrameDataset``.
        """
        def forward(self, img_list):
            """
            Converts each PIL image in a list to
            a torch Tensor and stacks them into
            a single tensor.
            Args:
                img_list: list of PIL images.
            Returns:
                tensor of size ``NUM_IMAGES x CHANNELS x HEIGHT x WIDTH``
            """
            return torch.stack([transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(transforms.functional.to_tensor(pic)) for pic in img_list])

class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseCNN,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass
    
    def predict(self, input: Tensor) -> Any:
        return self.model.predict(input)

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = labels.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['batch_size']/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        return train_loss

    
    def training_epoch_end(self, training_step_outputs):
        for pred in training_step_outputs:
            pass


    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = labels.device

        results = self.forward(real_img, labels = labels)
        val_acc = self.model.accuracy(*results,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)
        val_loss = self.model.loss_function(*results,
                                            M_N = self.params['batch_size']/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        return {**val_loss , **val_acc}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['accuracy'] for x in outputs]).mean()
        tensorboard_logs = {'avg_loss': avg_loss, 'avg_acc': avg_acc}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}


    def get_latent_space(self, inputs):

        return self.model.get_latent_space(inputs)


    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels = test_label)
        vutils.save_image(recons.data,
                          f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                          f"recons_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True,
                          nrow=12)


        del test_input, recons #, samples


    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(True),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    @data_loader
    def train_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'asvspoof':
            dataset = ASVSpoof(root = self.params['data_path'],
                             split = "Train",
                             ext = self.params['ext'],
                             sliding_window = self.params['sliding_window'],
                             frames = self.params['frames'],
                             partition=self.params['partition'],
                             img_size=self.params['img_size'],
                             transform = transform)

        elif self.params['dataset'] == 'asvspoofh':
            dataset = ASVSpoofH(root = self.params['data_path'],
                             split = "Train",
                             ext = self.params['ext'],
                             sliding_window = self.params['sliding_window'],
                             frames = self.params['frames'],
                             img_size=self.params['img_size'],
                             transform = transform)

        else:
            raise ValueError('Undefined dataset type')

        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size = self.params['batch_size'],
                          shuffle = True,
                          drop_last=True)

    @data_loader
    def val_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'asvspoof':
            self.sample_dataloader =  DataLoader(ASVSpoof(root = self.params['data_path'],
                                                        split = "Dev",
                                                        ext = self.params['ext'],
                                                        sliding_window = self.params['sliding_window'],
                                                        frames = self.params['frames'],
                                                        partition=self.params['partition'],
                                                        img_size=self.params['img_size'],
                                                        transform = transform),
                                                 batch_size= self.params['batch_size'],
                                                 shuffle = True)
            self.num_val_imgs = len(self.sample_dataloader)

        elif self.params['dataset'] == 'asvspoofh':
            self.sample_dataloader =  DataLoader(ASVSpoofH(root = self.params['data_path'],
                                                        split = "Dev",
                                                        ext = self.params['ext'],
                                                        sliding_window = self.params['sliding_window'],
                                                        frames = self.params['frames'],
                                                        img_size=self.params['img_size'],
                                                        transform = transform),
                                                 batch_size= self.params['batch_size'],
                                                 shuffle = True)
            self.num_val_imgs = len(self.sample_dataloader)

        else:
            raise ValueError('Undefined dataset type')

        return self.sample_dataloader


    def data_transforms(self):

        if self.params['dataset'] == 'asvspoof' or self.params['dataset'] == 'asvspoofh':
            transform = transforms.Compose([ImglistToTensor()])
        
        elif self.params['dataset'] == 'casia' or self.params['dataset'] == 'msu' or self.params['dataset'] == 'RM' or self.params['dataset'] == 'glassesDB' :
            transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.Resize((self.params['img_size'], self.params['img_size'])),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                                            
        else:
            raise ValueError('Undefined dataset type')
        return transform

