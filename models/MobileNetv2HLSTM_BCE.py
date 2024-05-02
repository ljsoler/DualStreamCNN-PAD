import torch
from torch.nn.modules.loss import BCELoss
from torchvision.models.resnet import resnet101
from models import BaseCNN
from torchvision import models
from torch.nn import functional as F
from archs.types_ import *
from torch import nn


class MobileNetv2HLSTM_BCE(BaseCNN):

    def __init__(self,
                 in_channels: int,
                 **kwargs) -> None:
        super(MobileNetv2HLSTM_BCE, self).__init__()

        self.mobilenet_cqt = models.mobilenet_v2(pretrained=True)
        # self.set_parameter_requires_grad(self.mobilenet_cqt, True)

        self.mobilenet_stft = models.mobilenet_v2(pretrained=True)
        # self.set_parameter_requires_grad(self.mobilenet_stft, True)

        self.mobilenet_cqt.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.mobilenet_cqt.last_channel, 512),
            nn.ReLU(),
        )

        self.mobilenet_stft.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.mobilenet_cqt.last_channel, 512),
            nn.ReLU(),
        )

        self.lstm_cqt = nn.LSTM(input_size=512, hidden_size=256, num_layers=4)

        self.lstm_stft = nn.LSTM(input_size=512, hidden_size=256, num_layers=4)

        self.fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        ) 

        self.loss = BCELoss()
        
    
    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False


    def predict(self, input: Tensor) -> Any:
        
        tensor_cqt, tensor_stft = input
        
        hidden = None
        for t in range(tensor_cqt.size(1)):
            # with torch.no_grad():
            x = self.mobilenet_cqt(tensor_cqt[:, t, :, :, :])  
            # x = self.classifier_cqt(x)
            out_cqt, hidden = self.lstm_cqt(x.unsqueeze(0), hidden)  

        hidden = None
        for t in range(tensor_stft.size(1)):
            # with torch.no_grad():
            x = self.mobilenet_stft(tensor_stft[:, t, :, :, :])
            # x = self.classifier_stft(x)  
            out_stft, hidden = self.lstm_stft(x.unsqueeze(0), hidden)   

        out = torch.cat((out_cqt, out_stft), dim=2)

        x = self.fc1(out[-1, :, :])

        x = self.fc2(x)
    
        return x


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:

        return  [self.predict(input), kwargs['labels']]
        

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        x = args[0]
        real_labels = args[1]

        loss_binary = self.loss(x, real_labels.unsqueeze(1).float())

        return {'loss': loss_binary}


    def accuracy(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the accuracy.
        :param args:
        :param kwargs:
        :return:
        """
        x = args[0]
        real_labels = args[1]

        winner = x

        winner[winner >= 0.5] = 1

        winner[winner < 0.5] = 0

        accuracy = (winner[:, 0] == real_labels).sum().float()/real_labels.size(0)*100

        return {'accuracy': accuracy}