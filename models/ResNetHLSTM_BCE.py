import torch
from models import BaseCNN
from torchvision import models
from torch.nn import functional as F
from archs.types_ import *
from torch import nn


class ResNetHLSTM_BCE(BaseCNN):

    def __init__(self,
                 in_channels: int,
                 **kwargs) -> None:
        super(ResNetHLSTM_BCE, self).__init__()

        self.resnet_cqt = models.resnet34(pretrained=True)

        self.resnet_stft = models.resnet34(pretrained=True)

        self.resnet_cqt.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.resnet_cqt.fc.in_features, 512),
            nn.ReLU()
        )

        self.resnet_stft.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.resnet_stft.fc.in_features, 512),
            nn.ReLU()
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
        
        self.loss = nn.BCELoss()
        
    
    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False


    def predict(self, input: Tensor) -> Any:
        
        tensor_cqt, tensor_stft = input
        
        hidden = None
        for t in range(tensor_cqt.size(1)):
            x = self.resnet_cqt(tensor_cqt[:, t, :, :, :])  
            out_cqt, hidden = self.lstm_cqt(x.unsqueeze(0), hidden)  

        hidden = None
        for t in range(tensor_stft.size(1)):
            x = self.resnet_stft(tensor_stft[:, t, :, :, :]) 
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