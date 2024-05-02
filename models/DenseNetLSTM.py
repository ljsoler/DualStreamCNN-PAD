import torch
from torchvision.models.resnet import resnet101
from models import BaseCNN
from torchvision import models
from torch.nn import functional as F
from archs.types_ import *
from torch import nn


class DenseNetLSTM(BaseCNN):

    def __init__(self,
                 in_channels: int,
                 **kwargs) -> None:
        super(DenseNetLSTM, self).__init__()

        self.densenet = models.densenet121(pretrained=True)

        self.densenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.densenet.classifier.in_features, 512),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=4)

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
        
        hidden = None
        for t in range(input.size(1)):
            # with torch.no_grad():
            x = self.densenet(input[:, t, :, :, :])  
            # x = self.classifier_cqt(x)
            out, hidden = self.lstm(x.unsqueeze(0), hidden)  

        x = self.fc2(out[-1, :, :])
    
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