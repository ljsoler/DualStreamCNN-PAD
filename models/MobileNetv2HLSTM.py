import torch
from torchvision.models.resnet import resnet101
from losses import BinaryMarginAngularLoss
from models import BaseCNN
from torchvision import models
from torch.nn import functional as F
from archs.types_ import *
from torch import nn
from losses.BinaryMarginAngularLoss import BinaryMarginAngularLoss, BinaryAngularLoss


class MobileNetv2HLSTM(BaseCNN):

    def __init__(self,
                 in_channels: int,
                 **kwargs) -> None:
        super(MobileNetv2HLSTM, self).__init__()

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
            # nn.ReLU()
        )

        # self.fc2 = nn.Linear(256, 1, bias=False) 

        self.angular_loss = BinaryAngularLoss(256, 1)
        
    
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

        embedding = self.fc1(out[-1, :, :])

        return embedding


    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:

        labels = kwargs['labels']

        em = self.predict(input)

        L1, sig = self.angular_loss(em, labels)

        return  [(L1, sig), kwargs['labels']]
        

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        L1, _ = args[0]
        # real_labels = args[1]

        return {'loss': L1}


    def accuracy(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the accuracy.
        :param args:
        :param kwargs:
        :return:
        """
        _, sig = args[0]
        real_labels = args[1]

        winner = sig

        winner[winner >= 0.5] = 1

        winner[winner < 0.5] = 0

        accuracy = (winner[:, 0] == real_labels).sum().float()/real_labels.size(0)*100

        return {'accuracy': accuracy}