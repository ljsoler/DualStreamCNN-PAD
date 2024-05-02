import torch
from models import BaseCNN
from efficientnet_pytorch import EfficientNet
from torch.nn import functional as F
from archs.types_ import *
from torch import nn
from losses.BinaryMarginAngularLoss import BinaryMarginAngularLoss


class EfficientnetHLSTM(BaseCNN):

    def __init__(self,
                 in_channels: int,
                 **kwargs) -> None:
        super(EfficientnetHLSTM, self).__init__()

        self.efficientnet_cqt = EfficientNet.from_pretrained('efficientnet-b7', include_top=False, in_channels=3)

        self.efficientnet_stft = EfficientNet.from_pretrained('efficientnet-b7', include_top=False, in_channels=3)

        self.lstm_cqt = nn.LSTM(input_size=2560, hidden_size=256, num_layers=4)

        self.lstm_stft = nn.LSTM(input_size=2560, hidden_size=256, num_layers=4)

        self.fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        ) 
        self.fc2 = nn.Linear(256, 1, bias=False)
        
        self.ce = BinaryMarginAngularLoss()
        
    
    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False


    def predict(self, input: Tensor) -> Any:
        
        tensor_cqt, tensor_stft = input
        
        hidden = None
        for t in range(tensor_cqt.size(1)):
            with torch.no_grad():
                x = self.efficientnet_cqt(tensor_cqt[:, t, :, :, :])  
            x = self.fc0_cqt(x.view(-1, 2560))
            out_cqt, hidden = self.lstm_cqt(x.unsqueeze(0), hidden)  

        hidden = None
        for t in range(tensor_stft.size(1)):
            with torch.no_grad():
                x = self.efficientnet_stft(tensor_stft[:, t, :, :, :]) 
            x = self.fc0_stft(x.view(-1, 2560)) 
            out_stft, hidden = self.lstm_stft(x.unsqueeze(0), hidden)   

        out = torch.cat((out_cqt, out_stft), dim=2)

        embedding = self.fc1(out[-1, :, :])

        for W in self.fc2.parameters():
            W = F.normalize(W, p=2, dim=1)
        embedding = F.normalize(embedding, p=2, dim=1)

        x = self.fc2(embedding)
    
        return x, embedding


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
        x, _ = args[0]
        real_labels = args[1]

        loss_binary = self.loss(x, real_labels)

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
        x, _ = args[0]
        real_labels = args[1]

        classification_output = nn.Sigmoid()(x)

        winner = classification_output

        winner[winner >= 0.5] = 1

        winner[winner < 0.5] = 0

        accuracy = (winner[:, 0] == real_labels).sum().float()/real_labels.size(0)*100

        return {'accuracy': accuracy}