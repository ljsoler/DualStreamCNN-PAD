from .base import *
from .DenseNetHLSTM_BCE import DenseNetHLSTM_BCE
from.DenseNetHLSTM import DenseNetHLSTM
from .ResNetHLSTM_BCE import ResNetHLSTM_BCE
from .ResNetHLSTM import ResNetHLSTM
from .MobileNetv2HLSTM_BCE import MobileNetv2HLSTM_BCE
from .MobileNetv2HLSTM import MobileNetv2HLSTM
from .EfficientNetHLSTM_BCE import EfficientNetHLSTM_BCE
from .EfficientNetHLSTM import EfficientnetHLSTM
from .DenseNetLSTM import DenseNetLSTM


# Aliases
# VAE = VanillaVAE
# GaussianVAE = VanillaVAE
# CVAE = ConditionalVAE
# GumbelVAE = CategoricalVAE

cnn_models = {'DenseNetLSTM_B': DenseNetHLSTM_BCE,
              'DenseNetHLSTM':DenseNetHLSTM,
              'ResNetLSTM_B':ResNetHLSTM_BCE,
              'ResNetHLSTM': ResNetHLSTM,
              'MobileNetV2LSTM_B': MobileNetv2HLSTM_BCE,
              'MobileNetv2HLSTM': MobileNetv2HLSTM,
              'EfficientNetLSTM_B': EfficientNetHLSTM_BCE,
              'EfficientnetHLSTM': EfficientnetHLSTM,
              'DenseNetLSTM': DenseNetLSTM}
        
