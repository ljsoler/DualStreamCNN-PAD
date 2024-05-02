import yaml
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from pytorch_lightning.logging import TestTubeLogger
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision import transforms
from models import * 
from experiment import VAEXperiment
from experiment import ImglistToTensor
from datasets.ASVSpoofH import ASVSpoofH
from pathlib import Path
import scipy

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/LA/hybridlstm_D.yaml')

parser.add_argument('--input', '-i', type=str, default='Test')

parser.add_argument('--model', '-m', type=str, default='/home/janier.soler/Projects/[Torch]Dual-StreamVoicePAD/logs/LA/RandomSelection/DenseNetLSTM_B/version_3/checkpoints/_ckpt_epoch_2.ckpt')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


tt_logger = TestTubeLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
    debug=False,
    create_git_tag=False,
)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

model = cnn_models[config['model_params']['name']](**config['model_params'])
model.to(torch.device('cpu'))

model.eval()

experiment = VAEXperiment(model,
                    config['exp_params'])

checkpoint = torch.load(str(args.model))

experiment.load_state_dict(checkpoint['state_dict'])

experiment.eval()

transform = transforms.Compose([ImglistToTensor()])

img_path_CQT = '/home/janier.soler/Databases/AUDIO/ASVSpoof2021/LA/Eval/CQT/bonafide/LA_E_1000450.mat'

img_path_STFT = '/home/janier.soler/Databases/AUDIO/ASVSpoof2021/LA/Eval/STFT/bonafide/LA_E_1000450.mat'

mat = scipy.io.loadmat(img_path_CQT, verify_compressed_data_integrity=False)

m = mat['features']

img_cqt = Image.fromarray(m).convert("RGB")

# _, h = img_cqt.size

img_cqt = img_cqt.resize((int(config['exp_params']['sliding_window'])*int(config['exp_params']['frames']), int(config['exp_params']['img_size'])))

narray_cqt = np.array(img_cqt)

#loading stft images
mat = scipy.io.loadmat(img_path_STFT, verify_compressed_data_integrity=False)

m = mat['features']

img_stft = Image.fromarray(m).convert("RGB")

# _, h = img_stft.size

img_stft = img_stft.resize((int(config['exp_params']['sliding_window'])*int(config['exp_params']['frames']), int(config['exp_params']['img_size'])))

narray_stft = np.array(img_stft)

images_cqt, images_stft = [], []

for i in range(int(config['exp_params']['frames'])):

    im_tmp = narray_cqt[:, i*int(config['exp_params']['sliding_window']): i*int(config['exp_params']['sliding_window']) + int(config['exp_params']['sliding_window']), :]

    images_cqt.append(Image.fromarray(im_tmp))

    #adding sftp frame
    im_tmp = narray_stft[:, i*int(config['exp_params']['sliding_window']): i*int(config['exp_params']['sliding_window']) + int(config['exp_params']['sliding_window']), :]

    images_stft.append(Image.fromarray(im_tmp))


tensor_cqt = transform(images_cqt)

print(tensor_cqt.shape)

tensor_stft = transform(images_stft)

target_layers = [model.densenet_cqt.features[0], model.densenet_stft.features[0]]

cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

grayscale_cam = cam(input_tensor=(tensor_cqt.unsqueeze(0), tensor_stft.unsqueeze(0)))

print(grayscale_cam.shape)

# result = overlay_mask(img, to_pil_image(activation_map, mode='F'), alpha=0.5)
# # Display it
# plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()

# plt.imshow(activation_map.numpy()); plt.axis('off'); plt.tight_layout(); plt.show()