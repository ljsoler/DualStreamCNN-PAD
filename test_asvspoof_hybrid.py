import yaml
import argparse
from torchvision import transforms
from models import * 
from experiment import VAEXperiment
from experiment import ImglistToTensor
import torch
from torch.utils.data import DataLoader
from datasets.ASVSpoofH import ASVSpoofH
from pathlib import Path
import numpy as np
from pyeer.eer_info import get_eer_stats
import os

models = ['configs/LA/hybridlstm_D.yaml']

arch = ['HybridLSTM']

parser = argparse.ArgumentParser(description='Generic runner for CNN models')
parser.add_argument('--input',  '-i',
                    dest="input_folder",
                    metavar='FILE',
                    help =  'path to the input folder which contains images',
                    default='D:/Work/Databases/ASVSpoof2019/CQT/LA')

parser.add_argument('--model',  '-m',
                    dest="model_folder",
                    metavar='FILE',
                    help =  'path to the input folder which contains models',
                    default='D:/Work/Projects/PAD_projects/presentation-attack-detection-sdk/[Torch]TransferLearning2Classification/logs/CNNLSTM/version_0/checkpoints/_ckpt_epoch_2.ckpt')

parser.add_argument('--output',  '-o',
                    dest="output_folder",
                    metavar='FILE',
                    help =  'path to output input folder where scores will be saved')

parser.add_argument('--partition',  '-p',
                    dest="partition",
                    type=str,
                    help =  'partition to evaluate',
                    default='Dev')

args = parser.parse_args()

model_weights = args.model_folder

for m, a in zip(models, arch):
    with open(m, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    config['exp_params']['data_path'] = args.input_folder

    print("Computing stats for {}".format(a))

    extension = 'mat'

    transform = transforms.Compose([ImglistToTensor()])

    approach = cnn_models[config['model_params']['name']](**config['model_params'])
    approach.to(torch.device('cuda:{}'.format(list(config['trainer_params']['gpus'])[0])))

    experiment = VAEXperiment(approach,
                    config['exp_params'])

    checkpoint = torch.load(str(model_weights), map_location=lambda storage, loc: storage.cuda(int(list(config['trainer_params']['gpus'])[0])))

    experiment.load_state_dict(checkpoint['state_dict'])

    experiment.eval()

    dataset = DataLoader(ASVSpoofH(root = config['exp_params']['data_path'],
                        split = args.partition,
                        ext=config['exp_params']['ext'],
                        sliding_window= config['exp_params']['sliding_window'],
                        frames=config['exp_params']['frames'],
                        img_size=config['exp_params']['img_size'],
                        transform=transform),
                    batch_size= config['exp_params']['batch_size'],
                    shuffle = True)

    scores, labels = [], []

    c = 0

    for (img_cqt, img_stft), l in dataset:

        if c != 0 and c % 1000 == 0:
            print('Processing {}'.format(c))

        c += int(config['exp_params']['batch_size'])

        img_cqt = img_cqt.to('cuda:{}'.format(list(config['trainer_params']['gpus'])[0]))

        img_stft = img_stft.to('cuda:{}'.format(list(config['trainer_params']['gpus'])[0]))

        classification = experiment.predict((img_cqt, img_stft))

        targets = l.detach().numpy()

        sc = classification.cpu().detach().numpy()[:, 0]

        scores = [*scores, *sc]

        labels = [*labels, *targets]

    bp = np.array([s for s, la in zip(scores, labels) if la > 0])

    ap = np.array([s for s, la in zip(scores, labels) if la == 0])

    stats_a = get_eer_stats(bp, ap)

    print('D-EER: {}'.format(stats_a.eer*100))

    np.savetxt(os.path.join(args.output_folder, 'genuine_STFT_CQT_{}.txt'.format(args.partition)), bp, fmt='%.4e')

    np.savetxt(os.path.join(args.output_folder, 'impostor_STFT_CQT_{}.txt'.format(args.partition)), ap, fmt='%.4e')

    # print('Size: {}, region {} - D-EER: {}'.format(sz, f, stats_a.eer*100))

            

            


