#!/bin/bash

python test_asvspoof_hybrid.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/PA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/PA/MobileNetV2LSTM_B/version_0/checkpoints/_ckpt_epoch_3.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/PA/MobileNetV2LSTM_B/version_0/checkpoints -p Eval;

python test_asvspoof_hybrid.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/PA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/PA/MobileNetV2LSTM_B/version_1/checkpoints/_ckpt_epoch_4.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/PA/MobileNetV2LSTM_B/version_1/checkpoints -p Eval;

python test_asvspoof_hybrid.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/PA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/PA/MobileNetV2LSTM_B/version_2/checkpoints/_ckpt_epoch_3.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/PA/MobileNetV2LSTM_B/version_2/checkpoints -p Eval;

python test_asvspoof_hybrid.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/PA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/PA/MobileNetV2LSTM_B/version_3/checkpoints/_ckpt_epoch_3.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/PA/MobileNetV2LSTM_B/version_3/checkpoints -p Eval;

python test_asvspoof_hybrid.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/PA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/PA/MobileNetV2LSTM_B/version_4/checkpoints/_ckpt_epoch_4.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/PA/MobileNetV2LSTM_B/version_4/checkpoints -p Eval;
