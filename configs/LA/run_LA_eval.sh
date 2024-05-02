#!/bin/bash

python test_asvspoof_hybrid.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/LA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/MobileNetV2LSTM_B/version_0/checkpoints/_ckpt_epoch_0.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/MobileNetV2LSTM_B/version_0/checkpoints -p Eval;

python test_asvspoof_hybrid.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/LA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/MobileNetV2LSTM_B/version_1/checkpoints/_ckpt_epoch_1.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/MobileNetV2LSTM_B/version_1/checkpoints -p Eval;

python test_asvspoof_hybrid.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/LA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/MobileNetV2LSTM_B/version_2/checkpoints/_ckpt_epoch_4.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/MobileNetV2LSTM_B/version_2/checkpoints -p Eval;

python test_asvspoof_hybrid.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/LA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/MobileNetV2LSTM_B/version_3/checkpoints/_ckpt_epoch_0.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/MobileNetV2LSTM_B/version_3/checkpoints -p Eval;

python test_asvspoof_hybrid.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/LA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/MobileNetV2LSTM_B/version_4/checkpoints/_ckpt_epoch_4.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/MobileNetV2LSTM_B/version_4/checkpoints -p Eval;
