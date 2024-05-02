#!/bin/bash

#DEV STFT
# echo "DEV STFT"
# python test_asvspoof.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/LA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/STFT/DenseNetLSTM/version_0/checkpoints/_ckpt_epoch_0.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/STFT/DenseNetLSTM/version_0/checkpoints -p Dev;

# python test_asvspoof.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/LA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/STFT/DenseNetLSTM/version_1/checkpoints/_ckpt_epoch_4.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/STFT/DenseNetLSTM/version_1/checkpoints -p Dev;

# python test_asvspoof.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/LA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/STFT/DenseNetLSTM/version_2/checkpoints/_ckpt_epoch_0.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/STFT/DenseNetLSTM/version_2/checkpoints -p Dev;

# python test_asvspoof.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/LA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/STFT/DenseNetLSTM/version_3/checkpoints/_ckpt_epoch_1.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/STFT/DenseNetLSTM/version_3/checkpoints -p Dev;

# python test_asvspoof.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/LA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/STFT/DenseNetLSTM/version_4/checkpoints/_ckpt_epoch_0.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/STFT/DenseNetLSTM/version_4/checkpoints -p Dev;

# #EVAL STFT
# echo "EVAL STFT"
# python test_asvspoof.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/LA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/STFT/DenseNetLSTM/version_0/checkpoints/_ckpt_epoch_0.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/STFT/DenseNetLSTM/version_0/checkpoints -p Eval;

# python test_asvspoof.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/LA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/STFT/DenseNetLSTM/version_1/checkpoints/_ckpt_epoch_4.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/STFT/DenseNetLSTM/version_1/checkpoints -p Eval;

# python test_asvspoof.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/LA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/STFT/DenseNetLSTM/version_2/checkpoints/_ckpt_epoch_0.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/STFT/DenseNetLSTM/version_2/checkpoints -p Eval;

# python test_asvspoof.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/LA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/STFT/DenseNetLSTM/version_3/checkpoints/_ckpt_epoch_1.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/STFT/DenseNetLSTM/version_3/checkpoints -p Eval;

# python test_asvspoof.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/LA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/STFT/DenseNetLSTM/version_4/checkpoints/_ckpt_epoch_0.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/STFT/DenseNetLSTM/version_4/checkpoints -p Eval;

#DEV CQT
echo "DEV CQT"
python test_asvspoof.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/LA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/CQT/DenseNetLSTM/version_0/checkpoints/_ckpt_epoch_3.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/CQT/DenseNetLSTM/version_0/checkpoints -p Dev;

python test_asvspoof.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/LA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/CQT/DenseNetLSTM/version_1/checkpoints/_ckpt_epoch_1.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/CQT/DenseNetLSTM/version_1/checkpoints -p Dev;

python test_asvspoof.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/LA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/CQT/DenseNetLSTM/version_2/checkpoints/_ckpt_epoch_2.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/CQT/DenseNetLSTM/version_2/checkpoints -p Dev;

python test_asvspoof.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/LA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/CQT/DenseNetLSTM/version_3/checkpoints/_ckpt_epoch_1.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/CQT/DenseNetLSTM/version_3/checkpoints -p Dev;

python test_asvspoof.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/LA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/CQT/DenseNetLSTM/version_4/checkpoints/_ckpt_epoch_2.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/CQT/DenseNetLSTM/version_4/checkpoints -p Dev;

#EVAL CQT
echo "EVAL CQT"
python test_asvspoof.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/LA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/CQT/DenseNetLSTM/version_0/checkpoints/_ckpt_epoch_3.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/CQT/DenseNetLSTM/version_0/checkpoints -p Eval;

python test_asvspoof.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/LA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/CQT/DenseNetLSTM/version_1/checkpoints/_ckpt_epoch_1.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/CQT/DenseNetLSTM/version_1/checkpoints -p Eval;

python test_asvspoof.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/LA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/CQT/DenseNetLSTM/version_2/checkpoints/_ckpt_epoch_2.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/CQT/DenseNetLSTM/version_2/checkpoints -p Eval;

python test_asvspoof.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/LA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/CQT/DenseNetLSTM/version_3/checkpoints/_ckpt_epoch_1.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/CQT/DenseNetLSTM/version_3/checkpoints -p Eval;

python test_asvspoof.py -i /home/janier.soler/Databases/AUDIO/ASVspoof2019/MAT/LA -m /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/CQT/DenseNetLSTM/version_4/checkpoints/_ckpt_epoch_2.ckpt -o /home/janier.soler/Projects/\[Torch\]Dual-StreamVoicePAD/logs/LA/CQT/DenseNetLSTM/version_4/checkpoints -p Eval;
