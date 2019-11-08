#!/usr/bin/zsh


python a3c.py \
    --mode train \
	--test_epochs 0 \
	--epochs 10 \
	--steps 500 \
	--save_weights_per_epochs 5 \
	--config_path ../doom_configuration/doom_config/health_gathering.cfg \
	--weights_save_path weights/easy/health_gathering.ckpt \

python a3c.py \
    --mode train \
	--test_epochs 5 \
	--epochs 20 \
	--steps 1000 \
	--save_weights_per_epochs 5 \
	--config_path ../doom_configuration/doom_config/health_gathering_hard.cfg \
	--weights_save_path weights/health_gathering_hard.ckpt \
	--weights_load_dir weights/easy/

./test.sh
