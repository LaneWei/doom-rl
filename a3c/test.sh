#!/usr/bin/zsh

python a3c.py \
	--test_epochs 2 \
	--config_path ../doom_configuration/doom_config/health_gathering.cfg \
	--weights_load_dir weights/ \
	--test_visible \
	--test_verbose \
	--test_spectator


