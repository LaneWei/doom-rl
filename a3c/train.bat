python a3c.py ^
    --mode train ^
	--threads 8 ^
	--epochs 10 ^
	--test_epochs 0 ^
	--steps 1000 ^
	--save_weights_per_epochs 5 ^
	--config_path ..\doom_configuration\doom_config\health_gathering.cfg ^
	--weights_save_path weights\health_gathering.ckpt

python a3c.py ^
    --mode train ^
	--threads 8 ^
	--epochs 50 ^
	--steps 5000 ^
	--save_weights_per_epochs 2 ^
	--config_path ..\doom_configuration\doom_config\health_gathering_hard.cfg ^
	--weights_load_path weights\health_gathering.ckpt ^
	--weights_save_path weights\health_gathering_hard.ckpt

