python /home/featurize/work/Dreambooth-Stable-Diffusion/scripts/pruning_linux.py --base /home/featurize/work/Dreambooth-Stable-Diffusion/configs/stable-diffusion/v1-frozen.yaml -t --actual_resume /home/featurize/work/models_fact/sd-v1-4.ckpt -n trainfroze --gpus 0, --data_root /home/featurize/data/train2014 --reg_data_root /home/featurize/data/horse-or-human/validation/horses --class_word horse


