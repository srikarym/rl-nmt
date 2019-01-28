# rl-nmt

Requires [fairseq](https://github.com/pytorch/fairseq) 

Create an account in [wandb](https://www.wandb.com/) and login using your API_KEY  

Example usage of train.py :
```
python3 -B train.py --env-name nmt_red-v0  --n-epochs-per-word 1000 --n-epochs 1000  --num-processes 25 --ppo-batch-size 800 
 --log-dir log/hun_sen/tb  --save-interval 100 --save-dir log/hundred_wandb/trained_models/ --num-steps 100 --sen_per_epoch 1 --use-wandb --wandb-name hundred_data 
```
