import os

mal_args = ' --processes 10 --malicious 3 --mal_type sign' #TODO check act impl and impl 3rd attack
env_args = ' --env CartPole-v1 --steps 100000 --lr 1e-3 --beta 2e-5 --t-max 5 --activation 1 --hidden_size 64 --ucb_disable 500'
for seed in range(0, 1):
    os.system('python train_a3c.py --seed ' + str(seed) + mal_args + env_args)
