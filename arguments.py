import argparse


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')

    parser.add_argument('--n-epochs', type=int, default=50,
                        help='Number of epochs')

    parser.add_argument('--num-processes', type=int, default=16,
                        help='how many training CPU processes to use (default: 16)')

    parser.add_argument('--num-steps', type=int, default=5,
                        help='number of forward steps in A2C (default: 5)')

    parser.add_argument('--max-missing-words', type=int, default=20,
                        help='Maximum number of words missing in target sentence (default: 20)')

    parser.add_argument('--n-epochs-per-word', type=int, default=10,
                        help='Num of epochs per missing word in target sentence (default: 10)')

    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')

    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')

    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='Number of ppo epochs (default: 4)')

    parser.add_argument('--ppo-batch-size', type=int, default=40,
                        help='Batch size in ppo')

    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')

    parser.add_argument('--use-gae', action='store_true', default=False,
                        help='use generalized advantage estimation')

    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')

    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')

    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')

    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')

    parser.add_argument('--log-dir', default='log/new-exp/',
                        help='directory to save tensorboard logs (default: /log/new-exp)')

    parser.add_argument('--save-dir', default='trained_models/',
                        help='directory to trained models (default: trained_models/)')

    parser.add_argument('--env-name', default='nmt-v0',
                        help='environment to train on (default: nmt-v0)')

    parser.add_argument('--save-interval', type=int, default=1,
                        help='save interval, one save per n epochs (default: 1)')

    parser.add_argument('--checkpoint', action='store_true', default=False,
                        help='resume training from checkpoint')

    parser.add_argument('--sen_per_epoch', type=int, default=0,
                        help='sentences per epoch, change this to collect more data when training with fewer sentences (default: 0) ')

    parser.add_argument('--num-sentences',type=int,default=10,
                        help='Num of sentences to train, set this to -1 to use entire data')
    
    parser.add_argument('--file-path', default='log/new-exp/',
                        help='directory to load trained models (default: /log/new-exp)')

    parser.add_argument('--wandb-name', default='new-exp/',
                        help='Project name in weights and bias (default: new-exp/)')

    parser.add_argument('--use-wandb', action='store_true', default=False,
                        help='use generalized advantage estimation')

    args = parser.parse_args()
    return args
