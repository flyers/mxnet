import mxnet as mx
from rllab.envs.gym_env import RLVREPHierarchyTargetEnv
import rllab.misc.logger as logger
import os
import joblib
import argparse
from ddpg import DDPG
from policies import DeterministicMLPPolicy
from qfuncs import ContinuousMLPQ
from strategies import OUStrategy

parser = argparse.ArgumentParser()
parser.add_argument('--exp-name', type=str, required=True, help='saved experiment directory name')
parser.add_argument('--resume', type=str,
                    help='resume training from previously saved location')
parser.add_argument('--itr', type=int, help='iteration number')

parser.add_argument('--reward-func', type=int, default=0, help='reward function choice')
parser.add_argument('--reward-baseline', type=float, default=2.95, help='reward baseline constant')
parser.add_argument('--reward-terminal', type=float, default=0.0, help='terminal reward penalty')

parser.add_argument('--max-path-length', type=int, default=200, help='max path length')
parser.add_argument('--batch-size', type=int, default=32, help='batch size sampled from replay memory')
parser.add_argument('--min-pool-size', type=int, default=10000, help='minimum replay pool size to start training')
parser.add_argument('--epoch-length', type=int, default=1000, help='epoch length')
parser.add_argument('--n-epochs', type=int, default=1000, help='total epoch number')
parser.add_argument('--qf-lr', type=float, default=1e-3, help='Q function learning rate')
parser.add_argument('--policy-lr', type=float, default=1e-4, help='policy network learning rate')
parser.add_argument('--scale-reward', type=float, default=1., help='scale reward by a multiplier')
args = parser.parse_args()

default_log_dir = '/home/sliay/Documents/RL-models/mxddpg'
if not os.path.exists(default_log_dir):
    os.makedirs(default_log_dir)
default_exp_name = args.exp_name

log_dir = os.path.join(default_log_dir, default_exp_name)
tabular_log_file = os.path.join(log_dir, 'progress.csv')
text_log_file = os.path.join(log_dir, 'debug.log')
params_log_file = os.path.join(log_dir, 'params.json')

logger.add_text_output(text_log_file)
logger.add_tabular_output(tabular_log_file)
logger.set_snapshot_dir(log_dir)
logger.set_snapshot_mode('all')
logger.set_log_tabular_only(False)
logger.push_prefix("[%s] " % default_exp_name)
logger.log(str(args))

env = RLVREPHierarchyTargetEnv(headless=True, reward_func=args.reward_func,
                               state_type='body',
                               reward_baseline=args.reward_baseline,
                               terminal_penalty=args.reward_terminal,
                               )

if args.resume is None:
    policy = DeterministicMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(128, 128)
    )
    strategy = OUStrategy(env.spec)
    qfunc = ContinuousMLPQ(
        env.spec,
        hidden_sizes=(32, 32)
    )
else:
    raise NotImplementedError
    # last_snapshot_dir = os.path.join(default_log_dir, args.resume)
    # data = joblib.load(os.path.join(last_snapshot_dir, 'itr_'+str(args.itr)+'.pkl'))
    # policy = data['policy']
    # qf = data['qf']
    # es = data['es']

algo = DDPG(
    env=env,
    policy=policy,
    qfunc=qfunc,
    strategy=strategy,
    ctx=mx.gpu(),
    batch_size=args.batch_size,
    max_path_length=args.max_path_length,
    epoch_length=args.epoch_length,
    memory_start_size=args.min_pool_size,
    n_epochs=args.n_epochs,
    discount=0.99,
    eval_samples=5*args.max_path_length,
    scale_reward=args.scale_reward,
    qfunc_lr=args.qf_lr,
    policy_lr=args.policy_lr,
    save_dir=log_dir
    # Uncomment both lines (this and the plot parameter below) to enable plotting
    # plot=True,
)

algo.train()