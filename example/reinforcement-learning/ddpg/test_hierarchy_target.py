import os
import argparse
import logging
import sys
import numpy

import mxnet as mx
from rllab.envs.gym_env import RLVREPHierarchyTargetEnv
from policies import DeterministicMLPPolicy


parser = argparse.ArgumentParser(description='Script to simulate controllers for vrep simple env.')
parser.add_argument('--alg', type=str, help='using which training algorithm')
parser.add_argument('--path', type=str, help='model saved location')
parser.add_argument('--itr', type=int, help='iteration number')
parser.add_argument('--log', type=str, help='log file')

parser.add_argument('--reward-func', type=int, default=0, help='reward function choice')
parser.add_argument('--reward-baseline', type=float, default=2.95, help='reward baseline constant')
parser.add_argument('--reward-terminal', type=float, default=0.0, help='terminal reward penalty')
args = parser.parse_args()

logger = logging.getLogger()
if args.log is None:
    ch = logging.StreamHandler(sys.stdout)
else:
    ch = logging.FileHandler(os.path.join('log', 'ros', args.log))
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

default_log_dir = '/home/sliay/Documents/RL-models'

# data = joblib.load(os.path.join(default_log_dir, args.alg, args.path, 'itr_'+str(args.itr)+'.pkl'))
# policy = data['policy']

env = RLVREPHierarchyTargetEnv(headless=False, log=False,
                               reward_func=args.reward_func,
                               reward_baseline=args.reward_baseline,
                               terminal_penalty=args.reward_terminal,
                               state_type="body",
                               # obs_type="image",
                               scene_path='/home/sliay/Documents/vrep-uav/scenes/quadcopter_hierarchy_64x64.ttt'
                               )

policy = DeterministicMLPPolicy(
    env_spec=env.spec,
    hidden_sizes=(128, 128)
)
updater = mx.optimizer.get_updater(
    mx.optimizer.create('adam', learning_rate=1e-3)
)
policy.define_exe(ctx=mx.cpu(), updater=updater, input_shapes={'obs': (1, 14)})
policy.load_params(dir_path=os.path.join(default_log_dir, args.alg, args.path), itr=143)

rollout_num = 1
max_path_length = 10000
for rollout in xrange(rollout_num):
    observations = []
    actions = []
    rewards = []
    o = env.reset()
    # env.render()
    for path_length in xrange(max_path_length):
        o = o.reshape(1, -1)
        a = policy.get_action(o)
        print 'action: ', a
        next_o, r, terminate, env_info = env.step(a)
        # observations.append(env.observation_space.flatten(o))
        actions.append(env.action_space.flatten(a))
        rewards.append(r)
        # print 'state:', next_o
        o = next_o
        if terminate:
            break
        # env.render()
    print('Accumulated Rewards:%.4f' % numpy.sum(rewards))

