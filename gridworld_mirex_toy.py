import pickle
import numpy as np
import argparse
from joblib import Parallel, delayed
import datetime
from algorithms import mirex
import random




def run(id, seed, args):
    np.random.seed(seed)
    K = args.K
    goal = np.array([8, 4])
    n_iterations = args.n_iterations
    # all_states = [[[-1]], [[-1]], [[1]]]   # shape = num_trajs, num_trans, num_feats
    # all_actions = [[-1],[1],[1]]    # shape = num_trajs, num_trans
    # preferences = [(2, 1), (1, 0)]    # [(i, better_j), ...] # ! change other pref format: j>i

    # all_states = [[[2]], [[1]], [[0]]]   # shape = num_trajs, num_trans, num_feats
    # all_actions = [[0],[0],[-1]]    # shape = num_trajs, num_trans
    # preferences = [(2, 1), (1, 0)]    # [(i, better_j), ...] # ! change other pref format: j>i

    all_states = [[[2]], [[1]], [[3]]]   # shape = num_trajs, num_trans, num_feats
    all_actions = [[],[],[]]    # shape = num_trajs, num_trans
    preferences = [(2, 1), (1, 0)]    # [(i, better_j), ...] # ! change other pref format: j>i
    
    # t0 = [2] --> 0
    # t1 = [1] --> 0
    # t2 = [3] --> 1

    # t0 > t1 > t2

    # theta_0 = [1] or anything > 0
    # theta_1 = [0] or anything < 0



    all_states = np.array(all_states)
    all_actions = np.array(all_actions)
    len_trajs = [1]*3  # len = num_trajs, content = num_trans
    r = mirex.multiple_intention_irl(all_states, all_actions, preferences,
                                    len_trajs, args.num_features, K, n_iterations=n_iterations)

    return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', default=2, type=int, help='number of clusters')
    parser.add_argument('--state_dim', type=int, default=81, help='state space dimension')
    parser.add_argument('--action_dim', type=float, default=4, help='action space cardinality')
    parser.add_argument('--num_features', type=int, default=1)
    parser.add_argument('--beta', type=str, default='.5', help='comma separated valued of beta parmaeter to consider')
    parser.add_argument('--gamma', type=int, default=0.99, help='discount_factor')
    parser.add_argument('--load_path', type=str, default='data/cont_gridworld_multiple/gpomdp/')
    parser.add_argument('--n_jobs', type=int, default=1, help='number of parallel jobs')
    parser.add_argument('--n_iterations', type=int, default=20, help='number of iterations of ml-irl')
    parser.add_argument('--n_experiments', type=int, default=20, help='number of parallel jobs')
    parser.add_argument('--seed', type=int, default=-1, help='random seed, -1 to have a random seed')
    args = parser.parse_args()
    seed = args.seed
    if seed == -1:
        seed = None
    np.random.seed(seed)
    seeds = [np.random.randint(1000000) for _ in range(args.n_experiments)]
    results = Parallel(n_jobs=args.n_jobs, backend='loky')(
        delayed(run)(id, seed, args) for id, seed in zip(range(args.n_experiments), seeds))
    np.save(args.load_path + '/res_mlirl_final13.npy', results)
