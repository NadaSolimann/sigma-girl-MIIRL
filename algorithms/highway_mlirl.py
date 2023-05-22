import numpy as np


##Implementation of Monica Babes-Vroman Apprenticeship Learning About Multiple Intentions

def _stateIndex(s, states_dict):
    assert(not np.isnan(s[0]))
    s = s.astype(int)
    return states_dict[f"{s[0]}_{s[1]}_{s[2]}_{s[3]}_{s[4]}"]


def get_transition_matrix(states, len_trajs, states_idx_dict, actions, state_space, action_space):
    transition_mat = np.zeros((state_space, action_space, state_space))

    for t in range(len(states)):    # trajs
        for i in range(len_trajs[t] - 1):  # trans states in traj
            curr_state = states[t][i]
            next_state = states[t][i + 1]
            action = int(actions[t][i])

            curr_idx = _stateIndex(curr_state, states_idx_dict)
            next_idx = _stateIndex(next_state, states_idx_dict)

            # Update the transition probability
            transition_mat[curr_idx, action, next_idx] += 1

    # Normalize the transition probabilities
    transition_mat /= np.expand_dims(transition_mat.sum(-1), axis=-1)
    # equal probabilities to all next states if (s, a) not visited in data
    transition_mat = np.nan_to_num(transition_mat, nan=-1000)

    # ? their trans_mat has prob[n,:].sum() = # actions
    # ? may have issue with -1000 (unvisited states)
    # ? for them, all states are visited/can calulate trans_prob
    return transition_mat


def initialize_variables(rewards, param, state_space, action_space):
    q = np.zeros((state_space, action_space))
    V = np.zeros(state_space)
    d_V = np.zeros((state_space, rewards.shape[1]))
    for state in range(state_space):
        for action in range(action_space):
            q[state, action] = np.dot(rewards[state], param)
        V[state] = np.dot(rewards[state], param)
        d_V[state] = rewards[state]
    return q, V, d_V


def evaluate_gradients(states, actions, param, len_trajs, states_idx_dict, state_space, action_space, gamma, features, prob, beta, weights, p=0,
                       n_iterations=1):
    num_feat_rewards = len(param)
    q, V, d_V = initialize_variables(features, param, state_space, action_space)
    d_q = np.zeros((state_space, action_space, num_feat_rewards))
    pi = np.zeros((state_space, action_space))
    d_pi = np.zeros((state_space, action_space, num_feat_rewards))
    grad = np.zeros((len(states), len(states[0]), num_feat_rewards))

    for i in range(n_iterations):
        for state in range(state_space):  ##OK
            for action in range(action_space):
                q[state, action] = np.dot(param, features[state]) + gamma * np.dot(prob[state, action, :], V)
                d_q[state, action] = features[state] + gamma * np.dot(prob[state, action, :], d_V)

        exp_q = np.exp(beta * (q - np.max(q, axis=1)[:, np.newaxis]))
        zeta = np.sum(exp_q, axis=1)[:, np.newaxis]
        d_zeta = beta * np.sum(exp_q[:, :, np.newaxis] * d_q, axis=1)

        pi = exp_q / zeta
        d_pi_first = beta * (zeta * exp_q)[:, :, np.newaxis] * d_q
        d_pi_second = exp_q[:, :, np.newaxis] * d_zeta[:, np.newaxis, :]
        d_pi = (d_pi_first - d_pi_second) / zeta[:, :, np.newaxis] ** 2

        V = np.sum(pi * q, axis=1)
        d_V = np.sum(q[:, :, np.newaxis] * d_pi + pi[:, :, np.newaxis] * d_q, axis=1)

    L = 0
    for n in range(len(states)):
        traj_norm_L = 0
        for t in range(len_trajs[n]):
            state = _stateIndex(states[n, t], states_idx_dict)
            action = int(actions[n, t])
            grad[n, t] = 1 / (pi[state, action] + 1.e-10) * d_pi[state, action]
            traj_norm_L += beta * (q[state, action] - np.max(q[state], axis=-1)) - np.log(zeta[state])
        grad[n, :] *= weights[n]
        traj_norm_L /= len_trajs[n]
        L += traj_norm_L

    return L, np.mean(np.sum(grad, axis=1), axis=0), q


def maximum_likelihood_irl(states, actions, features, probs, init_param, len_trajs, states_idx_dict, state_space, action_space, beta,
                           q, weights=None, gamma=0.99, n_iteration=10, gradient_iterations=100):
    param = np.array(init_param)
    grad = []
    for i in range(n_iteration):
        L, gradients, q = evaluate_gradients(states, actions, param, len_trajs, states_idx_dict, state_space, action_space, gamma, features,
                                          probs, beta, weights,
                                          n_iterations=gradient_iterations)
        param += .1 * gradients
    return param, q, grad


def multiple_intention_irl(states, actions, features, K, gt_intents, len_trajs, states_idx_dict, state_space, action_space, beta,
                           gamma=0.99, n_iterations=20,
                           tolerance=1.e-5,
                           p=0.):
    ## transition probs
    probs = get_transition_matrix(states, len_trajs, states_idx_dict, actions, state_space, action_space)
    rho_s = np.ones(K)  ## define prior probability
    rho_s = rho_s / np.sum(rho_s)
    theta = np.random.random((K, features.shape[1]))  # reward feature
    q = np.zeros((K, state_space, action_space))  # Q values

    z = np.random.random((rho_s.shape[0], states.shape[0]))
    z /= np.sum(z, axis=0)[np.newaxis, :]
    prev_assignment = np.ones(z.shape)
    it = 0

    max_iteration = 20
    while it < max_iteration and np.max(np.abs(z - prev_assignment)) > tolerance:
        print(f"Iteration {it}")
        prev_assignment = z
        z = e_step(states, actions, len_trajs, states_idx_dict, theta, rho_s, action_space, q, beta).T  # K,N
        it += 1
        for i in range(z.shape[0]):
            theta[i, :], q[i, :], grad = maximum_likelihood_irl(states=states,
                                                                actions=actions,
                                                                features=features,
                                                                probs=probs,
                                                                init_param=theta[i],
                                                                len_trajs=len_trajs,
                                                                states_idx_dict=states_idx_dict,
                                                                state_space=state_space,
                                                                action_space=action_space,
                                                                beta=beta,
                                                                q=q[i],
                                                                weights=z[i],
                                                                gamma=gamma,
                                                                n_iteration=n_iterations)
        rho_s = np.sum(z, axis=1) / len(states)
    return z, theta


def e_step(trajs, actions, len_trajs, states_idx_dict, theta, rho_s, action_space, q, beta):
    zeta = np.ones((trajs.shape[0], rho_s.shape[0]))
    for t in range(len(trajs)):
        for s in range(len_trajs[t]):
            state = int(_stateIndex(trajs[t, s], states_idx_dict))
            action = int(actions[t, s])
            if t == 38:
                pass
            for k in range(theta.shape[0]):
                prob = [np.exp(beta * (q[k, state, act] - max(q[k, state]))) for act in range(action_space)]
                pi = prob[action] / np.sum(prob)
                zeta[t, k] *= pi
        for k in range(theta.shape[0]):
            zeta[t, k] *= rho_s[k]
        zeta[t, :] /= np.sum(zeta[t, :])
    return zeta

