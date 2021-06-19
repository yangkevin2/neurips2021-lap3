import numpy as np

from lamcts_planning.util import rollout

class VOO:
    def __init__(self, domain, explr_p, sampling_mode, switch_counter, distance_fn=None):
        self.domain = domain
        self.dim_x = domain.shape[-1]
        self.explr_p = explr_p
        if distance_fn is None:
            self.distance_fn = lambda x, y: np.linalg.norm(x-y)

        self.switch_counter = np.inf
        self.sampling_mode = sampling_mode
        self.GAUSSIAN = False
        self.CENTERED_UNIFORM = False
        self.UNIFORM = False
        if sampling_mode == 'centered_uniform':
            self.CENTERED_UNIFORM = True
        elif sampling_mode == 'gaussian':
            self.GAUSSIAN = True
        elif sampling_mode.find('hybrid') != -1:
            self.UNIFORM = True
            self.switch_counter = switch_counter
        elif sampling_mode.find('uniform') != -1:
            self.UNIFORM = True
            self.switch_counter = switch_counter
        else:
            raise NotImplementedError

        self.UNIFORM_TOUCHING_BOUNDARY = False

    def sample_next_point(self, evaled_x, evaled_y):
        rnd = np.random.random()  # this should lie outside
        is_sample_from_best_v_region = (rnd < 1 - self.explr_p) and len(evaled_x) > 1
        if is_sample_from_best_v_region:
            x = self.sample_from_best_voronoi_region(evaled_x, evaled_y)
        else:
            x = self.sample_from_uniform()
        return x

    def choose_next_point(self, evaled_x, evaled_y):
        return self.sample_next_point(evaled_x, evaled_y)

    def sample_from_best_voronoi_region(self, evaled_x, evaled_y):
        best_dist = np.inf
        other_dists = np.array([-1])
        counter = 1

        best_evaled_x_idxs = np.argwhere(evaled_y == np.amax(evaled_y))
        best_evaled_x_idxs = best_evaled_x_idxs.reshape((len(best_evaled_x_idxs,)))
        best_evaled_x_idx = best_evaled_x_idxs[0] #np.random.choice(best_evaled_x_idxs)
        best_evaled_x = evaled_x[best_evaled_x_idx]
        other_best_evaled_xs = evaled_x

        # todo perhaps this is reason why it performs so poorly
        curr_closest_dist = np.inf

        while np.any(best_dist > other_dists):
            if self.GAUSSIAN:
                possible_max = (self.domain[1] - best_evaled_x) / np.exp(counter)
                possible_min = (self.domain[0] - best_evaled_x) / np.exp(counter)
                possible_values = np.max(np.vstack([np.abs(possible_max), np.abs(possible_min)]), axis=0)
                new_x = np.random.normal(best_evaled_x, possible_values)
                while np.any(new_x > self.domain[1]) or np.any(new_x < self.domain[0]):
                    new_x = np.random.normal(best_evaled_x, possible_values)
            elif self.CENTERED_UNIFORM:
                dim_x = self.domain[1].shape[-1]
                possible_max = (self.domain[1] - best_evaled_x) / np.exp(counter)
                possible_min = (self.domain[0] - best_evaled_x) / np.exp(counter)

                possible_values = np.random.uniform(possible_min, possible_max, (dim_x,))
                new_x = best_evaled_x + possible_values
                while np.any(new_x > self.domain[1]) or np.any(new_x < self.domain[0]):
                    possible_values = np.random.uniform(possible_min, possible_max, (dim_x,))
                    new_x = best_evaled_x + possible_values
            elif self.UNIFORM:
                new_x = np.random.uniform(self.domain[0], self.domain[1])
                if counter > self.switch_counter:
                    if self.sampling_mode.find('hybrid') != -1:
                        if self.sampling_mode.find('gaussian'):
                            self.GAUSSIAN = True
                        else:
                            self.CENTERED_UNIFORM = True
                    else:
                        break
            else:
                raise NotImplementedError

            best_dist = self.distance_fn(new_x, best_evaled_x)
            other_dists = np.array([self.distance_fn(other, new_x) for other in other_best_evaled_xs])
            counter += 1
            if best_dist < curr_closest_dist:
                curr_closest_dist = best_dist
                curr_closest_pt = new_x

        return curr_closest_pt


    def sample_from_uniform(self):
        dim_parameters = self.domain.shape[-1]
        domain_min = self.domain[0]
        domain_max = self.domain[1]
        return np.random.uniform(domain_min, domain_max, (1, dim_parameters)).squeeze()


def plan(env, env_info, args):
    lb = np.concatenate([env_info['lb'] for _ in range(args.horizon)], axis=0)
    ub = np.concatenate([env_info['ub'] for _ in range(args.horizon)], axis=0)
    domain = np.stack([lb, ub], axis=0) * args.init_sigma_mult # allow specifying a different sized region when we have something that's actually unbounded in principle
    voo = VOO(domain, args.Cp, 'centered_uniform', None) # switch_counter unused on this mode

    evaled_x = []
    evaled_y = []
    max_y = []
    action_seqs = []

    for i in range(args.iterations):
        # print "%d / %d" % (i, n_fcn_evals)
        # if i > 0:
        #     print 'max value is ', np.max(evaled_y)
        x = voo.choose_next_point(evaled_x, evaled_y)
        if len(x.shape) == 0:
            x = np.array([x])
        action_seq = x.reshape((args.horizon, env_info['action_dims']))
        action_seqs.append(action_seq)
        y, _, _ = rollout(env, env_info, action_seq, args.gamma, return_final_obs=True)
        evaled_x.append(x)
        evaled_y.append(y)
        max_y.append(np.max(evaled_y))
        if i > 0 and i % 25 == 0:
            print(i, np.max(evaled_y))

    best_idx = np.where(evaled_y == max_y[-1])[0][0]
    # print evaled_x[best_idx], evaled_y[best_idx]
    # print "Max value found", np.max(evaled_y)
    # print "Magnitude", np.linalg.norm(evaled_x[best_idx])
    # print "Explr p", explr_p

    return action_seqs[best_idx], action_seqs
