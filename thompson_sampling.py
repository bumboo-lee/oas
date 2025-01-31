import numpy as np

ACTIONS = ["Accept", "Reject", "Postpone", "Outsource"]

# Thompson Sampling 파라미터(액션별)
action_params = {
    a: {"mean": 0.0, "var": 1000.0, "count": 1}
    for a in ACTIONS
}

def thompson_sampling_select_action():
    samples = {}
    for a in ACTIONS:
        mu = action_params[a]["mean"]
        var = action_params[a]["var"]
        s = np.random.normal(mu, var**0.5)
        samples[a] = s
    return max(samples, key=samples.get)

def update_thompson_params(action, reward):
    c = action_params[action]["count"]
    old_mean = action_params[action]["mean"]
    new_mean = old_mean + (reward - old_mean)/(c+1)

    old_var = action_params[action]["var"]
    new_var = ((c*old_var)
               + (reward - old_mean)*(reward - new_mean)) / (c+1)

    action_params[action]["mean"] = new_mean
    action_params[action]["var"] = max(1e-9, new_var)
    action_params[action]["count"] = c + 1
