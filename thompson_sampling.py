import numpy as np
import random
from sklearn.tree import DecisionTreeRegressor

ACTIONS = ["Accept", "Reject", "Postpone", "Outsource"]

# 기존 Thompson Sampling 파라미터 (컨텍스트 밴딧용)
action_params = {
    a: {"mean": 0.0, "var": 1000.0, "count": 1}
    for a in ACTIONS
}

def thompson_sampling_select_action():
    samples = {}
    for a in ACTIONS:
        mu = action_params[a]["mean"]
        var = action_params[a]["var"]
        s = np.random.normal(mu, np.sqrt(var))
        samples[a] = s
    return max(samples, key=samples.get)

def update_thompson_params(action, reward):
    c = action_params[action]["count"]
    old_mean = action_params[action]["mean"]
    new_mean = old_mean + (reward - old_mean) / (c + 1)
    old_var = action_params[action]["var"]
    new_var = ((c * old_var) + (reward - old_mean) * (reward - new_mean)) / (c + 1)
    action_params[action]["mean"] = new_mean
    action_params[action]["var"] = max(1e-9, new_var)
    action_params[action]["count"] = c + 1

# treebootstrap을 위한 데이터 저장소 (각 액션별 관측된 context와 reward 기록)
tree_data = {a: {"X": [], "y": []} for a in ACTIONS}

def treebootstrap_select_action(context):
    """
    context: feature 벡터 (list 또는 array)
    각 액션별로 bootstrap 샘플을 추출하여 결정트리 회귀모델을 학습한 후,
    현재 context에 대한 예측 보상을 산출하고 가장 높은 액션을 선택합니다.
    """
    predicted_rewards = {}
    for a in ACTIONS:
        data = tree_data[a]
        if len(data["X"]) < 5:
            predicted_rewards[a] = 0.5  # 충분한 데이터가 없으면 기본값
        else:
            n = len(data["X"])
            indices = [np.random.randint(0, n) for _ in range(n)]
            X_sample = [data["X"][i] for i in indices]
            y_sample = [data["y"][i] for i in indices]
            model = DecisionTreeRegressor(random_state=42)
            model.fit(X_sample, y_sample)
            pred = model.predict([context])[0]
            predicted_rewards[a] = pred
    return max(predicted_rewards, key=predicted_rewards.get)

def update_treebootstrap_params(action, context, reward):
    """
    treebootstrap 학습 데이터에 (context, reward) 쌍을 추가합니다.
    """
    tree_data[action]["X"].append(context)
    tree_data[action]["y"].append(reward)
