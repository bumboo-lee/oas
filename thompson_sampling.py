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

# tree_data: 각 액션별로 context와 reward 쌍을 저장하는 딕셔너리.
# 초기에는 fabricated prior 데이터를 추가합니다.
# Fabricated prior: 임의의 context (예, 모든 feature가 0인 벡터)와 함께 1 성공, 1 실패를 추가.
# 이때 context 차원은 실제 사용 context와 동일하게 설정해야 합니다. (예제에서는 8차원으로 가정)
INITIAL_CONTEXT = [0.0] * 8  # 필요에 따라 context 차원을 조정하세요.

tree_data = {a: {"X": [], "y": []} for a in ACTIONS}
for a in ACTIONS:
    tree_data[a]["X"].append(INITIAL_CONTEXT)
    tree_data[a]["y"].append(1.0)  # fabricated success
    tree_data[a]["X"].append(INITIAL_CONTEXT)
    tree_data[a]["y"].append(0.0)  # fabricated failure


def treebootstrap_select_action(context):
    """
    각 액션 a에 대해, 지금까지 저장된 Dt,a (fabricated prior 포함)에서
    부트스트랩 샘플을 생성하고, 이를 기반으로 DecisionTreeRegressor를 학습합니다.
    이후, 현재 context에 대해 예측된 보상을 산출하고, 그 중 가장 높은 액션을 선택합니다.

    이 방식은 논문에서 제시한 TreeBootstrap 알고리즘의 기본 아이디어를 반영하며,
    fabricated prior를 추가하여 초반 관측 부족으로 인한 액션 조기 배제를 완화합니다.
    """
    predicted_rewards = {}
    for a in ACTIONS:
        data = tree_data[a]
        n = len(data["X"])
        # 데이터가 극히 부족하면 기본값 0.5를 사용
        if n < 5:
            predicted_rewards[a] = 0.5
        else:
            # n개의 관측치에서 복원 추출
            indices = [np.random.randint(0, n) for _ in range(n)]
            X_sample = [data["X"][i] for i in indices]
            y_sample = [data["y"][i] for i in indices]
            model = DecisionTreeRegressor(random_state=42)
            model.fit(X_sample, y_sample)
            pred = model.predict([context])[0]
            predicted_rewards[a] = pred
    # 예측 보상이 가장 높은 액션 선택
    return max(predicted_rewards, key=predicted_rewards.get)


def update_treebootstrap_params(action, context, reward):
    """
    관측된 (context, reward) 쌍을 해당 액션의 데이터에 추가합니다.
    이렇게 축적된 데이터는 이후 부트스트랩 샘플 생성에 사용되어,
    액션 선택 시 점진적으로 더 정확한 예측을 할 수 있게 합니다.
    """
    tree_data[action]["X"].append(context)
    tree_data[action]["y"].append(reward)