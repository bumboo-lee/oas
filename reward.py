import statistics
from config import PENALTY, OUTSOURCE_FRACTION, CLAIM_PROCESSING_COST

# 전역 변수: 지금까지 최종 결정된 주문의 reward를 기록합니다.
order_reward_history = []

# 모델별 주문 수와 claim 발생 건수를 기록하는 전역 딕셔너리
model_order_count = {}  # key: model_name, value: 주문 건수
model_claim_count = {}  # key: model_name, value: claim 발생 건수

# history가 없을 때의 기본값 (튜닝 가능한 초기 값)
baseline_future_expected_revenue = 300
baseline_future_prob = 0.5


def get_dynamic_parameters(o):
    """
    현재까지의 order_reward_history를 기반으로 동적으로
    FUTURE_ORDER_EXPECTED_REVENUE, FUTURE_ORDER_PROB와 각 lambda 계수를 산출합니다.

    - FUTURE_ORDER_EXPECTED_REVENUE: 지금까지의 평균 reward에서 해당 모델에 대한 claim 비용(평균)을 차감한 값
       (기본값 baseline에서 시작)
    - FUTURE_ORDER_PROB: 현재 주문 revenue보다 더 높은 reward를 기록한 주문 비율
    - lambda 계수들은 주문 reward의 분산에 따라 약간 조정됩니다.
    """
    if order_reward_history:
        avg_reward = sum(order_reward_history) / len(order_reward_history)
        # 해당 주문의 model에 대해, 과거 claim 발생 비율을 반영
        m = o.model_name
        if m in model_order_count and model_order_count[m] > 0:
            claim_ratio = model_claim_count.get(m, 0) / model_order_count[m]
            # 평균 claim penalty: claim_ratio * CLAIM_PROCESSING_COST
            avg_claim_penalty = claim_ratio * CLAIM_PROCESSING_COST
        else:
            avg_claim_penalty = 0
        FUTURE_ORDER_EXPECTED_REVENUE = avg_reward - avg_claim_penalty
        prob_higher = sum(1 for r in order_reward_history if r > o.revenue) / len(order_reward_history)
        FUTURE_ORDER_PROB = prob_higher
        if len(order_reward_history) > 1:
            var = statistics.variance(order_reward_history)
        else:
            var = 0
        lambda_accept = 1.0 + var * 0.001
        lambda_reject = 0.5 + var * 0.001
        lambda_outsource = 0.5 + var * 0.001
        lambda_postpone = 0.7 + var * 0.001
    else:
        FUTURE_ORDER_EXPECTED_REVENUE = baseline_future_expected_revenue
        FUTURE_ORDER_PROB = baseline_future_prob
        lambda_accept = 1.0
        lambda_reject = 0.5
        lambda_outsource = 0.5
        lambda_postpone = 0.7
    return (FUTURE_ORDER_EXPECTED_REVENUE, FUTURE_ORDER_PROB,
            lambda_accept, lambda_reject, lambda_outsource, lambda_postpone)


def update_order_reward_history(reward, model_name):
    """
    최종적으로 결정된 주문의 reward를 전역 history에 추가하고,
    해당 주문의 model_name에 대해 주문 건수를 업데이트합니다.
    """
    order_reward_history.append(reward)
    if model_name in model_order_count:
        model_order_count[model_name] += 1
    else:
        model_order_count[model_name] = 1


def update_model_claim_count(model_name):
    """
    해당 모델에 대해 claim 발생 건수를 업데이트합니다.
    """
    if model_name in model_claim_count:
        model_claim_count[model_name] += 1
    else:
        model_claim_count[model_name] = 1


def estimate_reward(o, act, t, available_ratio):
    """
    미래 기대치와 regret(또는 bonus)을 함께 고려하여 주문 o에 대해
    선택한 액션(act)의 reward를 산출합니다.

    매개변수:
      o               : 주문(order) 객체 (o.model_name 포함)
      act             : 선택된 액션 ("Accept", "Reject", "Postpone", "Outsource")
      t               : 현재 timestep
      available_ratio : 현재 머신 가용률 (0 ~ 1)

    내부적으로, 현재까지의 order_reward_history를 기반으로
    FUTURE_ORDER_EXPECTED_REVENUE, FUTURE_ORDER_PROB, 그리고 각 lambda 계수를 동적으로 산출합니다.

    각 액션별 산출 방식:
      * Accept:
            - 주문 revenue에 due_date 조건을 반영하여 PENALTY를 차감한 기본 reward를 계산.
            - 기회비용(opportunity_cost)을 차감.
            - 미래에 더 좋은 order를 놓칠 경우의 regret을 lambda_accept 배 만큼 감가.

      * Outsource:
            - 주문 revenue의 일정 비율(OUTSOURCE_FRACTION)을 기본 reward로 사용하고,
            - 기회비용을 차감한 후, 미래 기대치를 bonus로 lambda_outsource 배 만큼 추가.

      * Reject:
            - 주문 revenue의 10%에 해당하는 음수값을 기본 reward로 설정하고,
            - 미래 기대치에 따른 regret을 lambda_reject 배 만큼 추가(감가).

      * Postpone:
            - 주문 revenue의 5%에 해당하는 음수값을 기본 reward로 설정하고,
            - 미래 bonus를 lambda_postpone 배 만큼 반영.
    """
    opportunity_cost = o.revenue * (1 - available_ratio) * 0.3

    (FUTURE_ORDER_EXPECTED_REVENUE, FUTURE_ORDER_PROB,
     lambda_accept, lambda_reject, lambda_outsource, lambda_postpone) = get_dynamic_parameters(o)

    future_opportunity = FUTURE_ORDER_PROB * max(0, FUTURE_ORDER_EXPECTED_REVENUE - o.revenue)

    if act == "Accept":
        finish_est = t + o.processing_time
        if finish_est <= o.due_date:
            base_accept = o.revenue
        else:
            base_accept = o.revenue - PENALTY
        base_accept -= opportunity_cost
        reward_value = base_accept - lambda_accept * future_opportunity

    elif act == "Outsource":
        base_outsource = o.revenue * OUTSOURCE_FRACTION - opportunity_cost
        reward_value = base_outsource + lambda_outsource * future_opportunity

    elif act == "Reject":
        base_reject = -0.1 * o.revenue
        reward_value = base_reject - lambda_reject * future_opportunity

    elif act == "Postpone":
        base_postpone = -0.05 * o.revenue
        reward_value = base_postpone + lambda_postpone * future_opportunity

    else:
        reward_value = 0.0

    # Accept, Outsource, Reject 주문의 reward를 기록하고, 해당 모델의 주문 건수 업데이트
    if act in ["Accept", "Outsource", "Reject"]:
        update_order_reward_history(reward_value, o.model_name)

    return reward_value