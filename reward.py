import statistics
from config import PENALTY, OUTSOURCE_FRACTION, RISK_DISCOUNT_FACTOR

# 전역 변수: 지금까지 최종 결정(Reject, Accept, Outsource)으로 산출된 reward들을 기록합니다.
order_reward_history = []

# history가 없을 때 기본값 (튜닝 가능한 기본 파라미터)
baseline_future_expected_revenue = 300
baseline_future_prob = 0.5


def get_dynamic_parameters(o):
    """
    현재까지의 order_reward_history를 기반으로 동적으로
    FUTURE_ORDER_EXPECTED_REVENUE, FUTURE_ORDER_PROB와 각 lambda 계수를 산출합니다.

    - FUTURE_ORDER_EXPECTED_REVENUE: 지금까지의 평균 reward (기본값 baseline에서 시작)
    - FUTURE_ORDER_PROB: 현재 주문 revenue보다 더 높은 reward를 기록한 주문 비율
    - lambda 계수들은 주문 reward의 분산에 따라 약간 조정됩니다.
    """
    if order_reward_history:
        avg_reward = sum(order_reward_history) / len(order_reward_history)
        FUTURE_ORDER_EXPECTED_REVENUE = avg_reward
        # 현재 주문 revenue보다 더 높은 reward를 기록한 주문의 비율로 확률 추정
        prob_higher = sum(1 for r in order_reward_history if r > o.revenue) / len(order_reward_history)
        FUTURE_ORDER_PROB = prob_higher
        # 주문 reward의 분산을 계산 (데이터가 2개 이상일 때)
        if len(order_reward_history) > 1:
            var = statistics.variance(order_reward_history)
        else:
            var = 0
        # lambda 계수: 기본값에 분산의 영향을 소폭 반영 (튜닝 가능)
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


def update_order_reward_history(reward):
    """
    최종적으로 결정된 주문의 reward를 전역 history에 추가합니다.
    """
    order_reward_history.append(reward)


def estimate_reward(o, act, t, available_ratio):
    """
    미래 기대치와 regret(또는 bonus)을 함께 고려하여 주문 o에 대해
    선택한 액션(act)의 reward를 산출합니다.

    매개변수:
      o               : 주문(order) 객체
      act             : 선택된 액션 ("Accept", "Reject", "Postpone", "Outsource")
      t               : 현재 timestep
      available_ratio : 현재 머신 가용률 (0 ~ 1)

    내부적으로, 현재까지의 order_reward_history를 기반으로
    FUTURE_ORDER_EXPECTED_REVENUE, FUTURE_ORDER_PROB, 그리고 각 lambda 계수를 동적으로 산출합니다.

    각 액션별 산출 방식:
      * Accept:
            기본 reward는 주문 revenue에 risk discount를 적용하고, due_date를 고려하여 PENALTY를 차감합니다.
            여기에 기회비용(opportunity_cost)을 빼고, 미래에 더 좋은 order를 놓칠 경우의 regret을 lambda_accept 배 만큼 감가합니다.

      * Outsource:
            기본 reward는 revenue의 일정 비율에 risk discount를 적용한 값에서 기회비용을 뺍니다.
            미래 기대치를 bonus로 lambda_outsource 배 만큼 추가합니다.

      * Reject:
            기본 reward는 revenue의 10%에 해당하는 음수값으로 설정하고,
            미래 기대치에 따른 regret을 lambda_reject 배 만큼 추가(감가)합니다.

      * Postpone:
            기본 reward는 revenue의 5%에 해당하는 음수값으로 설정하고,
            미래 bonus를 lambda_postpone 배 만큼 반영합니다.
    """
    # 기회비용: 현재 머신 가용률에 따른 비용
    opportunity_cost = o.revenue * (1 - available_ratio) * 0.3

    # 주문 o와 현재 history를 기반으로 동적 파라미터 산출
    (FUTURE_ORDER_EXPECTED_REVENUE, FUTURE_ORDER_PROB,
     lambda_accept, lambda_reject, lambda_outsource, lambda_postpone) = get_dynamic_parameters(o)

    future_opportunity = FUTURE_ORDER_PROB * max(0, FUTURE_ORDER_EXPECTED_REVENUE - o.revenue)

    if act == "Accept":
        finish_est = t + o.processing_time
        disc = RISK_DISCOUNT_FACTOR(o.risk)
        if finish_est <= o.due_date:
            base_accept = o.revenue * disc
        else:
            base_accept = o.revenue * disc - PENALTY
        base_accept -= opportunity_cost
        reward_value = base_accept - lambda_accept * future_opportunity

    elif act == "Outsource":
        base_outsource = o.revenue * OUTSOURCE_FRACTION * RISK_DISCOUNT_FACTOR(o.risk) - opportunity_cost
        reward_value = base_outsource + lambda_outsource * future_opportunity

    elif act == "Reject":
        base_reject = -0.1 * o.revenue
        reward_value = base_reject - lambda_reject * future_opportunity

    elif act == "Postpone":
        base_postpone = -0.05 * o.revenue
        reward_value = base_postpone + lambda_postpone * future_opportunity

    else:
        reward_value = 0.0

    # 최종 액션(Reject, Accept, Outsource)인 경우에 한해 history에 업데이트합니다.
    if act in ["Accept", "Outsource", "Reject"]:
        update_order_reward_history(reward_value)

    return reward_value
