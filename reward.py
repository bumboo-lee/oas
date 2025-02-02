from config import PENALTY, OUTSOURCE_FRACTION, RISK_DISCOUNT_FACTOR


def estimate_reward(o, act, t, available_ratio):
    """
    미래 기대 가치(또는 regret)를 함께 고려하여 각 액션의 보상을 산출합니다.

    매개변수:
      o               : 주문(order) 객체
      act             : 선택된 액션 ("Accept", "Reject", "Postpone", "Outsource")
      t               : 현재 timestep
      available_ratio : 현재 머신 가용률 (0 ~ 1)

    내부 계산:
      - opportunity_cost: 주문의 revenue에 기반한 기회비용 (현재 머신 가용률 반영)
      - FUTURE_ORDER_EXPECTED_REVENUE: 미래에 도착할 수 있는 평균적인 더 좋은 order의 revenue (예: 300)
      - FUTURE_ORDER_PROB: 미래에 더 좋은 order가 들어올 확률 (예: 0.5)
      - future_opportunity: 미래에 받을 수 있는 추가 수익 기대치
          = FUTURE_ORDER_PROB * max(0, FUTURE_ORDER_EXPECTED_REVENUE - o.revenue)

    각 액션별로:
      * Accept:
           - finish_est (t + processing_time)와 due_date에 따라 revenue 또는 penalty를 적용합니다.
           - 기회비용(opportunity_cost)을 차감한 후, 미래에 더 좋은 order를 놓칠 경우의 regret을 lambda_accept 배 만큼 감가하여 최종 보상을 산출합니다.

      * Outsource:
           - Outsource 시 revenue의 일정 비율(OUTSOURCE_FRACTION)과 risk discount를 적용한 기본 보상에 기회비용을 차감하고,
           - 미래 기대 가치(future_opportunity)를 lambda_outsource 배 만큼 bonus로 추가합니다.

      * Reject:
           - Reject의 즉각 보상은 주문 revenue의 10%에 해당하는 음수값(base_reject)으로 설정하고,
           - 미래에 더 좋은 order에 대한 기대가 실현되지 않았을 경우의 regret을 lambda_reject 배 만큼 감가합니다.

      * Postpone:
           - Postpone의 즉각 보상은 주문 revenue의 5%에 해당하는 음수값(base_postpone)으로 설정하고,
           - 미래 기대 가치가 실현될 경우의 bonus를 lambda_postpone 배 만큼 반영합니다.
    """
    # 기회비용 (현재 머신 가용률에 따른 비용)
    opportunity_cost = o.revenue * (1 - available_ratio) * 0.3

    # 미래 기대치 (더 좋은 order에 대한 기대)
    FUTURE_ORDER_EXPECTED_REVENUE = 300  # 조정 가능한 파라미터
    FUTURE_ORDER_PROB = 0.5  # 조정 가능한 파라미터
    future_opportunity = FUTURE_ORDER_PROB * max(0, FUTURE_ORDER_EXPECTED_REVENUE - o.revenue)

    # 각 액션별 lambda 계수 (튜닝 가능)
    lambda_accept = 1.0
    lambda_reject = 0.5
    lambda_outsource = 0.5
    lambda_postpone = 0.7

    if act == "Accept":
        finish_est = t + o.processing_time
        disc = RISK_DISCOUNT_FACTOR(o.risk)
        if finish_est <= o.due_date:
            base_accept = o.revenue * disc
        else:
            base_accept = o.revenue * disc - PENALTY
        base_accept -= opportunity_cost
        # Accept 시, 미래에 더 좋은 order를 놓칠 경우의 regret 반영
        return base_accept - lambda_accept * future_opportunity

    elif act == "Outsource":
        base_outsource = o.revenue * OUTSOURCE_FRACTION * RISK_DISCOUNT_FACTOR(o.risk) - opportunity_cost
        # Outsource 시, 미래에 더 좋은 order를 받을 기회 bonus 추가
        return base_outsource + lambda_outsource * future_opportunity

    elif act == "Reject":
        base_reject = -0.1 * o.revenue
        # Reject 시, 미래 기대치에 따른 regret 반영
        return base_reject - lambda_reject * future_opportunity

    elif act == "Postpone":
        base_postpone = -0.05 * o.revenue
        # Postpone 시, 미래 bonus 반영
        return base_postpone + lambda_postpone * future_opportunity

    else:
        return 0.0
