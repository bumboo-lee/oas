import statistics
from config import CLAIM_PROCESSING_COST, OUTSOURCE_FRACTION, PENALTY
from data_generation import model_info

# 전역 변수: 지금까지의 주문 reward를 기록 (기존)
order_reward_history = []

# 새로운 전역 변수: 실제 주문의 revenue를 기록 (평균 revenue 계산용)
order_revenue_list = []

# 모델별 주문 수와 claim 발생 건수를 기록하는 전역 딕셔너리
model_order_count = {}  # key: model_name, value: 주문 건수
model_claim_count = {}  # key: model_name, value: claim 발생 건수

# data_generation.py의 model_info를 기반으로 초기 평균 revenue 계산
initial_revenues = [info[1] for info in model_info.values()]
baseline_average_revenue = sum(initial_revenues) / len(initial_revenues)


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


def update_order_revenue_history(o):
    """
    주문 o의 revenue를 기록합니다.
    Accept, Outsource, Reject 등에서 처리된 주문의 revenue를 저장합니다.
    """
    order_revenue_list.append(o.revenue)


def update_model_claim_count(model_name):
    """
    해당 모델에 대해 claim 발생 건수를 업데이트합니다.
    """
    if model_name in model_claim_count:
        model_claim_count[model_name] += 1
    else:
        model_claim_count[model_name] = 1


def get_average_revenue():
    """
    지금까지의 한 order당 평균 revenue를 계산합니다.
    만약 기록이 없다면 data_generation.py의 model_info 기반 초기값(baseline_average_revenue)을 반환합니다.
    """
    if order_revenue_list:
        return sum(order_revenue_list) / len(order_revenue_list)
    else:
        return baseline_average_revenue


def estimate_reward(o, act, t, available_ratio,
                    next_order_revenue=None, last_reject_order_revenue=None):
    """
    주문 o에 대해 선택한 액션(act)의 reward를 산출하는 함수입니다.

    [Reward 로직]
      - Accept: max(0, 현재 order의 revenue - average_revenue)
      - Outsource: max(0, 현재 order의 revenue * OUTSOURCE_FRACTION - average_revenue)
      - Postpone: max(0, average_revenue - 현재 order의 revenue)
      - Reject: max(0, (reject 이후 바로 다음 order의 revenue) - (reject한 order의 revenue))
         → next_order_revenue와 last_reject_order_revenue 인자를 활용.

    [Regret 로직]
      - Accept: 만약 Accept한 order에서 claim이 발생하면, CLAIM_PROCESSING_COST 만큼 차감.
      - Outsource: 동일하게, claim 발생 시 CLAIM_PROCESSING_COST 만큼 차감.
      - Postpone: 만약 해당 order가 Accept된 order에 연결되어 있고, postpone으로 인해
                   t + processing_time이 due_date를 초과하여 penalty가 발생했다면,
                   그 penalty (PENALTY * ((t + o.processing_time) - o.due_date)) 만큼 차감.
      - Reject: max(0, 가장 최근에 reject한 order의 revenue - 현재 새로 들어온 order들 중 최대 revenue)

    최종 reward = reward_value - regret_value.

    매개변수:
      o                     : 주문(order) 객체 (o.revenue, o.processing_time, o.due_date 등 포함)
      act                   : 선택한 액션 ("Accept", "Outsource", "Postpone", "Reject")
      t                     : 현재 timestep
      available_ratio       : 현재 머신 가용률 (0~1; 본 로직에서는 직접 사용하지 않음)
      next_order_revenue    : (선택적) Reject 액션 계산 시, reject 이후 바로 다음 order의 revenue
      last_reject_order_revenue : (선택적) Reject 액션 계산 시, 가장 최근에 reject한 order의 revenue
    """

    average_revenue = get_average_revenue()

    reward_value = 0
    regret_value = 0

    if act == "Accept":
        reward_value = max(0, o.revenue - average_revenue)
        if getattr(o, "claim_occurred", False):
            regret_value = CLAIM_PROCESSING_COST
        # 주문 revenue를 기록 (평균 계산용)
        update_order_revenue_history(o)

    elif act == "Outsource":
        reward_value = max(0, o.revenue * OUTSOURCE_FRACTION - average_revenue)
        if getattr(o, "claim_occurred", False):
            regret_value = CLAIM_PROCESSING_COST
        update_order_revenue_history(o)

    elif act == "Postpone":
        reward_value = max(0, average_revenue - o.revenue)
        finish_est = t + o.processing_time
        if finish_est > o.due_date:
            regret_value = PENALTY * (finish_est - o.due_date)
        else:
            regret_value = 0

    elif act == "Reject":
        if next_order_revenue is not None and last_reject_order_revenue is not None:
            reward_value = max(0, next_order_revenue - last_reject_order_revenue)
            regret_value = max(0, last_reject_order_revenue - next_order_revenue)
        else:
            reward_value = 0
            regret_value = 0
        update_order_revenue_history(o)

    else:
        reward_value = 0
        regret_value = 0

    total_reward = reward_value - regret_value

    if act in ["Accept", "Outsource", "Reject"]:
        update_order_reward_history(total_reward, o.model_name)

    return total_reward
