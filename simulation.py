import random
from config import NUM_TIMESTEPS, MACHINE_CAPACITY, CLAIM_PROCESSING_COST, CLAIM_PROB_PER_MODEL
from thompson_sampling import ACTIONS, action_params, thompson_sampling_select_action, update_thompson_params, \
    treebootstrap_select_action, update_treebootstrap_params, tree_data
from reward import estimate_reward


# (기타 필요한 모듈 임포트)

def simulate(orders, num_timesteps=NUM_TIMESTEPS, random_policy=False, policy="contextual", claim_callback=None):
    """
    시뮬레이션 실행
    return: (timestep_logs, th_history, total_claim_cost)
           total_claim_cost: 모든 주문에서 실제 claim 발생 시 차감된 총 비용
    """
    local_thompson_history = {a: {"timesteps": [], "mean": []} for a in ACTIONS}

    machine_status = {}
    orders_by_ts = {}
    for o in orders:
        orders_by_ts.setdefault(o.order_date, []).append(o)

    total_orders_arrived = 0
    timestep_logs = []
    # Reject 액션 관련 정보를 위해 last_reject_revenue를 전역(루프 외부) 변수로 초기화
    last_reject_revenue = None

    for t in range(num_timesteps + 1):
        # 신규 주문 도착 처리
        if t in orders_by_ts:
            for o in orders_by_ts[t]:
                o.decision_history.append((t, "Arrived"))
                total_orders_arrived += 1

        # 생산 중인 주문 진행: 남은 처리시간 감소 및 완료 주문 처리
        finished = []
        for onum in list(machine_status.keys()):
            machine_status[onum] -= 1
            if machine_status[onum] <= 0:
                finished.append(onum)
        for fo in finished:
            del machine_status[fo]
            fo_obj = next((x for x in orders if x.order_no == fo), None)
            if fo_obj:
                fo_obj.finish_time = t
                fo_obj.is_completed = True

        # 의사결정이 필요한 주문 선정
        decision_needed = []
        for o in orders:
            if o.is_completed:
                continue
            if o.final_action in ["Accept", "Reject", "Outsource"]:
                continue
            if t < o.order_date:
                continue
            decision_needed.append(o)

        available_ratio = (MACHINE_CAPACITY - len(machine_status)) / MACHINE_CAPACITY

        # t+1 시점에 도착하는 주문들 중 최대 revenue 계산 (없으면 None)
        if (t + 1) in orders_by_ts:
            future_orders = orders_by_ts[t + 1]
            if future_orders:
                max_new_order_revenue = max(o_tmp.revenue for o_tmp in future_orders)
            else:
                max_new_order_revenue = None
        else:
            max_new_order_revenue = None

        for o in decision_needed:
            # 결정 마감일 이전에는 모든 액션이 가능, 그 이후에는 일부 제한
            if t >= o.decision_due_date:
                possible_acts = ["Accept", "Reject", "Outsource"]
            else:
                possible_acts = ACTIONS[:]

            # 정책에 따른 액션 선택
            if random_policy:
                act = random.choice(possible_acts)
            else:
                if policy == "contextual":
                    act = thompson_sampling_select_action()
                    if act not in possible_acts:
                        act = random.choice(possible_acts)
                elif policy == "treebootstrap":
                    context = [t, available_ratio, o.order_date, o.decision_due_date, o.processing_time, o.due_date,
                               o.revenue, o.risk]
                    act = treebootstrap_select_action(context)
                    if act not in possible_acts:
                        act = random.choice(possible_acts)
                else:
                    act = random.choice(possible_acts)

            # 머신 용량 초과시 Accept 대신 다른 액션 선택
            if act == "Accept" and len(machine_status) >= MACHINE_CAPACITY:
                if t >= o.decision_due_date:
                    act = "Reject"
                else:
                    act = "Postpone"

            o.decision_history.append((t, act))

            # estimate_reward 호출 시, 새로 도착한 주문의 최대 revenue와 최근 reject 주문의 revenue를 전달
            reward_est = estimate_reward(o, act, t, available_ratio, max_new_order_revenue, last_reject_revenue)

            # 정책별 파라미터 업데이트
            if not random_policy:
                if policy == "contextual":
                    update_thompson_params(act, reward_est)
                elif policy == "treebootstrap":
                    # context가 필요한 경우
                    context = [t, available_ratio, o.order_date, o.decision_due_date, o.processing_time, o.due_date,
                               o.revenue, o.risk]
                    update_treebootstrap_params(act, context, reward_est)

            # Reject 액션의 경우, 최근 reject 주문의 revenue 업데이트
            if act == "Reject":
                last_reject_revenue = o.revenue

            # 주문의 최종 결정 및 스케줄링
            if act != "Postpone":
                o.final_action = act
                if act == "Accept":
                    o.start_time = t
                    machine_status[o.order_no] = o.processing_time
                else:
                    o.is_completed = True

        # 각 timestep마다 Thompson Sampling 관련 정보 업데이트
        for a in ACTIONS:
            local_thompson_history[a]["timesteps"].append(t)
            if not random_policy:
                if policy == "contextual":
                    local_thompson_history[a]["mean"].append(action_params[a]["mean"])
                elif policy == "treebootstrap":
                    data = tree_data[a]
                    if len(data["y"]) > 0:
                        avg = sum(data["y"]) / len(data["y"])
                        local_thompson_history[a]["mean"].append(avg)
                    else:
                        local_thompson_history[a]["mean"].append(0.5)
            else:
                local_thompson_history[a]["mean"].append(0.5)

        timestep_logs.append({
            "timestep": t,
            "active_orders": [od.order_no for od in decision_needed],
            "machine_status": dict(machine_status)
        })

    # 생산 중인 Accept 주문에 대해 시뮬레이션 종료 후 finish_time 처리
    for o in orders:
        if o.final_action == "Accept" and not o.is_completed:
            o.finish_time = num_timesteps
            o.is_completed = True

    # Claim 후처리: Accept 주문에 대해 확률적으로 claim 처리
    total_claim_cost = 0.0
    for o in orders:
        if o.final_action == "Accept" and o.finish_time is not None:
            claim_prob = CLAIM_PROB_PER_MODEL.get(o.model_name, 0.0)
            if random.random() < claim_prob:
                total_claim_cost += CLAIM_PROCESSING_COST
                CLAIM_PROB_PER_MODEL[o.model_name] = min(1.0, claim_prob + 0.01)
                # claim 처리 관련 추가 로직이 있을 경우 호출
                if claim_callback:
                    claim_callback(o)
            else:
                CLAIM_PROB_PER_MODEL[o.model_name] = max(0.0, claim_prob - 0.005)

    return timestep_logs, local_thompson_history, total_claim_cost
