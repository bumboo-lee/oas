import random
from config import NUM_TIMESTEPS, MACHINE_CAPACITY
from thompson_sampling import ACTIONS, action_params, thompson_sampling_select_action, update_thompson_params
from reward import estimate_reward

def simulate(orders, num_timesteps=NUM_TIMESTEPS, random_policy=False):
    """
    시뮬레이션 실행
    return: (timestep_logs, th_history)
        - timestep_logs: timestep별 로그
        - th_history: {action: {"timesteps": [...], "mean": [...]}, ...}
    """
    # 시뮬레이션마다 action_params를 재초기화 하고 싶다면, 여기서 별도 처리를 해야 함
    # 일단 여기서는 기존전역 action_params 유지한다고 가정

    # 시뮬레이션에서 각 액션 mean 변화를 기록할 로컬 딕셔너리
    local_thompson_history = {a: {"timesteps": [], "mean": []} for a in ACTIONS}

    machine_status = {}
    orders_by_ts = {}
    for o in orders:
        orders_by_ts.setdefault(o.order_date, []).append(o)

    model_counts = {}
    total_orders_arrived = 0

    timestep_logs = []

    for t in range(num_timesteps+1):
        if t in orders_by_ts:
            for o in orders_by_ts[t]:
                o.decision_history.append((t, "Arrived"))
                total_orders_arrived += 1
                model_counts[o.model_name] = model_counts.get(o.model_name, 0) + 1

        # 생산 중인 주문 진행
        finished = []
        for onum in list(machine_status.keys()):
            machine_status[onum] -= 1
            if machine_status[onum] <= 0:
                finished.append(onum)
        for fo in finished:
            del machine_status[fo]
            fo_obj = next((x for x in orders if x.order_no==fo), None)
            if fo_obj:
                fo_obj.finish_time = t
                fo_obj.is_completed = True

        # 의사결정
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

        for o in decision_needed:
            # t >= decision_due_date -> Postpone 불가
            if t >= o.decision_due_date:
                possible_acts = ["Accept", "Reject", "Outsource"]
            else:
                possible_acts = ACTIONS[:]

            if random_policy:
                act = random.choice(possible_acts)
            else:
                act = thompson_sampling_select_action()
                if act not in possible_acts:
                    act = random.choice(possible_acts)

            if act == "Accept" and len(machine_status) >= MACHINE_CAPACITY:
                if t >= o.decision_due_date:
                    act = "Reject"
                else:
                    act = "Postpone"

            o.decision_history.append((t, act))

            if total_orders_arrived > 0:
                model_ratio = model_counts.get(o.model_name, 0) / total_orders_arrived
            else:
                model_ratio = 1.0

            # 보상 추정
            r_est = estimate_reward(o, act, t, available_ratio, model_ratio)
            update_thompson_params(act, r_est)

            if act != "Postpone":
                o.final_action = act
                if act == "Accept":
                    o.start_time = t
                    machine_status[o.order_no] = o.processing_time
                else:
                    o.is_completed = True

        # timestep 로그
        # 현재 timestep 끝난 후 action_params의 mean 기록
        for a in ACTIONS:
            local_thompson_history[a]["timesteps"].append(t)
            local_thompson_history[a]["mean"].append(action_params[a]["mean"])

        timestep_logs.append({
            "timestep": t,
            "active_orders": [od.order_no for od in decision_needed],
            "machine_status": dict(machine_status)
        })

    # 생산 중인 주문이 남은 경우 finish_time 설정
    for o in orders:
        if o.final_action == "Accept" and not o.is_completed:
            o.finish_time = num_timesteps
            o.is_completed = True

    return timestep_logs, local_thompson_history
