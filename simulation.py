import random
from config import NUM_TIMESTEPS, MACHINE_CAPACITY
from thompson_sampling import (
    ACTIONS, action_params, thompson_sampling_select_action, update_thompson_params,
    treebootstrap_select_action, update_treebootstrap_params, tree_data
)
from reward import estimate_reward

def simulate(orders, num_timesteps=NUM_TIMESTEPS, random_policy=False, policy="contextual"):
    """
    시뮬레이션 실행
    return: (timestep_logs, th_history)
        - timestep_logs: timestep별 로그
        - th_history: {action: {"timesteps": [...], "mean": [...]}, ...}
    """
    local_thompson_history = {a: {"timesteps": [], "mean": []} for a in ACTIONS}

    machine_status = {}
    orders_by_ts = {}
    for o in orders:
        orders_by_ts.setdefault(o.order_date, []).append(o)

    total_orders_arrived = 0
    timestep_logs = []

    for t in range(num_timesteps + 1):
        if t in orders_by_ts:
            for o in orders_by_ts[t]:
                o.decision_history.append((t, "Arrived"))
                total_orders_arrived += 1

        # 생산 중인 주문 진행
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

        for o in decision_needed:
            if t >= o.decision_due_date:
                possible_acts = ["Accept", "Reject", "Outsource"]
            else:
                possible_acts = ACTIONS[:]

            if random_policy:
                act = random.choice(possible_acts)
            else:
                if policy == "contextual":
                    act = thompson_sampling_select_action()
                    if act not in possible_acts:
                        act = random.choice(possible_acts)
                elif policy == "treebootstrap":
                    context = [t, available_ratio, o.order_date, o.decision_due_date,
                               o.processing_time, o.due_date, o.revenue, o.risk]
                    act = treebootstrap_select_action(context)
                    if act not in possible_acts:
                        act = random.choice(possible_acts)
                else:
                    act = random.choice(possible_acts)

            # 머신 용량 체크
            if act == "Accept" and len(machine_status) >= MACHINE_CAPACITY:
                if t >= o.decision_due_date:
                    act = "Reject"
                else:
                    act = "Postpone"

            o.decision_history.append((t, act))

            r_est = estimate_reward(o, act, t, available_ratio)
            if not random_policy:
                if policy == "contextual":
                    update_thompson_params(act, r_est)
                elif policy == "treebootstrap":
                    update_treebootstrap_params(act, context, r_est)

            if act != "Postpone":
                o.final_action = act
                if act == "Accept":
                    o.start_time = t
                    machine_status[o.order_no] = o.processing_time
                else:
                    o.is_completed = True

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

    # 남은 Accept 주문의 finish_time 처리
    for o in orders:
        if o.final_action == "Accept" and not o.is_completed:
            o.finish_time = num_timesteps
            o.is_completed = True

    return timestep_logs, local_thompson_history
