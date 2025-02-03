import argparse
import random
import numpy as np
import pandas as pd
import pulp

from data_generation import generate_orders
from order_class import Order
from config import (
    NUM_TIMESTEPS, MACHINE_CAPACITY, PENALTY, OUTSOURCE_FRACTION,
    CLAIM_PROCESSING_COST, CLAIM_PROB_PER_MODEL
)


# ------------------- MILP 기반 최적 스케줄링 함수 -------------------

def solve_milp():
    # 주문 데이터 생성 (혹은 캐시된 데이터 사용)
    orders_data = generate_orders()
    orders = [Order(*row) for row in orders_data]
    num_orders = len(orders)
    T = NUM_TIMESTEPS
    ACTIONS = ["Accept", "Outsource", "Reject", "Postpone"]

    # MILP 문제 정의 (총 Revenue 최대화)
    prob = pulp.LpProblem("Order_Scheduling", pulp.LpMaximize)

    # (1) 주문 i별 액션 결정 변수: x[(i,a)] ∈ {0,1}
    x = {}
    for i in range(num_orders):
        for a in ACTIONS:
            x[(i, a)] = pulp.LpVariable(f"x_{i}_{a}", cat="Binary")
        prob += pulp.lpSum([x[(i, a)] for a in ACTIONS]) == 1, f"OneAction_order_{i}"

    # (2) Accept 선택 시 스케줄링 변수: y[(i,t)] ∈ {0,1}
    y = {}
    feasible_times = {}
    for i, o in enumerate(orders):
        start_min = o.order_date
        start_max = T - o.processing_time
        if start_max < start_min:
            feasible_times[i] = []
        else:
            feasible_times[i] = list(range(start_min, start_max + 1))
        for t in feasible_times[i]:
            y[(i, t)] = pulp.LpVariable(f"y_{i}_{t}", cat="Binary")
        if feasible_times[i]:
            # 주문 i가 Accept라면 정확히 하나의 시작 시간 선택
            prob += pulp.lpSum([y[(i, t)] for t in feasible_times[i]]) == x[(i, "Accept")], f"Link_order_{i}"
        else:
            prob += x[(i, "Accept")] == 0, f"NoAccept_possible_{i}"

    # Revenue 파라미터 계산 (risk 할인 등 보정 없이)
    # Accept revenue: finish_time이 due_date 이하이면 o.revenue, 초과하면 o.revenue - PENALTY
    r_accept = {}
    for i, o in enumerate(orders):
        for t in feasible_times[i]:
            finish = t + o.processing_time
            if finish <= o.due_date:
                r_accept[(i, t)] = o.revenue
            else:
                r_accept[(i, t)] = o.revenue - PENALTY

    # Outsource, Reject, Postpone의 revenue
    r_outsource = {}
    r_reject = {}
    r_postpone = {}
    for i, o in enumerate(orders):
        r_outsource[i] = o.revenue * OUTSOURCE_FRACTION
        r_reject[i] = -0.1 * o.revenue
        r_postpone[i] = -0.05 * o.revenue

    # 목적 함수: 각 주문의 선택된 액션에 따른 revenue 합 최대화
    objective = []
    for i in range(num_orders):
        if feasible_times[i]:
            objective.append(pulp.lpSum([r_accept[(i, t)] * y[(i, t)] for t in feasible_times[i]]))
        objective.append(r_outsource[i] * x[(i, "Outsource")])
        objective.append(r_reject[i] * x[(i, "Reject")])
        objective.append(r_postpone[i] * x[(i, "Postpone")])
    prob += pulp.lpSum(objective), "Total_Revenue"

    # 머신 용량 제약: 각 시간 tau마다 Accept로 스케줄된 주문 수 ≤ MACHINE_CAPACITY
    for tau in range(T):
        capacity_expr = []
        for i, o in enumerate(orders):
            for t in feasible_times[i]:
                if t <= tau < t + o.processing_time:
                    capacity_expr.append(y[(i, t)])
        prob += pulp.lpSum(capacity_expr) <= MACHINE_CAPACITY, f"Capacity_time_{tau}"

    # MILP 문제 해결 (CBC solver 사용)
    solver = pulp.PULP_CBC_CMD(msg=1)
    prob.solve(solver)

    print("MILP Status:", pulp.LpStatus[prob.status])
    optimal_revenue = pulp.value(prob.objective)
    print("Optimal Total Revenue (pre-claim adjustment):", optimal_revenue)

    # MILP 결과 해석 및 결과 기록 (CSV 저장용)
    schedule_records = []
    for i, o in enumerate(orders):
        chosen_action = None
        chosen_start = None
        for a in ACTIONS:
            if pulp.value(x[(i, a)]) is not None and pulp.value(x[(i, a)]) > 0.5:
                chosen_action = a
                break
        if chosen_action == "Accept" and feasible_times[i]:
            for t in feasible_times[i]:
                if pulp.value(y[(i, t)]) is not None and pulp.value(y[(i, t)]) > 0.5:
                    chosen_start = t
                    break
        # 최종 액션을 주문 객체에 저장
        o.final_action = chosen_action
        if chosen_action == "Accept" and chosen_start is not None:
            finish_time = chosen_start + o.processing_time
            calc_revenue = r_accept[(i, chosen_start)]
            o.start_time = chosen_start
            o.finish_time = finish_time
        elif chosen_action == "Outsource":
            finish_time = None
            calc_revenue = r_outsource[i]
        elif chosen_action == "Reject":
            finish_time = None
            calc_revenue = r_reject[i]
        elif chosen_action == "Postpone":
            finish_time = None
            calc_revenue = r_postpone[i]
        else:
            chosen_action = "NotScheduled"
            finish_time = None
            calc_revenue = 0

        record = {
            "OrderNo": o.order_no,
            "OrderDate": o.order_date,
            "DecisionDueDate": o.decision_due_date,
            "ModelName": o.model_name,
            "ProcessingTime": o.processing_time,
            "DueDate": o.due_date,
            "Revenue": o.revenue,
            "Risk": o.risk,
            "ChosenAction": chosen_action,
            "StartTime": o.start_time,
            "FinishTime": o.finish_time,
            "CalculatedRevenue": calc_revenue,
            "ClaimOccurred": "N/A"
        }
        schedule_records.append(record)

    # ------------------ Claim 후처리 시뮬레이션 (MILP) ------------------
    total_claim_cost = 0.0
    # Accept와 Outsource 주문 모두에 대해 claim 발생 여부 확률적으로 결정
    for o in orders:
        if o.final_action in ["Accept", "Outsource"]:
            claim_prob = CLAIM_PROB_PER_MODEL.get(o.model_name, 0.0)
            if random.random() < claim_prob:
                total_claim_cost += CLAIM_PROCESSING_COST
                CLAIM_PROB_PER_MODEL[o.model_name] = min(1.0, claim_prob + 0.01)
                o.claim_occurred = True
            else:
                CLAIM_PROB_PER_MODEL[o.model_name] = max(0.0, claim_prob - 0.005)
                o.claim_occurred = False
        else:
            o.claim_occurred = False
    print("Total Claim Cost (MILP):", total_claim_cost)
    adjusted_revenue = optimal_revenue - total_claim_cost
    print("Optimal Total Revenue (after claim adjustment) [MILP]:", adjusted_revenue)

    # 각 주문의 기록에 claim 발생 여부 업데이트 ("Yes" 또는 "No" for Accept/Outsource, else "N/A")
    for record, o in zip(schedule_records, orders):
        if o.final_action in ["Accept", "Outsource"]:
            record["ClaimOccurred"] = "Yes" if o.claim_occurred else "No"
        else:
            record["ClaimOccurred"] = "N/A"

    df = pd.DataFrame(schedule_records)
    df.to_csv("order_data_milp.csv", index=False)
    print("order_data_milp.csv 파일이 저장되었습니다.")


def run_milp():
    solve_milp()


# ------------------- Simulation 기반 정책 실행 함수 -------------------

def run_simulation_policy(policy):
    from simulation import simulate
    from data_generation import generate_orders
    from order_class import Order
    from config import NUM_TIMESTEPS, PENALTY, OUTSOURCE_FRACTION, CLAIM_PROCESSING_COST, CLAIM_PROB_PER_MODEL
    from reward import estimate_reward
    from plot_result import plot_thompson_mean, plot_gantt

    all_orders_data = generate_orders()
    orders = [Order(*row) for row in all_orders_data]

    if policy == "random":
        print("랜덤 정책으로 시뮬레이션 실행합니다.")
        timestep_data, th_history, _ = simulate(orders, num_timesteps=NUM_TIMESTEPS, random_policy=True)
    elif policy == "contextual":
        print("컨텍스트 밴딧 (Thompson Sampling) 정책으로 시뮬레이션 실행합니다.")
        timestep_data, th_history, _ = simulate(orders, num_timesteps=NUM_TIMESTEPS, random_policy=False,
                                                policy="contextual")
    elif policy == "treebootstrap":
        print("TreeBootstrap (논문 알고리즘) 정책으로 시뮬레이션 실행합니다.")
        timestep_data, th_history, _ = simulate(orders, num_timesteps=NUM_TIMESTEPS, random_policy=False,
                                                policy="treebootstrap")
    else:
        raise ValueError("Unknown simulation policy: " + policy)

    # Claim 후처리: Accept와 Outsource 주문에 대해 claim 발생 여부 확률적으로 결정
    total_claim_cost = 0.0
    for o in orders:
        if o.final_action in ["Accept", "Outsource"]:
            claim_prob = CLAIM_PROB_PER_MODEL.get(o.model_name, 0.0)
            if random.random() < claim_prob:
                total_claim_cost += CLAIM_PROCESSING_COST
                CLAIM_PROB_PER_MODEL[o.model_name] = min(1.0, claim_prob + 0.01)
                o.claim_occurred = True
            else:
                CLAIM_PROB_PER_MODEL[o.model_name] = max(0.0, claim_prob - 0.005)
                o.claim_occurred = False
        else:
            o.claim_occurred = False

    total_reward = 0.0
    for o in orders:
        if o.final_action:
            total_reward += estimate_reward(o, o.final_action, NUM_TIMESTEPS, 0)

    total_revenue = 0.0
    for o in orders:
        if o.final_action == "Accept":
            if o.finish_time is not None and o.finish_time <= o.due_date:
                actual_revenue = o.revenue
            else:
                actual_revenue = o.revenue - PENALTY
            total_revenue += actual_revenue
        elif o.final_action == "Outsource":
            actual_revenue = o.revenue * OUTSOURCE_FRACTION
            total_revenue += actual_revenue
    total_revenue -= total_claim_cost

    print(f"[{policy.capitalize()} Policy] Total Reward = {total_reward}")
    print(f"[{policy.capitalize()} Policy] Total Revenue (after claim cost) = {total_revenue}")

    print(f"\n=== {policy.capitalize()} Policy Thompson Sampling / TreeBootstrap Means ===")
    plot_thompson_mean(th_history)

    print(f"\n=== Gantt Chart: {policy.capitalize()} Policy ===")
    plot_gantt(orders, NUM_TIMESTEPS, title=f"{policy.capitalize()} Policy - Gantt")

    order_records = []
    for o in orders:
        record = {
            "OrderNo": o.order_no,
            "OrderDate": o.order_date,
            "DecisionDueDate": o.decision_due_date,
            "ModelName": o.model_name,
            "ProcessingTime": o.processing_time,
            "DueDate": o.due_date,
            "Revenue": o.revenue,
            "Risk": o.risk,
            "FinalAction": o.final_action,
            "StartTime": o.start_time,
            "FinishTime": o.finish_time,
            "IsCompleted": o.is_completed,
            "DecisionHistory": ";".join([f"{t}:{a}" for t, a in o.decision_history]),
            "ClaimOccurred": "Yes" if (o.final_action in ["Accept", "Outsource"] and o.claim_occurred) else (
                "No" if o.final_action in ["Accept", "Outsource"] else "N/A")
        }
        order_records.append(record)
    df = pd.DataFrame(order_records)
    orders_csv = f"order_data_{policy}.csv"
    df.to_csv(orders_csv, index=False)
    print(f"{orders_csv} 파일이 저장되었습니다.")


# ------------------- 정책 선택 및 실행 -------------------

def run_policy(policy):
    if policy == "milp":
        run_milp()
    elif policy in ["random", "contextual", "treebootstrap"]:
        run_simulation_policy(policy)
    else:
        raise ValueError("Unknown policy: " + policy)


def main():
    parser = argparse.ArgumentParser(
        description="랜덤, 컨텍스트 밴딧, 트리부트스트랩, MILP 중 정책을 선택하여 시뮬레이션 또는 최적 스케줄링을 실행합니다."
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["random", "contextual", "treebootstrap", "milp"],
        default="contextual",
        help="실행할 정책: random, contextual, treebootstrap, milp 중 선택"
    )
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    run_policy(args.policy)

    # 최종적으로 업데이트된 모델별 Claim 확률 출력
    from config import CLAIM_PROB_PER_MODEL
    print("\n최종 모델별 Claim 발생 확률:")
    for model, prob in CLAIM_PROB_PER_MODEL.items():
        print(f"{model}: {prob:.3f}")


if __name__ == "__main__":
    main()
