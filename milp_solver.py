import pulp
import pandas as pd
import random

from data_generation import generate_orders
from order_class import Order
from config import NUM_TIMESTEPS, MACHINE_CAPACITY, PENALTY, OUTSOURCE_FRACTION, CLAIM_PROCESSING_COST, \
    CLAIM_PROB_PER_MODEL
from reward import estimate_reward
from plot_result import plot_gantt


def solve_milp():
    orders_data = generate_orders()
    orders = [Order(*row) for row in orders_data]
    num_orders = len(orders)
    T = NUM_TIMESTEPS
    ACTIONS = ["Accept", "Outsource", "Reject", "Postpone"]

    prob = pulp.LpProblem("Order_Scheduling", pulp.LpMaximize)

    # 주문별 액션 결정 변수 생성
    x = {}
    for i, o in enumerate(orders):
        for a in ACTIONS:
            x[(i, a)] = pulp.LpVariable(f"x_{i}_{a}", cat="Binary")
        prob += pulp.lpSum([x[(i, a)] for a in ACTIONS]) == 1, f"OneAction_order_{i}"
        # decision due date가 현재 시뮬레이션 내에 도달했다면 "Postpone" 옵션 배제
        if o.decision_due_date <= T:
            prob += x[(i, "Postpone")] == 0, f"DisallowPostpone_order_{i}"

    # Accept 선택 시 스케줄링 변수 생성
    y = {}
    feasible_times = {}
    for i, o in enumerate(orders):
        start_min = o.order_date
        start_max = T - o.processing_time
        feasible_times[i] = list(range(start_min, start_max + 1)) if start_max >= start_min else []
        for t in feasible_times[i]:
            y[(i, t)] = pulp.LpVariable(f"y_{i}_{t}", cat="Binary")
        if feasible_times[i]:
            prob += pulp.lpSum([y[(i, t)] for t in feasible_times[i]]) == x[(i, "Accept")], f"Link_order_{i}"
        else:
            prob += x[(i, "Accept")] == 0, f"NoAccept_possible_{i}"

    # Revenue 파라미터 (risk 할인 없이)
    r_accept = {}
    for i, o in enumerate(orders):
        for t in feasible_times[i]:
            finish = t + o.processing_time
            r_accept[(i, t)] = o.revenue if finish <= o.due_date else o.revenue - PENALTY

    r_outsource, r_reject, r_postpone = {}, {}, {}
    for i, o in enumerate(orders):
        r_outsource[i] = o.revenue * OUTSOURCE_FRACTION
        r_reject[i] = -0.1 * o.revenue
        r_postpone[i] = -0.05 * o.revenue

    objective = []
    for i in range(num_orders):
        if feasible_times[i]:
            objective.append(pulp.lpSum([r_accept[(i, t)] * y[(i, t)] for t in feasible_times[i]]))
        objective.append(r_outsource[i] * x[(i, "Outsource")])
        objective.append(r_reject[i] * x[(i, "Reject")])
        objective.append(r_postpone[i] * x[(i, "Postpone")])
    prob += pulp.lpSum(objective), "Total_Revenue"

    for tau in range(T):
        capacity_expr = []
        for i, o in enumerate(orders):
            for t in feasible_times[i]:
                if t <= tau < t + o.processing_time:
                    capacity_expr.append(y[(i, t)])
        prob += pulp.lpSum(capacity_expr) <= MACHINE_CAPACITY, f"Capacity_time_{tau}"

    solver = pulp.PULP_CBC_CMD(msg=1)
    prob.solve(solver)

    print("MILP Status:", pulp.LpStatus[prob.status])
    optimal_revenue = pulp.value(prob.objective)
    print("Optimal Total Revenue (pre-claim adjustment):", optimal_revenue)

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
        o.final_action = chosen_action
        if chosen_action == "Accept" and chosen_start is not None:
            o.start_time = chosen_start
            o.finish_time = chosen_start + o.processing_time
            calc_revenue = r_accept[(i, chosen_start)]
        elif chosen_action in ["Outsource", "Reject", "Postpone"]:
            # 최소 1 타임스텝 바를 그리도록 order_date를 기준으로 설정
            o.start_time = o.order_date
            o.finish_time = o.order_date + 1
            if chosen_action == "Outsource":
                calc_revenue = r_outsource[i]
            elif chosen_action == "Reject":
                calc_revenue = r_reject[i]
            elif chosen_action == "Postpone":
                calc_revenue = r_postpone[i]
        else:
            chosen_action = "NotScheduled"
            o.start_time = None
            o.finish_time = None
            calc_revenue = 0

        record = {
            "OrderNo": o.order_no,
            "OrderDate": o.order_date,
            "DecisionDueDate": o.decision_due_date,
            "ModelName": o.model_name,
            "ProcessingTime": o.processing_time,
            "DueDate": o.due_date,
            "Revenue": o.revenue,
            "FinalAction": chosen_action,
            "StartTime": o.start_time,
            "FinishTime": o.finish_time,
            "CalculatedRevenue": calc_revenue,
            "ClaimOccurred": "N/A",
            "Claim": "N/A",
            "Cause": "N/A",
            "Position": "N/A"
        }
        schedule_records.append(record)

    # MILP 과정 중 Claim 후처리: Accept와 Outsource 주문에 대해 실시간 처리
    total_claim_cost = 0.0
    from gpt_model.models import generate_claim, analysis_claim
    for o in orders:
        if o.final_action in ["Accept", "Outsource"]:
            claim_prob = CLAIM_PROB_PER_MODEL.get(o.model_name, 0.0)
            if random.random() < claim_prob:
                total_claim_cost += CLAIM_PROCESSING_COST
                CLAIM_PROB_PER_MODEL[o.model_name] = min(1.0, claim_prob + 0.01)
                o.claim_occurred = True
                prompt_gen = [{"role": "user",
                               "content": "A claim Occurs. Generate a claim randomly. You must answer only about the claim. Exclude everything else from your response."}]
                claim_text = generate_claim(prompt_gen)
                prompt_analysis = [{"role": "user",
                                    "content": f'You must respond strictly in the following format. Exclude everything else from your response:{{"Position": "your answer about defect position", "Cause": "your answer about defect cause"}}\n"Claim: {claim_text}'}]
                analysis_text = analysis_claim(prompt_analysis)
                if ";" in analysis_text:
                    parts = analysis_text.split(";")
                    cause = parts[0].replace("Cause:", "").strip() if "Cause:" in parts[0] else parts[0].strip()
                    position = parts[1].replace("Position:", "").strip() if len(parts) > 1 and "Position:" in parts[
                        1] else parts[1].strip()
                else:
                    cause = analysis_text
                    position = ""
                o.claim = claim_text
                o.cause = cause
                o.position = position
            else:
                CLAIM_PROB_PER_MODEL[o.model_name] = max(0.0, claim_prob - 0.005)
                o.claim_occurred = False
                o.claim = "N/A"
                o.cause = "N/A"
                o.position = "N/A"
        else:
            o.claim_occurred = False
            o.claim = "N/A"
            o.cause = "N/A"
            o.position = "N/A"

    print("Total Claim Cost (MILP):", total_claim_cost)
    adjusted_revenue = optimal_revenue - total_claim_cost
    print("Optimal Total Revenue (after claim adjustment) [MILP]:", adjusted_revenue)

    for record, o in zip(schedule_records, orders):
        if o.final_action in ["Accept", "Outsource"]:
            record["ClaimOccurred"] = "Yes" if o.claim_occurred else "No"
        else:
            record["ClaimOccurred"] = "N/A"
        record["Claim"] = o.claim
        record["Cause"] = o.cause
        record["Position"] = o.position

    df = pd.DataFrame(schedule_records)
    df.to_csv("order_data_milp.csv", index=False)
    print("order_data_milp.csv 파일이 저장되었습니다.")
    plot_gantt(orders, NUM_TIMESTEPS, title="MILP Policy - Gantt")


def run_milp():
    solve_milp()

