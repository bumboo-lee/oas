import argparse
import random
import numpy as np

from milp_solver import run_milp
from data_generation import generate_orders
from order_class import Order
from reward import estimate_reward
from config import NUM_TIMESTEPS, PENALTY, OUTSOURCE_FRACTION, CLAIM_PROCESSING_COST, CLAIM_PROB_PER_MODEL
from plot_result import plot_thompson_mean, plot_gantt
from simulation import simulate


# Claim 생성 및 분석 콜백 (실시간 처리)
def claim_callback(order):
    from gpt_model.models import generate_claim, analysis_claim
    prompt_gen = [{"role": "user",
                   "content": "A claim Occurs. Generate a claim randomly. You must answer only about the claim. Exclude everything else from your response."}]
    claim_text = generate_claim(prompt_gen)
    prompt_analysis = [{"role": "user",
                        "content": f'You must respond strictly in the following format. Exclude everything else from your response:{{"Position": "your answer about defect position", "Cause": "your answer about defect cause"}}\n"Claim: {claim_text}'}]
    analysis_text = analysis_claim(prompt_analysis)
    if ";" in analysis_text:
        parts = analysis_text.split(";")
        cause = parts[0].replace("Cause:", "").strip() if "Cause:" in parts[0] else parts[0].strip()
        position = parts[1].replace("Position:", "").strip() if len(parts) > 1 and "Position:" in parts[1] else parts[
            1].strip()
    else:
        cause = analysis_text
        position = ""
    order.claim = claim_text
    order.cause = cause
    order.position = position
    print(f"Order {order.order_no}: Claim generated and analyzed in simulation.")


def run_simulation_policy(policy):
    all_orders_data = generate_orders()
    orders = [Order(*row) for row in all_orders_data]

    # simulation 함수에 claim_callback 전달하여 실시간 claim 처리
    timestep_data, th_history, _ = simulate(
        orders, num_timesteps=NUM_TIMESTEPS, random_policy=(policy == "random"), policy=policy,
        claim_callback=claim_callback
    )

    total_claim_cost = 0.0
    for o in orders:
        if o.final_action in ["Accept", "Outsource"]:
            if not hasattr(o, "claim"):
                claim_prob = CLAIM_PROB_PER_MODEL.get(o.model_name, 0.0)
                if random.random() < claim_prob:
                    total_claim_cost += CLAIM_PROCESSING_COST
                    CLAIM_PROB_PER_MODEL[o.model_name] = min(1.0, claim_prob + 0.01)
                    o.claim_occurred = True
                    claim_callback(o)
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

    try:
        plot_thompson_mean(th_history)
    except ValueError as e:
        print("plot_thompson_mean skipped due to error:", e)

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
            "FinalAction": o.final_action,
            "StartTime": o.start_time,
            "FinishTime": o.finish_time,
            "CalculatedRevenue": o.revenue if o.final_action == "Accept" and (
                        o.finish_time is not None and o.finish_time <= o.due_date) else (
                o.revenue - PENALTY if o.final_action == "Accept" else (
                    o.revenue * OUTSOURCE_FRACTION if o.final_action == "Outsource" else "N/A")),
            "ClaimOccurred": "Yes" if (
                        o.final_action in ["Accept", "Outsource"] and getattr(o, "claim_occurred", False)) else (
                "No" if o.final_action in ["Accept", "Outsource"] else "N/A"),
            "Claim": o.claim if hasattr(o, "claim") else "N/A",
            "Cause": o.cause if hasattr(o, "cause") else "N/A",
            "Position": o.position if hasattr(o, "position") else "N/A"
        }
        order_records.append(record)

    import pandas as pd
    df = pd.DataFrame(order_records)
    df.to_csv(f"order_data_{policy}.csv", index=False)
    print(f"order_data_{policy}.csv 파일이 저장되었습니다.")


def run_policy(policy):
    if policy == "milp":
        from milp_solver import run_milp
        run_milp()
    elif policy in ["random", "contextual", "treebootstrap"]:
        run_simulation_policy(policy)
    else:
        raise ValueError("Unknown policy: " + policy)


def main():
    parser = argparse.ArgumentParser(
        description="랜덤, 컨텍스트 밴딧, 트리부트스트랩, MILP 중 정책을 선택하여 실행합니다."
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

    from config import CLAIM_PROB_PER_MODEL
    print("\n최종 모델별 Claim 발생 확률:")
    for model, prob in CLAIM_PROB_PER_MODEL.items():
        print(f"{model}: {prob:.3f}")


if __name__ == "__main__":
    main()

