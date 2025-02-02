import pandas as pd
import argparse
import random
import numpy as np

from data_generation import generate_orders
from order_class import Order
from simulation import simulate
from reward import estimate_reward
from config import NUM_TIMESTEPS, PENALTY, OUTSOURCE_FRACTION, RISK_DISCOUNT_FACTOR
from plot_result import plot_thompson_mean, plot_gantt

# 재현성을 위한 seed 설정
random.seed(42)
np.random.seed(42)


def run_policy(policy):
    all_orders_data = generate_orders()
    orders = [Order(*row) for row in all_orders_data]

    if policy == "random":
        print("Random Policy Simulation")
        timestep_data, th_history = simulate(orders, num_timesteps=NUM_TIMESTEPS, random_policy=True)
    elif policy == "contextual":
        print("Contextual Bandit (Thompson Sampling) Policy Simulation")
        timestep_data, th_history = simulate(orders, num_timesteps=NUM_TIMESTEPS, random_policy=False,
                                             policy="contextual")
    elif policy == "treebootstrap":
        print("TreeBootstrap Policy Simulation")
        timestep_data, th_history = simulate(orders, num_timesteps=NUM_TIMESTEPS, random_policy=False,
                                             policy="treebootstrap")
    else:
        raise ValueError(f"알 수 없는 정책입니다: {policy}")

    # 최종 보상 계산 (available_ratio=0으로 단순 계산)
    total_reward = 0.0
    for o in orders:
        if o.final_action:
            total_reward += estimate_reward(o, o.final_action, NUM_TIMESTEPS, 0)

    # 총 얻은 revenue 계산
    total_revenue = 0.0
    for o in orders:
        if o.final_action == "Accept":
            disc = RISK_DISCOUNT_FACTOR(o.risk)
            if o.finish_time is not None and o.finish_time <= o.due_date:
                actual_revenue = o.revenue * disc
            else:
                actual_revenue = o.revenue * disc - PENALTY
            total_revenue += actual_revenue
        elif o.final_action == "Outsource":
            disc = RISK_DISCOUNT_FACTOR(o.risk)
            actual_revenue = o.revenue * OUTSOURCE_FRACTION * disc
            total_revenue += actual_revenue
        # Reject와 Postpone은 실제 revenue 0으로 간주

    print(f"[{policy.capitalize()} Policy] Total Reward = {total_reward}")
    print(f"[{policy.capitalize()} Policy] Total Revenue = {total_revenue}")

    print(f"\n=== {policy.capitalize()} Policy Thompson Sampling / TreeBootstrap Means ===")
    plot_thompson_mean(th_history)

    print(f"\n=== Gantt Chart: {policy.capitalize()} Policy ===")
    plot_gantt(orders, NUM_TIMESTEPS, title=f"{policy.capitalize()} Policy - Gantt")

    # CSV 파일 저장
    order_records = []
    for o in orders:
        rec = {
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
            "DecisionHistory": ";".join([f"{t}:{a}" for t, a in o.decision_history])
        }
        order_records.append(rec)

    df_orders = pd.DataFrame(order_records)
    orders_csv = f"order_data_{policy}.csv"
    df_orders.to_csv(orders_csv, index=False)
    print(f"{orders_csv} 파일이 저장되었습니다.")

    df_timesteps = pd.DataFrame(timestep_data)
    df_timesteps["machine_status"] = df_timesteps["machine_status"].apply(lambda x: str(x))
    timesteps_csv = f"timestep_data_{policy}.csv"
    df_timesteps.to_csv(timesteps_csv, index=False)
    print(f"{timesteps_csv} 파일이 저장되었습니다.")


def main():
    parser = argparse.ArgumentParser(
        description="랜덤, 컨텍스트 밴딧, 트리부트스트랩 중 정책을 선택하여 시뮬레이션을 실행합니다."
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["random", "contextual", "treebootstrap"],
        default="contextual",
        help="실행할 정책: random, contextual, treebootstrap 중 선택"
    )
    args = parser.parse_args()
    run_policy(args.policy)


if __name__ == "__main__":
    main()
