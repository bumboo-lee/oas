import pandas as pd

from data_generation import generate_orders
from order_class import Order
from simulation import simulate
from reward import estimate_reward
from config import NUM_TIMESTEPS
from plot_result import plot_thompson_mean, plot_gantt

def run():
    all_orders_data = generate_orders()

    # Contextual Bandit
    orders_cb = [Order(*row) for row in all_orders_data]
    timestep_data_cb, th_history_cb = simulate(orders_cb, num_timesteps=NUM_TIMESTEPS, random_policy=False)

    # Random
    orders_random = [Order(*row) for row in all_orders_data]
    timestep_data_random, th_history_random = simulate(orders_random, num_timesteps=NUM_TIMESTEPS, random_policy=True)

    # 최종 보상 계산
    cb_total_reward = 0.0
    for o in orders_cb:
        if o.final_action:
            cb_total_reward += estimate_reward(o, o.final_action, NUM_TIMESTEPS, 0, 1)
    random_total_reward = 0.0
    for o in orders_random:
        if o.final_action:
            random_total_reward += estimate_reward(o, o.final_action, NUM_TIMESTEPS, 0, 1)

    print(f"[Contextual Bandit] Total Reward = {cb_total_reward}")
    print(f"[Random Policy]    Total Reward = {random_total_reward}")

    # Thompson Sampling Mean Plot
    print("\n=== Contextual Bandit Thompson Sampling Means ===")
    plot_thompson_mean(th_history_cb)

    print("\n=== Random Policy Thompson Sampling Means ===")
    plot_thompson_mean(th_history_random)

    # Gantt Chart
    print("\n=== Gantt Chart: Contextual Bandit ===")
    plot_gantt(orders_cb, NUM_TIMESTEPS, title="Contextual Bandit - Gantt")

    print("\n=== Gantt Chart: Random Policy ===")
    plot_gantt(orders_random, NUM_TIMESTEPS, title="Random Policy - Gantt")

    # CSV 저장 (Contextual Bandit)
    order_records = []
    for o in orders_cb:
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
    df_orders.to_csv("order_data_cb.csv", index=False)
    print("order_data_cb.csv 파일이 저장되었습니다.")

    df_timesteps = pd.DataFrame(timestep_data_cb)
    df_timesteps["machine_status"] = df_timesteps["machine_status"].apply(lambda x: str(x))
    df_timesteps.to_csv("timestep_data_cb.csv", index=False)
    print("timestep_data_cb.csv 파일이 저장되었습니다.")


if __name__ == "__main__":
    run()
