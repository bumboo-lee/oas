import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_thompson_mean(th_history):
    """
    시뮬레이션에서 리턴받은 th_history (각 액션별 timesteps, mean)를 사용
    """
    plt.figure(figsize=(12, 8))
    for a, hist in th_history.items():
        plt.plot(hist["timesteps"], hist["mean"], label=f"{a} mean")
    plt.xlabel("Timestep")
    plt.ylabel("Thompson Sampling Mean")
    plt.title("Thompson Sampling Mean Across Timesteps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_gantt(orders, num_timesteps=100, title="Gantt Chart"):
    color_map = {
        "Postpone":   "orange",
        "Accepted":   "green",
        "Outsourced": "purple",
        "Rejected":   "red",
    }

    fig, ax = plt.subplots(figsize=(15, 7))
    all_no = sorted(o.order_no for o in orders)
    y_dict = {n: i for i, n in enumerate(all_no)}

    for o in orders:
        y_pos = y_dict[o.order_no]
        if o.final_action == "Accept" and o.start_time is not None and o.finish_time is not None:
            st, fn = o.start_time, o.finish_time
            ax.barh(y_pos, fn - st, left=st, height=0.6,
                    color=color_map["Accepted"], edgecolor='black')
        elif o.final_action == "Reject":
            # Reject 시점
            rej_t = None
            for (tt, ac) in o.decision_history:
                if ac == "Reject":
                    rej_t = tt
                    break
            if rej_t is not None and rej_t <= num_timesteps:
                ax.barh(y_pos, 1, left=rej_t, height=0.6,
                        color=color_map["Rejected"], edgecolor='black')
        elif o.final_action == "Outsource":
            out_t = None
            for (tt, ac) in o.decision_history:
                if ac == "Outsource":
                    out_t = tt
                    break
            if out_t is not None and out_t <= num_timesteps:
                ax.barh(y_pos, 1, left=out_t, height=0.6,
                        color=color_map["Outsourced"], edgecolor='black')

        # Postpone 표시
        if o.final_action in ["Accept","Reject","Outsource"]:
            final_t = None
            for (tt, ac) in sorted(o.decision_history, key=lambda x:x[0]):
                if ac == o.final_action:
                    final_t = tt
                    break
            if final_t is None:
                final_t = num_timesteps+1
        else:
            final_t = num_timesteps+1

        in_postpone = False
        seg_start = None
        for ts in range(o.order_date, min(final_t, num_timesteps+1)):
            last_a = None
            for (tt,ac) in o.decision_history:
                if tt <= ts:
                    last_a = ac
                else:
                    break
            if last_a == "Postpone":
                if not in_postpone:
                    in_postpone = True
                    seg_start = ts
            else:
                if in_postpone:
                    ax.barh(y_pos, ts-seg_start, left=seg_start, height=0.6,
                            color=color_map["Postpone"], edgecolor='black')
                    in_postpone = False

        # if in_postpone ...
        if in_postpone and seg_start is not None:
            ax.barh(y_pos, final_t-seg_start, left=seg_start, height=0.6,
                    color=color_map["Postpone"], edgecolor='black')

        # 마커
        ax.plot(o.order_date, y_pos, marker='o', color='black', markersize=5)
        ax.plot(o.decision_due_date, y_pos, marker='D', color='blue', markersize=5)
        ax.plot(o.due_date, y_pos, marker='|', color='red', markersize=10)

    ax.set_xlim(0, num_timesteps+1)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Order No.")
    ax.set_yticks(range(len(all_no)))
    ax.set_yticklabels(all_no)
    ax.set_title(title)

    import matplotlib.lines as mlines
    legend_patches = [
        mpatches.Patch(color=color_map["Postpone"], label="Postpone"),
        mpatches.Patch(color=color_map["Accepted"], label="Accepted"),
        mpatches.Patch(color=color_map["Outsourced"], label="Outsourced"),
        mpatches.Patch(color=color_map["Rejected"], label="Rejected"),
    ]
    od_marker = mlines.Line2D([], [], marker='o', color='black', linestyle='None',
                              markersize=5, label='OrderDate')
    ddd_marker = mlines.Line2D([], [], marker='D', color='blue', linestyle='None',
                               markersize=5, label='DecisionDueDate')
    due_marker = mlines.Line2D([], [], marker='|', color='red', linestyle='None',
                               markersize=10, label='DueDate')

    ax.legend(handles=legend_patches + [od_marker, ddd_marker, due_marker],
              bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

