from config import PENALTY, OUTSOURCE_FRACTION, RISK_DISCOUNT_FACTOR

def estimate_reward(o, act, t, available_ratio, model_ratio):
    opportunity_cost = o.revenue * (1 - available_ratio) * 0.3

    if act == "Accept":
        finish_est = t + o.processing_time
        disc = RISK_DISCOUNT_FACTOR(o.risk)
        if finish_est <= o.due_date:
            base_reward = o.revenue * disc
        else:
            base_reward = o.revenue * disc - PENALTY
        return base_reward - opportunity_cost

    elif act == "Outsource":
        base_reward = o.revenue * OUTSOURCE_FRACTION * RISK_DISCOUNT_FACTOR(o.risk)
        return base_reward - opportunity_cost

    elif act in ["Reject", "Postpone"]:
        base_ratio = 0.15 if act == "Reject" else 0.10
        machine_multiplier = 1 + (1 - available_ratio)
        model_multiplier = 1 + (1 - model_ratio)
        future_multiplier = base_ratio * machine_multiplier * model_multiplier
        return o.revenue * future_multiplier

    else:
        return 0.0
