NUM_TIMESTEPS = 50
MACHINE_CAPACITY = 2
OUTSOURCE_FRACTION = 0.7
PENALTY = 30

def RISK_DISCOUNT_FACTOR(r):
    return max(0.0, 1 - r/100)
