import random
random.seed(42)

import numpy as np
from config import NUM_TIMESTEPS

model_info = {
    "15-EB11": (8, 250, 0),
    "25-EX20": (9, 300, 61.55),
    "30-EX20": (11, 200, 3.49),
    "20-EB20": (15, 220, 0),
    "35-EB10": (5, 180, 10.0),
    "40-EX10": (10, 280, 20.0),
    "45-EB30": (12, 320, 30.0),
    "50-EX30": (9, 350, 40.0),
    "55-EB40": (14, 400, 50.0),
    "60-EX40": (7, 500, 70.0),
}

initial_orders_data = [
    [1, 0, 5, "15-EB11", 8, 12, 250, 0],
    [2, 10, 15, "25-EX20", 7, 25, 300, 61.55],
    [3, 10, 15, "25-EX20", 9, 25, 300, 61.55],
    [4, 10, 15, "25-EX20", 9, 25, 300, 61.55],
    [5, 23, 30, "30-EX20", 11, 32, 200, 3.49],
    [6, 26, 29, "20-EB20", 15, 38, 220, 0],
    [7, 30, 35, "15-EB11", 8, 45, 250, 0],
]


def generate_orders():

    all_orders_data = []
    all_orders_data.extend(initial_orders_data)

    tmp_no = 100
    random_orders_data = []

    # 각 timestep마다 주문이 들어올 확률을 20%로 설정하고, 들어오면 1~5개의 주문이 동시에 도착하도록 함.
    for t in range(NUM_TIMESTEPS + 1):
        if random.random() < 0.2:
            new_count = random.randint(1, 5)
        else:
            new_count = 0

        for _ in range(new_count):
            tmp_no += 1
            m_name = random.choice(list(model_info.keys()))
            p_time, rev, rsk = model_info[m_name]
            dec_due = t + random.randint(3, 10)
            dd = t + random.randint(p_time + 2, p_time + 15)
            random_orders_data.append([
                tmp_no,
                t,
                dec_due,
                m_name,
                p_time,
                dd,
                rev,
                rsk
            ])
    all_orders_data.extend(random_orders_data)

    # OrderDate 기준 정렬 (row[1] = order_date)
    all_orders_data.sort(key=lambda row: row[1])

    # OrderNo 재할당
    for i, row in enumerate(all_orders_data, start=1):
        row[0] = i

    return all_orders_data
