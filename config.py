# config.py
NUM_TIMESTEPS = 200
MACHINE_CAPACITY = 3
OUTSOURCE_FRACTION = 0.6
PENALTY = 30

# claim 처리 비용 (예: 클레임 발생 시 차감할 비용)
CLAIM_PROCESSING_COST = 150

# 각 model_name별 초기 claim 발생 확률 (0~1)
# 기존 risk 값 대신 이 확률을 사용하여 claim 이벤트를 시뮬레이션
CLAIM_PROB_PER_MODEL = {
    "15-EB11": 0.05,
    "25-EX20": 0.10,
    "30-EX20": 0.08,
    "20-EB20": 0.13,
    "35-EB10": 0.16,
    "40-EX10": 0.07,
    "45-EB30": 0.09,
    "50-EX30": 0.10,
    "55-EB40": 0.12,
    "60-EX40": 0.15,
}
