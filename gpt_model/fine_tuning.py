from openai import OpenAI
import time

client = OpenAI(api_key="Bearer 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")

# 모델 1을 위한 Few-shot 파인튜닝 데이터셋 파일 (예: model1_claim_generation.jsonl)
gen_model_file = "generate.jsonl"

# 파일 업로드 (BOM 없이 UTF-8로 저장되어야 함)
gen_model_file = client.files.create(
    file=open(gen_model_file, "rb"),
    purpose='fine-tune'
)
print("Uploaded file ID (Model 1):", gen_model_file.id)

# 모델 1 파인튜닝 작업 생성 (예: GPT-4o-2024-08-06 기반)
gen_model = client.fine_tuning.jobs.create(
    training_file=gen_model_file.id,
    model="gpt-4o-2024-08-06",
    suffix="generate_claim"
)
print("Model 1 Fine-tune job created. Job ID:", gen_model.id)

# 모델 2 (Claim 분석 모델) 예제:
analysis_model_train_file = "analysis_train.jsonl"
analysis_model_test_file = "analysis_test.jsonl"

analysis_model_train_file = client.files.create(
    file=open(analysis_model_train_file, "rb"),
    purpose='fine-tune'
)
analysis_model_test_file = client.files.create(
    file=open(analysis_model_test_file, "rb"),
    purpose='fine-tune'
)
print("Uploaded file ID (Model 2)_train:", analysis_model_train_file.id)
print("Uploaded file ID (Model 2)_test:", analysis_model_test_file.id)

analysis_model = client.fine_tuning.jobs.create(
    training_file=analysis_model_train_file.id,
    validation_file=analysis_model_test_file.id,
    model="gpt-4o-2024-08-06",
    suffix="analyis_claim"
)
print("Model 2 Fine-tune job created. Job ID:", analysis_model.id)

# ------------------ Fine-tuning 진행 상황 모니터링 ------------------
def check_fine_tune_status(job_id):
    # fine-tuning 작업의 상태를 조회합니다.
    job = client.fine_tuning.jobs.retrieve(job_id)
    return job.status

print("Monitoring fine-tuning progress...")
while True:
    status1 = check_fine_tune_status(gen_model.id)
    status2 = check_fine_tune_status(analysis_model.id)
    print(f"Model 1 Fine-tune Status: {status1}")
    print(f"Model 2 Fine-tune Status: {status2}")
    # 두 작업 모두 succeeded 혹은 failed 상태가 되면 종료
    if status1 in ["succeeded", "failed"] and status2 in ["succeeded", "failed"]:
        break
    time.sleep(60)  # 1분마다 상태 체크

print("Fine-tuning completed for both models.")