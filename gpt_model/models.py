from openai import OpenAI

# API 키 설정
client = OpenAI(api_key="Bearer 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")
# print(client.fine_tuning.jobs.list(limit=2))

# 모델 1: Claim 생성 모델 호출 예제
def generate_claim(message):
    gen_model_id = "ft:gpt-4o-2024-08-06:personal:generate-claim:AxWbiQRk"
    claim = client.chat.completions.create(
        model=gen_model_id,
        messages=message,
        temperature=0.7,
        max_tokens=150
        )

    print("Model 1 Generated Claim:", claim.choices[0].message.content)
    return claim.choices[0].message.content

# 모델 2: Claim 분석 모델 호출 예제
def analysis_claim(message):
    analysis_model_id = "ft:gpt-4o-2024-08-06:personal:analyis-claim:AxWjyNA9"
    analysis = client.chat.completions.create(
        model=analysis_model_id,
        messages=message,
        max_tokens=150,
        temperature=0,
    )
    print("Model 2 Analysis Result:", analysis.choices[0].message.content)
    return analysis.choices[0].message.content


"""
# EXAMPLE
analysis_message = [{"role": "user", "content": "Claim: 10/27 PICKED UP SOLENOID PART AT SHOP AND DRIVING TO CUSTOMER. 10/27 REPLACED FORWARD SOLENOID AND TEST DROVE FOUND UNIT STILL TAKING OFF ON NEUTRAL. UNIT STILL HAVE ISSUES WITH GOING FORWARD WHEN IN NEUTRAL PERFORMED TEST FOUND THAT THE FORWARD RELAY IS"}]
gen_message = [{"role": "user", "content": "A claim Occurs. Generate a claim randomly."}]

generate_claim(gen_message)
analysis_claim(analysis_message)

"""