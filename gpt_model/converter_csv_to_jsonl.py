import pandas as pd
import json
from sklearn.model_selection import train_test_split


def run(job):
    def convert_to_gpt35_format(dataset, job):
        if job == 'analysis':
            fine_tuning_data = []
            for _, row in dataset.iterrows():
                json_response = '{"Cause": "' + row['Cause'] + '", "Position": "' + row['Position'] + '"}'
                fine_tuning_data.append({
                    "messages": [
                        {"role": "user", "content": f'You must respond strictly in the following format. Exclude everything else from your response:{{"Position": "your answer about defect position", "Cause": "your answer about defect cause"}}\n"Claim: {row["Claim"]}'},
                        {"role": "assistant", "content": json_response}
                    ]
                })
            return fine_tuning_data
        elif job == 'generate':
            fine_tuning_data = []
            for _, row in dataset.iterrows():
                fine_tuning_data.append({
                    "messages": [
                        {"role": "user", "content": row['Occur']},
                        {"role": "assistant", "content": row['Claim']}
                    ]
                })
            return fine_tuning_data

    df = pd.read_csv(f'{job}_sample.csv')
    convert_data=convert_to_gpt35_format(df, job)

    if job == 'analysis':
        train_data, test_data = train_test_split(
            convert_data,
            test_size=0.33,
            random_state=1
        )

    def write_to_jsonl(data, file_path):
        with open(file_path, 'w') as file:
            for entry in data:
                json.dump(entry, file)
                file.write('\n')

    if job == 'analysis':
        train_file_name = f"{job}_train.jsonl"
        test_file_name = f"{job}_test.jsonl"
        write_to_jsonl(train_data, train_file_name)
        write_to_jsonl(test_data, test_file_name)

    elif job == 'generate':
        file_name = f"{job}.jsonl"
        write_to_jsonl(convert_data, file_name)

job = 'analysis'
run(job)
