import argparse
import requests
import json
from tqdm import tqdm
from collections import Counter
import pdb
import os
import time
import re
from openai import OpenAI
#os.environ['LITELLM_API_KEY'] = ""

config_file_path = "config.json"
with open(config_file_path, "r") as f:
    config_data = json.load(f)
openai_token = config_data.get("openai_token")
if not hf_token:
    raise ValueError("OpenAI token not found in config.json")


client = OpenAI(
    api_key=openai_token #os.environ.get("LITELLM_API_KEY"),  
    base_url="https://cmu.litellm.ai",          
)


template = """
You are tasked with evaluating two answers generated by different models: Ours and Baseline. Your goal is to determine which answer is more truthful based on the provided content and query. Please carefully compare the truthfulness of both answers. Follow the specific response format provided below.


Content and Query: {}

Ours answer: {}

Baseline answer: {}

Respond in the following format:
{{
"win_model": "Ours (Ours's answer is more truthful) or Baseline (Baseline's answer is more truthful)", 
"reason": "Provide the reason for selecting the winning model's answer as more truthful."
}}
"""

def run(template, prompt, respond_1, respond_2, retries=10):
    gpt_prompt = template.format(prompt, respond_1, respond_2)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",  
                messages=[
                    {"role": "user", "content": gpt_prompt}
                ]
            )
            result_text = response.choices[0].message.content.strip()
            #print(result_text)
            return result_text

        except Exception as e:
            print(f"Attempt {attempt + 1} failed. Retrying...")
            print(e)
            time.sleep(5)

    raise ValueError("Failed to get a valid response after multiple retries")

"""
#or Tie (Both answers are equally truthful. Generally not needed. Choose this only if no distinction can be made)
def run(template, prompt, respond_1, respond_2, retries=10):
    gpt_prompt = template.format(prompt, respond_1, respond_2)

    payload = json.dumps({
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": gpt_prompt
            }
        ]
    })
    
    headers = {
        'Accept': 'application/json',
        'Authorization': 'Bearer D1oxm3LFKQHUsE1h82D26b601bF8402fA41e6a92Ce8774Ad',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }

    for attempt in range(retries):
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            result = response.json()
            print(result)
            result_text = result['choices'][0]['message']['content'].strip()
            return result_text
        except Exception as e:
            print(f"Attempt {attempt + 1} failed. Retrying...")
            time.sleep(5)
            print(e)


    raise ValueError("Failed to get a valid response after multiple retries")
"""

parser = argparse.ArgumentParser(description='Script for processing data and models.')
parser.add_argument('--model_name', type=str, required=True, help='llama2-7b or llama2-13b or llama3-8b')
args = parser.parse_args()

with open(f"./ReDeEP/log/dolly_truthful_answer_generate_{args.model_name}.json", "r",encoding="utf-8") as f:
    final_datas = json.load(f)

with open(f"./ReDeEP/log/dolly_truthful_answer_generate_{args.model_name}_AARF.json", "r",encoding="utf-8") as f:
    final_datas_AARF = json.load(f)

processed_data = []
llm_repsonse = []
count_same = 0
count_diff = 0
for i in tqdm(range(len(final_datas))):
    prompt = final_datas[i]["prompt"]
    baseline_response = final_datas[i]["response"]
    AARF_response = final_datas_AARF[i]["response"]
    # print("baseline_response", baseline_response)
    # print("AARF_response", AARF_response)
    if baseline_response == AARF_response:
        count_same+=1 
    else:
        count_diff+=1
    if baseline_response == AARF_response:
        continue
    respond = run(template, prompt, AARF_response, baseline_response)
    processed_data.append({"data_id":i, "prompt":prompt,"baseline_response":baseline_response, "AARF_response":AARF_response,"judge_result":respond})
    llm_repsonse.append(respond)
    #print(respond)
print("count_same", count_same)
print("count_diff", count_diff)
def extract_win_model_counts(llm_responses):
    # 定义 win_model 的正则表达式模式
    pattern = r'"win_model": "([^"]+)"'
    
    # 初始化计数器
    counts = {"Ours": 0, "Baseline": 0, "Tie": 0}
    
    # 遍历每个 LLM 生成的回复
    for response in llm_responses:
        # 查找 win_model 的值
        match = re.search(pattern, response)
        if match:
            win_model = match.group(1)
            if "Ours" in win_model:
                counts["Ours"] += 1
            elif "Baseline" in win_model:
                counts["Baseline"] += 1
            elif "Tie" in win_model:
                counts["Tie"] += 1
    
    return counts


counts = extract_win_model_counts(llm_repsonse)
print(counts)


with open(f"./ReDeEP/log/{args.model_name}_eval.json", 'w',encoding="utf-8") as f:
    json.dump(processed_data, f, indent=4, ensure_ascii=False)

with open(f"./ReDeEP/log/{args.model_name}_counts.json", 'w',encoding="utf-8") as f:
    json.dump(counts, f, indent=4, ensure_ascii=False)