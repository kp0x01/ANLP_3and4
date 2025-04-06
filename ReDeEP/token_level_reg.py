import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
import pdb
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser(description='Script for processing data and models.')
parser.add_argument('--model_name', type=str, required=True, help='llama2-7b or llama2-13b')
parser.add_argument(
    '--dataset', 
    type=str, 
    default="ragtruth", 
    help='ragtruth, dolly'
)

args = parser.parse_args()


def construct_dataframe(file_path, number):
    # Sample data for illustration
    with open(file_path, "r") as f:
        response = json.load(f)  

    # Create a dataframe to hold the combined information

    data_dict = {
        "identifier": [],
        **{f"external_similarity_{k}": [] for k in range(number)},
        **{f"parameter_knowledge_difference_{k}": [] for k in range(number)},
        "hallucination_label": []
    }

    for i, resp in enumerate(response):
        if resp["split"] != "test":
            continue
        for j in range(len(resp["external_similarity"])):
            data_dict["identifier"].append(f"response_{i}_item_{j}")
            for k in range(number):
                data_dict[f"external_similarity_{k}"].append(resp["external_similarity"][j][k])
                data_dict[f"parameter_knowledge_difference_{k}"].append(resp["parameter_knowledge_difference"][j][k])
            data_dict["hallucination_label"].append(resp["hallucination_label"][j])

    df = pd.DataFrame(data_dict)

    print(df["hallucination_label"].value_counts(normalize=True))
    return df


def linear_regression(df):
    # Extract features and labels
    features = df.drop(columns=["identifier", "hallucination_label"])
    labels = df["hallucination_label"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Initialize and train the logistic regression model
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(accuracy)
    print(report)


def calculate_auc_pcc(df, number):
    # Calculate AUC and Pearson correlation for each of the 64 values
    auc_external_similarity = []
    pearson_external_similarity = []

    auc_parameter_knowledge_difference = []
    pearson_parameter_knowledge_difference = []

    for k in range(number):
        # External similarity metrics
        auc_ext = roc_auc_score(1 - df['hallucination_label'], df[f'external_similarity_{k}'])
        pearson_ext, _ = pearsonr(df[f'external_similarity_{k}'], 1 - df['hallucination_label'])
        auc_external_similarity.append((auc_ext, f'external_similarity_{k}'))
        pearson_external_similarity.append((pearson_ext, f'external_similarity_{k}'))

        # Parameter knowledge difference metrics
        auc_param = roc_auc_score(df['hallucination_label'], df[f'parameter_knowledge_difference_{k}'])
        if df[f'parameter_knowledge_difference_{k}'].nunique() == 1:
            print(k)
        pearson_param, _ = pearsonr(df[f'parameter_knowledge_difference_{k}'], df['hallucination_label'])
        auc_parameter_knowledge_difference.append((auc_param, f'parameter_knowledge_difference_{k}'))
        pearson_parameter_knowledge_difference.append((pearson_param, f'parameter_knowledge_difference_{k}'))
    return auc_external_similarity, auc_parameter_knowledge_difference


def calculate_auc_pcc_32_32(df, top_n, top_k, alpha, auc_external_similarity, auc_parameter_knowledge_difference, m=1):
    collect_info = {}
    # Sort by AUC and select the top N features (for example, top 5)
    top_auc_external_similarity = sorted(auc_external_similarity, reverse=True)[:top_n]
    collect_info.update({"select_heads":[sorted_copy_heads[eval(name.split('_')[-1])] for _, name in top_auc_external_similarity]})

    top_auc_parameter_knowledge_difference = sorted(auc_parameter_knowledge_difference, reverse=True)[:top_k]
    if args.model_name == "llama2-13b":
        base_layer = 7
    else:
        base_layer = 0
    collect_info.update({"select_layers": [eval(name.split('_')[-1])+base_layer for _, name in top_auc_parameter_knowledge_difference]})

    # Sum the top N features for each type
    df['external_similarity_sum'] = df[[col for _, col in top_auc_external_similarity]].sum(axis=1)
    df['parameter_knowledge_difference_sum'] = df[[col for _, col in top_auc_parameter_knowledge_difference]].sum(axis=1)

    # Calculate AUC for the summed top N features
    final_auc_external_similarity = roc_auc_score(1 - df['hallucination_label'], df['external_similarity_sum'])
    final_auc_parameter_knowledge_difference = roc_auc_score(df['hallucination_label'], df['parameter_knowledge_difference_sum'])

    # Calculate Pearson correlation for the summed top N features
    final_pearson_external_similarity, _ = pearsonr(df['external_similarity_sum'], 1 - df['hallucination_label'])
    final_pearson_parameter_knowledge_difference, _ = pearsonr(df['parameter_knowledge_difference_sum'], df['hallucination_label'])

    results = {
        "Top N AUC External Similarity": final_auc_external_similarity,
        "Top N AUC Parameter Knowledge Difference": final_auc_parameter_knowledge_difference,
        "Top N Pearson Correlation External Similarity": final_pearson_external_similarity,
        "Top N Pearson Correlation Parameter Knowledge Difference": final_pearson_parameter_knowledge_difference
    }

    scaler = MinMaxScaler()
    # Normalize the columns
    df['external_similarity_sum_normalized'] = scaler.fit_transform(df[['external_similarity_sum']])
    external_similarity_sum_max_value = scaler.data_max_[0]
    external_similarity_sum_min_value = scaler.data_min_[0]
    collect_info.update({
        "head_max_min": [external_similarity_sum_max_value, external_similarity_sum_min_value],
    })
    df['parameter_knowledge_difference_sum_normalized'] = scaler.fit_transform(df[['parameter_knowledge_difference_sum']])
    parameter_knowledge_sum_max_value = scaler.data_max_[0]
    parameter_knowledge_sum_min_value = scaler.data_min_[0]
    collect_info.update({
        "layers_max_min": [parameter_knowledge_sum_max_value, parameter_knowledge_sum_min_value]
    })
    # Subtract the normalized columns
    df['difference_normalized'] = m*df['parameter_knowledge_difference_sum_normalized'] - alpha*df['external_similarity_sum_normalized']

    # Calculate AUC for the difference
    auc_difference_normalized = roc_auc_score(df['hallucination_label'], df['difference_normalized'])
    person_difference_normalized, _ = pearsonr(df['hallucination_label'], df['difference_normalized'])
    results.update({"Normalized Difference AUC": auc_difference_normalized})
    results.update({"Normalized Difference Pearson Correlation": person_difference_normalized})

    # Group by 'identifier' and calculate the sum of 'difference_normalized' and max of 'hallucination_label'
    df['response_group'] = df['identifier'].str.extract(r'(response_\d+)')

    # Group by 'response_group' and calculate the sum of 'difference_normalized' and max of 'hallucination_label'
    grouped_df = df.groupby('response_group').agg(
        difference_normalized_mean=('difference_normalized', 'mean'),
        hallucination_label=('hallucination_label', 'max')
    ).reset_index()

    min_val = grouped_df['difference_normalized_mean'].min()
    max_val = grouped_df['difference_normalized_mean'].max()
    collect_info.update({'final_max_min': [max_val, min_val]})
    # 进行归一化
    grouped_df['difference_normalized_mean_norm'] = (grouped_df['difference_normalized_mean'] - min_val) / (max_val - min_val)


    # Calculate AUC for the grouped means
    auc_difference_normalized = roc_auc_score(grouped_df['hallucination_label'], grouped_df['difference_normalized_mean_norm'])
    person_difference_normalized, _ = pearsonr(grouped_df['hallucination_label'], grouped_df['difference_normalized_mean_norm'])


    results.update({"Grouped means AUC": auc_difference_normalized})
    results.update({"Grouped means Pearson Correlation": person_difference_normalized})
    return auc_difference_normalized, person_difference_normalized



if __name__ == "__main__":
    if args.model_name == "llama2-7b":
        topk_head_path = "./log/test_llama2_7B/topk_heads.json"
    elif args.model_name == "llama2-13b":
        topk_head_path = "./log/test_llama2_13B/topk_heads.json"
    elif args.model_name == "llama3-8b":
        topk_head_path =  "./log/test_llama3_8B/topk_heads.json" 
    else:
        print("model name error")
        exit(-1)

    with open(topk_head_path,'r') as f:
        # [(layer, head)...]
        copy_heads = json.load(f)
    sorted_copy_heads = sorted(copy_heads, key=lambda x: (x[0], x[1]))

    if args.model_name == "llama2-7b":
        if args.dataset == "ragtruth":
            data_path = "./log/test_llama2_7B/llama2_7B_response_v1.json"
        elif args.dataset == "dolly":
            data_path = "./log/test_llama2_7B/llama2_7B_response_v1_dolly.json"
        number = 32
    elif args.model_name == "llama2-13b":
        if args.dataset == "ragtruth":
            data_path = "./log/test_llama2_13B/llama2_13B_response_v1.json"
        elif args.dataset == "dolly":
            data_path = "./log/test_llama2_13B/llama2_13B_response_v1_dolly.json"
        number = 32
    elif args.model_name == "llama3-8b":
        if args.dataset == "ragtruth":
            data_path = "./log/test_llama3_8B/llama3_8B_response_v1.json"
        elif args.dataset == "dolly":
            data_path = "./log/test_llama2_13B/llama3_8B_response_v1_dolly.json"
        number = 32
    else:
        print("model name error")
        exit(-1)
    df = construct_dataframe(data_path, number)
    auc_external_similarity, auc_parameter_knowledge_difference = calculate_auc_pcc(df.iloc[:, :int(df.shape[1] * 0.5)], number)
    run_all = False

    if args.model_name == "llama2-7b":
        if args.dataset == "ragtruth":
            i, j, k, m = 1, 10, 0.2, 1
        elif args.dataset == "dolly":
            i, j , k, m = 4, 3, 0.2, 1

    elif args.model_name == "llama2-13b":
        if args.dataset == "ragtruth":
            i, j, k, m = 2, 17, 0.6, 1
        elif args.dataset == "dolly":
            i, j, k, m = 4, 5, 0.6, 1
        
    elif args.model_name == "llama3-8b":
        if args.dataset == "ragtruth":
            i, j, k, m = 3, 30, 0.4, 1
        elif args.dataset == "dolly":
            i, j, k, m = 1, 1, 0.1, 1
    else:
        print("model name error")
        exit(-1)
    auc_difference_normalized, person_difference_normalized = calculate_auc_pcc_32_32(df, i, j, k, auc_external_similarity, auc_parameter_knowledge_difference, m)
    if args.model_name == "llama2-7b":
        save_path = "./log/test_llama2_7B/ReDeEP(token).json"
    elif args.model_name == "llama2-13b":
        save_path = "./log/test_llama2_13B/ReDeEP(token).json"
    elif args.model_name == "llama3-8b":
        save_path = "./log/test_llama3_8B/ReDeEP(token).json"
    else:
        print("model name error")
        exit(-1)
    result_dict = {"auc":auc_difference_normalized, "pcc": person_difference_normalized}
    print(result_dict)
    with open(save_path, 'w') as f:
        json.dump(result_dict, f, ensure_ascii=False)