import requests
import json
from typing import Dict, List
from datasets import load_dataset

ALLOWED_EVAL_DATASETS = ["minif2f", "proofnet"]

MINIF2F_RAW_URL: str = "https://raw.githubusercontent.com/deepseek-ai/DeepSeek-Prover-V1.5/refs/heads/main/datasets/minif2f.jsonl"
PROOFNET_RAW_URL: str = "https://raw.githubusercontent.com/deepseek-ai/DeepSeek-Prover-V1.5/refs/heads/main/datasets/proofnet.jsonl"

def load_jsonl_from_url(url):
    """
    Loads a .jsonl file from a URL and returns a list of dictionaries.

    Args:
        url (str): URL to the .jsonl file

    Returns:
        list: List of dictionaries parsed from the .jsonl file
    """
    # Download content from URL
    response = requests.get(url)
    response.raise_for_status()  # Raise exception for HTTP errors

    # Parse each line as a JSON object and collect in a list
    data = []
    for line in response.text.splitlines():
        if line.strip():  # Skip empty lines
            data.append(json.loads(line))

    return data

def load_json_from_url(url: str) -> Dict:
    response = requests.get(url)
    response.raise_for_status()  # Raise exception for HTTP errors
    return json.loads(response.text)

def load_eval_dataset(dataset_name: str, split: str) -> List[Dict]:
    assert (
        dataset_name in ALLOWED_EVAL_DATASETS
    ), f"Dataset {dataset_name} not in allowed datasets: {ALLOWED_EVAL_DATASETS}"

    if dataset_name == "minif2f":
        dataset = load_dataset("LukeBailey181/minif2f_lean", split="train")
        list_dataset = dataset.to_list()
        dataset = [x for x in list_dataset if x["split"] == split]

    elif dataset_name == "proofnet":
        raw_url = PROOFNET_RAW_URL
        dataset = load_jsonl_from_url(raw_url)
        dataset = [x for x in dataset if x["split"] == split]

    return dataset