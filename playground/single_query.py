from typing import Dict, List

from certainty_machine.querying.prompts import get_prover_prompt
from certainty_machine.data import load_eval_dataset


def main():

    dataset = "minif2f"
    split = "valid" # Can be "valid" or "test"

    dataset: List[Dict] = load_eval_dataset(dataset, split)
    print(f"Loaded {len(dataset)} examples from {dataset} {split} split")



if __name__ == "__main__":
    main()