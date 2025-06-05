from typing import Dict, List

from certainty_machine.querying.prompts import get_prover_prompt, extract_proof, NO_CODE_FOUND_TAG
from certainty_machine.verification.verify_client import verify_lean_code
from certainty_machine.querying.api import query_model, QueryResult
from certainty_machine.data import load_eval_dataset
from certainty_machine.verification.types import VerificationOutput

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'      # Magenta
    BLUE = '\033[94m'        # Blue
    CYAN = '\033[96m'        # Cyan
    GREEN = '\033[92m'       # Green
    YELLOW = '\033[93m'      # Yellow
    RED = '\033[91m'         # Red
    BOLD = '\033[1m'         # Bold
    UNDERLINE = '\033[4m'    # Underline
    END = '\033[0m'          # End formatting


VERIFIER_ADDRESS = "172.24.75.239:8001"

def main():

    dataset_name = "minif2f"
    split = "valid" # Can be "valid" or "test"

    dataset: List[Dict] = load_eval_dataset(dataset_name, split)
    print(f"{Colors.BLUE}{Colors.BOLD}📊 Dataset Info:{Colors.END}")
    print(f"{Colors.BLUE}Loaded {len(dataset)} examples from {dataset_name} {split} split{Colors.END}")
    print()

    datapoint: Dict = dataset[0]
    lean_file: str = datapoint["header"] + datapoint["formal_statement"]
    print(f"{Colors.GREEN}{Colors.BOLD}📝 Lean File:{Colors.END}")
    print(f"{Colors.GREEN}{lean_file}{Colors.END}")
    print()

    # Get the prompt
    prompt = get_prover_prompt(
        header=datapoint["header"], 
        theorem=datapoint["formal_statement"]
    )

    # Query the model
    response: QueryResult = query_model(
        prompt=prompt,
        model="gpt-4o",
        max_gen_tokens=1000,
        temperature=1.0
    )

    print(f"{Colors.YELLOW}{Colors.BOLD}🤖 Prompt:{Colors.END}")
    print(f"{Colors.YELLOW}{prompt}{Colors.END}")
    print()
    print(f"{Colors.CYAN}{Colors.BOLD}💬 Model Response:{Colors.END}")
    print(f"{Colors.CYAN}{response.response_text}{Colors.END}")

    # get outputted proof 
    proof = extract_proof(response.response_text)
    if proof == NO_CODE_FOUND_TAG:
        raise ValueError(f"No proof found in model response: {response.response_text}")

    # Verify this
    full_lean_file = datapoint["header"] + datapoint["formal_statement"] + proof

    print()
    print(f"{Colors.GREEN}{Colors.BOLD}🔍 Verifying Proof:{Colors.END}")
    print(f"{Colors.GREEN}{full_lean_file}{Colors.END}")


    verification_outputs: List[VerificationOutput] = verify_lean_code(
        verifier_address=VERIFIER_ADDRESS,
        lean_code=[full_lean_file]
    )
    print()
    print(f"{Colors.GREEN}{Colors.BOLD}🔍 Verification Outputs:{Colors.END}")
    print(f"{Colors.GREEN}{verification_outputs[0].verdict}{Colors.END}")

    output = verification_outputs[0]
    compiler_outputs = output.output
    if not output.verdict:
        print(f"{Colors.RED}{Colors.BOLD}❌ Verification Failed:{Colors.END}")
        if "errors" in compiler_outputs:
            print(f"{Colors.RED}{compiler_outputs['errors']}{Colors.END}")
    else:
        print(f"{Colors.GREEN}{Colors.BOLD}✅ Verification Passed:{Colors.END}")




if __name__ == "__main__":
    main()