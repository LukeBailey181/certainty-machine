from typing import Dict, List

from certainty_machine.querying.multi_turn import multi_turn_prompt, print_multi_turn_result, MultiTurnResult
from certainty_machine.data import load_eval_dataset

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
    split = "valid"  # Can be "valid" or "test"

    dataset: List[Dict] = load_eval_dataset(dataset_name, split)
    print(f"{Colors.BLUE}{Colors.BOLD}📊 Dataset Info:{Colors.END}")
    print(f"{Colors.BLUE}Loaded {len(dataset)} examples from {dataset_name} {split} split{Colors.END}")
    print()

    datapoint: Dict = dataset[0]
    header: str = datapoint["header"]
    theorem: str = datapoint["formal_statement"]
    
    print(f"{Colors.GREEN}{Colors.BOLD}📝 Problem:{Colors.END}")
    print(f"{Colors.GREEN}{header + theorem}{Colors.END}")
    print()

    # Run multi-turn prompting
    result: MultiTurnResult = multi_turn_prompt(
        header=header,
        theorem=theorem,
        verifier_address=VERIFIER_ADDRESS,
        model="gpt-4o",
        max_turns=3,
        max_gen_tokens=1000,
        temperature=1.0
    )

    # Print results
    print_multi_turn_result(result, Colors)
    
    # Print detailed turn-by-turn breakdown if desired
    #print(f"\n{Colors.YELLOW}{Colors.BOLD}📋 Detailed Turn Breakdown:{Colors.END}")
    #for i, (prompt, response, verification) in enumerate(result.turn_history):
    #    print(f"\n{Colors.CYAN}{Colors.BOLD}--- TURN {i+1} ---{Colors.END}")
    #    
    #    # Show prompt (truncated for readability)
    #    print(f"{Colors.YELLOW}Prompt (first 200 chars):{Colors.END}")
    #    print(f"{Colors.YELLOW}{prompt[:200]}...{Colors.END}")
    #    
    #    # Show model response
    #    print(f"{Colors.CYAN}Model Response:{Colors.END}")
    #    print(f"{Colors.CYAN}{response}{Colors.END}")
    #    
    #    # Show verification result
    #    status = "✅ SUCCESS" if verification.verdict else "❌ FAILED"
    #    print(f"{Colors.GREEN if verification.verdict else Colors.RED}Verification: {status}{Colors.END}")
    #    
    #    if not verification.verdict and "errors" in verification.output:
    #        print(f"{Colors.RED}Errors:{Colors.END}")
    #        print(f"{Colors.RED}{verification.output['errors']}{Colors.END}")

    # print the final prompt response and verification output
    prompt, response, verification = result.turn_history[-1]
    print(f"{Colors.YELLOW}{Colors.BOLD}🤖 Final Prompt Response:{Colors.END}")
    print(f"{Colors.YELLOW}{prompt}{Colors.END}")
    print()
    print(f"{Colors.CYAN}{Colors.BOLD}💬 Model Response:{Colors.END}")
    print(f"{Colors.CYAN}{response}{Colors.END}")
    print()
    print(f"{Colors.GREEN}{Colors.BOLD}🔍 Verification Output:{Colors.END}")
    print(f"{Colors.GREEN}{verification.verdict}{Colors.END}")
    if not verification.verdict:
        print(f"{Colors.RED}{verification.output['errors']}{Colors.END}")


if __name__ == "__main__":
    main()