"""
verify_client.py

Functions to submit lean code for verification to a verifier server.
"""

import requests
from typing import List, Dict, Tuple, Any

from matplotlib import pyplot as plt
import wandb

from certainty_machine.verification.types import VerificationOutput

VERIFIER_TIMEOUT = 60


def verify_lean_code(
    verifier_address: str, lean_code: List[str]
) -> List[VerificationOutput]:
    """Send Lean code to the verifier and check if it passes.

    Args:
        verifier_address: The address of the verifier server.
        lean_code: A list of strings, each representing a FULL lean file, header + theorem + proof.

    Returns:
        A tuple containing two lists:
        - The first list contains the verdicts for each proof.
        - The second list contains the verification outputs for each proof.
    """
    # Here we are hardocding the timeout to 1 minute, and no memory limit
    api_params = {
        "cleaned_proofs": lean_code,
        "timeout": VERIFIER_TIMEOUT,
        "memory_limit": -1,
    }
    try:
        api_response = requests.post(
            f"http://{verifier_address}/verify", json=api_params
        )
        verification_outputs = api_response.json()["verification_outputs"]

        # Extract verdicts from verification outputs.
        verdicts = [
            output.get("complete", False) if isinstance(output, dict) else False
            for output in verification_outputs
        ]
        verification_outputs = [
            VerificationOutput(verdict, output)
            for verdict, output in zip(verdicts, verification_outputs)
        ]

        # The second return is the big long dict containing the entire compiler output
        return verification_outputs
    except Exception as e:
        print("Something went wrong when verifying the code")
        raise e


def verify_dict(
    verifier_address: str, generations: Dict[str, List[str]]
) -> Dict[str, List[VerificationOutput]]:
    """Verifies a dict of generations.

    Args:
        verifier_address: The address of the verifier server.
        generations: A dictionary mapping problem names to lists of proof attempts.

    Returns:
        A dictionary mapping problem names to dictionaries containing verdicts and outputs.
        Each value is a dict with "verdicts" and "outputs" keys. Verdicts is bool list,
        outputs is a list of dicts, each representing the full veriifer output.
    """

    all_proofs: List[str] = []
    proof_map: Dict[
        int, Tuple[str, int]
    ] = {}  # Maps index in flat list to (problem_name, local_index)

    # Flatten proofs while tracking their origin
    for problem_name, proofs in generations.items():
        for i, proof in enumerate(proofs):
            proof_map[len(all_proofs)] = (problem_name, i)
            all_proofs.append(proof)

    # Verify all proofs in one batch
    verification_results: List[VerificationOutput] = verify_lean_code(
        verifier_address, all_proofs
    )

    if wandb.run is not None:
        # Log the verification times. If we ever see loads of timeouts, then
        # the server is orverloaded
        verification_times = [
            x.output.get("verify_time", -1) for x in verification_results
        ]
        # Plot histogram
        fig, ax = plt.subplots()
        ax.hist(verification_times, bins=100)
        ax.axvline(x=VERIFIER_TIMEOUT, color="red", linestyle="--", label="Timeout")
        ax.set_title("Verification Times")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency")
        ax.legend()
        wandb.log({"infra/verification_times_hist": wandb.Image(fig)})
        plt.close()

    # Initialize verifier_outputs dictionary
    verifier_outputs: Dict[str, Dict[str, List[Any]]] = {}
    for problem_name in generations.keys():
        verifier_outputs[problem_name] = {
            "verdicts": [False] * len(generations[problem_name]),
            "outputs": [None] * len(generations[problem_name]),
        }

    # Map verification results back to their problems
    for flat_idx, ver_output in enumerate(verification_results):
        verdict = ver_output.verdict
        compiler_output = ver_output.output
        if flat_idx in proof_map:
            problem_name, local_idx = proof_map[flat_idx]
            verifier_outputs[problem_name]["verdicts"][local_idx] = verdict
            verifier_outputs[problem_name]["outputs"][local_idx] = compiler_output
        else:
            raise ValueError(f"Flat index {flat_idx} not found in proof map")

    # Convert to list of VerificationOutput
    formatted_verifier_outputs: Dict[str, List[VerificationOutput]] = {}
    for problem_name in verifier_outputs.keys():
        formatted_verifier_outputs[problem_name] = [
            VerificationOutput(verdict, output)
            for verdict, output in zip(
                verifier_outputs[problem_name]["verdicts"],
                verifier_outputs[problem_name]["outputs"],
            )
        ]

    return formatted_verifier_outputs
