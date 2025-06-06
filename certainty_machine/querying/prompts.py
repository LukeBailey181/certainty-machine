"""
prompts.py

Helper functions to get prompts for models.
"""

from typing import Optional


def get_prover_prompt(*, header: str, theorem: str) -> str:
    part_1 = f"""You are a math expert. Here is a lean4 problem:

```lean4
{header + theorem}
```

Reason through how you to solve this problem, and then output the complete proof (exluding the theorem statement) in lean4 between
<begin_proof> and <end_proof> tags. Make sure that the proof is indented correctly. MAKE SURE TO INCLUDE THE <begin_proof> AND <end_proof> TAGS,
and ONLY includde the proof between these tags, NOT the theorem statement or imports.

REMEMBER: when you start your proof write <begin_proof> and when you finish your proof write <end_proof>.
Also DO NOT use sorry, this will be counted as a failed proof.
"""

    ic_example_1 = """\n\nHere is an example:

===========================================
INPUT:
```lean4
import Mathlib
import Aesop
set_option maxHeartbeats 0
open BigOperators Real Nat Topology Rat

theorem mathd_numbertheory_374 (n : ℤ) (h₀ : 2 * n^3 - 3 * n^2 + n - 2 = 0) : n = 2 ∨ n = -1 ∨ n = -1:= by
```

OUTPUT: <... your reasoning ...>

```lean4
<begin_proof>
  have h : 2 * n^3 - 3 * n^2 + n - 2 = 0 := h₀
  have h₁ : n = 2 ∨ n = -1 ∨ n = -1 := by
    apply Or.inl
    apply Eq.symm
    nlinarith [sq_nonneg (n - 2), sq_nonneg (n + 1)]
  apply h₁
<end_proof>
```
===========================================
"""

    return part_1 + ic_example_1


def get_refinement_prompt(
    header: str, 
    theorem: str, 
    turn_history: list
) -> str:
    """Create a refinement prompt based on full conversation history."""
    
    # Build the conversation history section
    history_section = ""
    for i, (prompt, response, verification) in enumerate(turn_history):
        history_section += f"\n--- TURN {i+1} ---\n"
        
        # For the first turn, show it was the initial attempt
        if i == 0:
            history_section += "INITIAL ATTEMPT:\n"
        else:
            history_section += f"REFINEMENT ATTEMPT {i}:\n"
        
        history_section += f"\nYour reasoning and response:\n{response}\n"
        
        # Show verification result
        if verification.verdict:
            history_section += f"\nVerification: SUCCESS\n"
        else:
            history_section += f"\nVerification: FAILED\n"
            if "errors" in verification.output:
                history_section += f"Errors:\n{verification.output['errors']}\n"
            else:
                history_section += f"Output: {verification.output}\n"
    
    return f"""You are a math expert working on a lean4 problem. You have made several attempts to solve this problem. Here is the complete history of your attempts:

ORIGINAL PROBLEM:
```lean4
{header + theorem}
```

CONVERSATION HISTORY:
{history_section}

Based on all the above attempts and their verification results, please analyze what went wrong and provide a corrected proof. Learn from ALL previous errors and attempts, not just the most recent one. Output the complete corrected proof (excluding the theorem statement) in lean4 between <begin_proof> and <end_proof> tags. Make sure that the proof is indented correctly.

REMEMBER: 
- Start your proof with <begin_proof> and end with <end_proof>
- ONLY include the proof between these tags, NOT the theorem statement or imports
- DO NOT use sorry, this will be counted as a failed proof
- Learn from ALL previous attempts and errors shown above
"""


# ---------------------------------
# Other model promtp output utils
# ---------------------------------

NO_CODE_FOUND_TAG = "NO CODE FOUND"
NO_CONJECTURE_FOUND_TAG = "NO CONJECTURE FOUND"
NO_REVIEWER_SCORE_FOUND_TAG = (
    -1234567890
)  # We set this to a majic number that the model wont generate and type checks


def extract_proof(generation: str) -> str:
    start_idx = generation.rfind("<begin_proof>")
    end_idx = generation.rfind("<end_proof>")

    if start_idx == -1 or end_idx == -1:
        return NO_CODE_FOUND_TAG

    return generation[start_idx + len("<begin_proof>") : end_idx]