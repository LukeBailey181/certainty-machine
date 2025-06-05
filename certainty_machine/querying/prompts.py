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