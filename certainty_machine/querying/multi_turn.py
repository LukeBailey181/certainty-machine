"""
multi_turn.py

Multi-turn prompting functionality for iterative proof refinement.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass

from certainty_machine.querying.api import query_model, query_model_batch, QueryResult
from certainty_machine.querying.prompts import get_prover_prompt, get_refinement_prompt, extract_proof, NO_CODE_FOUND_TAG
from certainty_machine.verification.verify_client import verify_lean_code
from certainty_machine.verification.types import VerificationOutput


@dataclass
class MultiTurnResult:
    """Result of multi-turn prompting."""
    success: bool
    turns_taken: int
    final_proof: Optional[str]
    final_verification: Optional[VerificationOutput]
    turn_history: List[Tuple[str, str, VerificationOutput]]  # (prompt, response, verification)


@dataclass
class BatchMultiTurnResult:
    """Result of batch multi-turn prompting."""
    individual_results: List[MultiTurnResult]
    total_problems: int
    successful_problems: int


def multi_turn_prompt(
    header: str,
    theorem: str,
    verifier_address: str,
    model: str,
    max_turns: int = 5,
    max_gen_tokens: int = 8000,
    temperature: float = 1.0
) -> MultiTurnResult:
    """
    Iteratively refine a proof using multi-turn prompting.
    
    Args:
        header: Lean file header (imports, etc.)
        theorem: Formal theorem statement
        verifier_address: Address of the verification server
        model: AI model to use for generation
        max_turns: Maximum number of refinement turns
        max_gen_tokens: Maximum tokens per generation
        temperature: Model temperature
        
    Returns:
        MultiTurnResult with success status and refinement history
    """
    turn_history = []
    current_proof = None
    
    for turn in range(max_turns):
        # Determine prompt for this turn
        if turn == 0:
            # First turn: initial attempt
            prompt = get_prover_prompt(header=header, theorem=theorem)
        else:
            # Subsequent turns: show full conversation history
            prompt = get_refinement_prompt(
                header=header,
                theorem=theorem,
                turn_history=turn_history
            )
        
        # Query the model
        response: QueryResult = query_model(
            prompt=prompt,
            model=model,
            max_gen_tokens=max_gen_tokens,
            temperature=temperature
        )
        
        if response is None:
            # Query failed completely
            return MultiTurnResult(
                success=False,
                turns_taken=turn + 1,
                final_proof=current_proof,
                final_verification=None,
                turn_history=turn_history
            )
        
        # Extract proof from response
        proof = extract_proof(response.response_text)
        if proof == NO_CODE_FOUND_TAG:
            # No valid proof found in response
            # Create a dummy verification for history
            dummy_verification = VerificationOutput(
                verdict=False,
                output={"error": "No proof found in model response"}
            )
            turn_history.append((prompt, response.response_text, dummy_verification))
            continue
        
        current_proof = proof
        
        # Verify the proof
        full_lean_file = header + theorem + proof
        verification_outputs: List[VerificationOutput] = verify_lean_code(
            verifier_address=verifier_address,
            lean_code=[full_lean_file]
        )
        
        verification = verification_outputs[0]
        turn_history.append((prompt, response.response_text, verification))
        
        # Check if proof is correct
        if verification.verdict:
            return MultiTurnResult(
                success=True,
                turns_taken=turn + 1,
                final_proof=current_proof,
                final_verification=verification,
                turn_history=turn_history
            )
    
    # Reached max turns without success
    final_verification = turn_history[-1][2] if turn_history else None
    return MultiTurnResult(
        success=False,
        turns_taken=max_turns,
        final_proof=current_proof,
        final_verification=final_verification,
        turn_history=turn_history
    )


def multi_turn_prompt_batch(
    headers: List[str],
    theorems: List[str],
    verifier_address: str,
    model: str,
    max_turns: int = 5,
    max_gen_tokens: int = 8000,
    temperature: float = 1.0,
    num_workers: int = 16
) -> BatchMultiTurnResult:
    """
    Iteratively refine proofs for multiple problems using batched multi-turn prompting.
    
    This is much more efficient than individual multi-turn calls because:
    1. Model queries are batched per turn across all problems
    2. Verification calls are batched per turn across all problems
    
    Args:
        headers: List of Lean file headers (imports, etc.)
        theorems: List of formal theorem statements
        verifier_address: Address of the verification server
        model: AI model to use for generation
        max_turns: Maximum number of refinement turns
        max_gen_tokens: Maximum tokens per generation
        temperature: Model temperature
        num_workers: Number of workers for batched model queries
        
    Returns:
        BatchMultiTurnResult with individual results for each problem
    """
    assert len(headers) == len(theorems), "Headers and theorems lists must have same length"
    
    num_problems = len(headers)
    
    # Initialize tracking for each problem
    problem_states = []
    for i in range(num_problems):
        problem_states.append({
            'header': headers[i],
            'theorem': theorems[i],
            'current_proof': None,
            'success': False,
            'turns_taken': 0,
            'turn_history': [],
            'active': True  # Whether this problem is still being worked on
        })
    
    # Process turn by turn
    for turn in range(max_turns):
        print(f"🔄 Processing turn {turn + 1}/{max_turns}...")
        
        # Collect prompts for all active problems
        active_indices = []
        prompts = []
        
        for i, state in enumerate(problem_states):
            if not state['active']:
                continue
                
            active_indices.append(i)

            state['turns_taken'] = turn + 1
            
            if turn == 0:
                # First turn: initial attempt
                prompt = get_prover_prompt(header=state['header'], theorem=state['theorem'])
            else:
                # Subsequent turns: show full conversation history
                prompt = get_refinement_prompt(
                    header=state['header'],
                    theorem=state['theorem'],
                    turn_history=state['turn_history']
                )
            
            prompts.append(prompt)
        
        if not active_indices:
            # All problems solved or failed
            break
        
        print(f"  🤖 Querying model for {len(active_indices)} active problems...")
        
        # Batch query the model for all active problems
        query_results: List[QueryResult] = query_model_batch(
            prompts=prompts,
            model=model,
            max_gen_tokens=max_gen_tokens,
            temperature=temperature,
            num_workers=num_workers
        )
        
        # Extract proofs and prepare for verification
        verification_proofs = []
        verification_indices = []
        
        for idx, (active_idx, query_result) in enumerate(zip(active_indices, query_results)):
            state = problem_states[active_idx]
            
            if query_result.is_error:
                # Query failed
                dummy_verification = VerificationOutput(
                    verdict=False,
                    output={"error": f"Query failed: {query_result.response_text}"}
                )
                state['turn_history'].append((prompts[idx], query_result.response_text, dummy_verification))
                continue
            
            # Extract proof from response
            proof = extract_proof(query_result.response_text)
            if proof == NO_CODE_FOUND_TAG:
                # No valid proof found
                dummy_verification = VerificationOutput(
                    verdict=False,
                    output={"error": "No proof found in model response"}
                )
                state['turn_history'].append((prompts[idx], query_result.response_text, dummy_verification))
                continue
            
            # Prepare for verification
            state['current_proof'] = proof
            full_lean_file = state['header'] + state['theorem'] + proof
            verification_proofs.append(full_lean_file)
            verification_indices.append((active_idx, idx))  # (problem_index, query_index)
        
        if not verification_proofs:
            # No valid proofs to verify this turn
            continue
        
        print(f"  🔍 Verifying {len(verification_proofs)} proofs...")
        
        # Batch verify all proofs
        verification_outputs: List[VerificationOutput] = verify_lean_code(
            verifier_address=verifier_address,
            lean_code=verification_proofs
        )
        
        # Process verification results
        for (active_idx, query_idx), verification in zip(verification_indices, verification_outputs):
            state = problem_states[active_idx]
            query_result = query_results[query_idx]
            prompt = prompts[query_idx]
            
            # Add to turn history
            state['turn_history'].append((prompt, query_result.response_text, verification))
            
            # Check if proof is correct
            if verification.verdict:
                state['success'] = True
                state['active'] = False
                print(f"    ✅ Problem {active_idx} solved in turn {turn + 1}!")
        
        # Deactivate problems that have reached max turns
        if turn == max_turns - 1:
            for state in problem_states:
                if state['active']:
                    state['active'] = False
    
    # Create individual results
    individual_results = []
    successful_count = 0
    
    for state in problem_states:
        final_verification = state['turn_history'][-1][2] if state['turn_history'] else None
        
        result = MultiTurnResult(
            success=state['success'],
            turns_taken=state['turns_taken'],
            final_proof=state['current_proof'],
            final_verification=final_verification,
            turn_history=state['turn_history']
        )
        individual_results.append(result)
        
        if state['success']:
            successful_count += 1
    
    return BatchMultiTurnResult(
        individual_results=individual_results,
        total_problems=num_problems,
        successful_problems=successful_count
    )


def print_multi_turn_result(result: MultiTurnResult, colors_class=None) -> None:
    """
    Print a formatted summary of multi-turn results.
    
    Args:
        result: MultiTurnResult to display
        colors_class: Optional Colors class for terminal formatting
    """
    if colors_class is None:
        # Create a dummy colors class if none provided
        class Colors:
            HEADER = BLUE = CYAN = GREEN = YELLOW = RED = BOLD = UNDERLINE = END = ''
    else:
        Colors = colors_class
    
    print(f"{Colors.HEADER}{Colors.BOLD}🔄 Multi-Turn Results:{Colors.END}")
    print(f"{Colors.BLUE}Success: {result.success}{Colors.END}")
    print(f"{Colors.BLUE}Turns taken: {result.turns_taken}{Colors.END}")
    
    if result.success:
        print(f"{Colors.GREEN}{Colors.BOLD}✅ Final proof succeeded!{Colors.END}")
        if result.final_proof:
            print(f"{Colors.GREEN}Final proof:{Colors.END}")
            print(f"{Colors.GREEN}{result.final_proof}{Colors.END}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ Failed to find correct proof{Colors.END}")
        if result.final_verification and "errors" in result.final_verification.output:
            print(f"{Colors.RED}Final errors:{Colors.END}")
            print(f"{Colors.RED}{result.final_verification.output['errors']}{Colors.END}")
    
    print(f"{Colors.CYAN}{Colors.BOLD}Turn History:{Colors.END}")
    for i, (prompt, response, verification) in enumerate(result.turn_history):
        status = "✅" if verification.verdict else "❌"
        print(f"{Colors.CYAN}Turn {i+1}: {status}{Colors.END}")