"""
benchmark_multiturn.py

Comprehensive evaluation of multi-turn prompting on the minif2f benchmark.
Runs parallel processes to evaluate multiple problems simultaneously and 
generates performance analysis plots.
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pydra.config import Config
from pydra.cli import main

from certainty_machine.data import load_eval_dataset
from certainty_machine.querying.multi_turn import multi_turn_prompt, multi_turn_prompt_batch, MultiTurnResult, BatchMultiTurnResult


def solve_single_problem(
    problem_idx: int,
    problem_data: Dict,
    verifier_address: str,
    model: str,
    max_turns: int,
    max_gen_tokens: int,
    temperature: float
) -> Tuple[int, Dict]:
    """
    Solve a single problem using multi-turn prompting.
    
    Args:
        problem_idx: Index of the problem in the dataset
        problem_data: Problem data dictionary
        verifier_address: Address of the verification server
        model: Model name to use
        max_turns: Maximum number of turns
        max_gen_tokens: Maximum generation tokens
        temperature: Model temperature
        
    Returns:
        Tuple of (problem_idx, result_dict)
    """
    try:
        header = problem_data["header"]
        theorem = problem_data["formal_statement"]
        
        result: MultiTurnResult = multi_turn_prompt(
            header=header,
            theorem=theorem,
            verifier_address=verifier_address,
            model=model,
            max_turns=max_turns,
            max_gen_tokens=max_gen_tokens,
            temperature=temperature
        )
        
        # Convert to serializable format
        result_dict = {
            "success": result.success,
            "turns_taken": result.turns_taken,
            "final_proof": result.final_proof,
            "problem_name": problem_data.get("name", f"problem_{problem_idx}"),
            "turn_results": [verdict for _, _, verification in result.turn_history for verdict in [verification.verdict]]
        }
        
        return problem_idx, result_dict
        
    except Exception as e:
        # Return error result
        error_result = {
            "success": False,
            "turns_taken": 0,
            "final_proof": None,
            "problem_name": problem_data.get("name", f"problem_{problem_idx}"),
            "error": str(e),
            "turn_results": []
        }
        return problem_idx, error_result


def solve_batch_problems(
    dataset: List[Dict],
    verifier_address: str,
    model: str,
    max_turns: int,
    max_gen_tokens: int,
    temperature: float,
    num_workers: int
) -> List[Dict]:
    """
    Solve multiple problems using batched multi-turn prompting.
    
    Args:
        dataset: List of problem data dictionaries
        verifier_address: Address of the verification server
        model: Model name to use
        max_turns: Maximum number of turns
        max_gen_tokens: Maximum generation tokens
        temperature: Model temperature
        num_workers: Number of workers for model queries
        
    Returns:
        List of result dictionaries
    """
    # Extract headers and theorems
    headers = [problem["header"] for problem in dataset]
    theorems = [problem["formal_statement"] for problem in dataset]
    
    # Run batch multi-turn prompting
    batch_result: BatchMultiTurnResult = multi_turn_prompt_batch(
        headers=headers,
        theorems=theorems,
        verifier_address=verifier_address,
        model=model,
        max_turns=max_turns,
        max_gen_tokens=max_gen_tokens,
        temperature=temperature,
        num_workers=num_workers
    )
    
    # Convert to the same format as individual results
    results = []
    for i, (problem, result) in enumerate(zip(dataset, batch_result.individual_results)):
        result_dict = {
            "success": result.success,
            "turns_taken": result.turns_taken,
            "final_proof": result.final_proof,
            "problem_name": problem.get("name", f"problem_{i}"),
            "turn_results": [verification.verdict for _, _, verification in result.turn_history]
        }
        results.append(result_dict)
    
    return results


def analyze_results(results: List[Dict], max_turns: int) -> Dict:
    """
    Analyze benchmark results and compute turn-by-turn performance.
    
    Args:
        results: List of result dictionaries
        max_turns: Maximum number of turns used
        
    Returns:
        Analysis dictionary with performance metrics
    """
    total_problems = len(results)
    successful_problems = sum(1 for r in results if r["success"])
    
    # Calculate turn-by-turn performance
    turn_performance = []
    
    for turn in range(1, max_turns + 1):
        solved_by_turn = 0
        for result in results:
            if result["success"] and result["turns_taken"] <= turn:
                solved_by_turn += 1
        
        performance = (solved_by_turn / total_problems) * 100
        turn_performance.append(performance)
    
    # Calculate distribution of turns taken for successful problems
    turns_distribution = {}
    for result in results:
        if result["success"]:
            turns = result["turns_taken"]
            turns_distribution[turns] = turns_distribution.get(turns, 0) + 1
    
    analysis = {
        "total_problems": total_problems,
        "successful_problems": successful_problems,
        "overall_success_rate": (successful_problems / total_problems) * 100,
        "turn_performance": turn_performance,
        "turns_distribution": turns_distribution,
        "avg_turns_for_success": np.mean([r["turns_taken"] for r in results if r["success"]]) if successful_problems > 0 else 0
    }
    
    return analysis


def plot_results(analysis: Dict, max_turns: int, output_dir: str, model: str):
    """
    Create performance plots.
    
    Args:
        analysis: Analysis results dictionary
        max_turns: Maximum number of turns
        output_dir: Directory to save plots
        model: Model name for plot titles
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Turn-by-turn cumulative performance
    plt.figure(figsize=(10, 6))
    turns = list(range(1, max_turns + 1))
    performance = analysis["turn_performance"]
    
    plt.plot(turns, performance, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Turn Number', fontsize=12)
    plt.ylabel('Cumulative Success Rate (%)', fontsize=12)
    plt.title(f'Multi-Turn Performance on minif2f ({model})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(turns)
    
    # Add annotations for key points
    final_performance = performance[-1]
    plt.annotate(f'Final: {final_performance:.1f}%', 
                xy=(max_turns, final_performance), 
                xytext=(max_turns-1, final_performance+2),
                fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'turn_performance_{model.replace("-", "_")}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Distribution of turns taken for successful problems
    if analysis["turns_distribution"]:
        plt.figure(figsize=(8, 6))
        turns_dist = analysis["turns_distribution"]
        turns = list(turns_dist.keys())
        counts = list(turns_dist.values())
        
        plt.bar(turns, counts, alpha=0.7, color='green')
        plt.xlabel('Turns Taken', fontsize=12)
        plt.ylabel('Number of Problems Solved', fontsize=12)
        plt.title(f'Distribution of Turns for Successful Problems ({model})', fontsize=14)
        plt.xticks(turns)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'turns_distribution_{model.replace("-", "_")}.png'), dpi=300, bbox_inches='tight')
        plt.close()

class BenchmarkConfig(Config):
    def __init__(self):
        super().__init__()
        self.dataset = "minif2f"
        self.split = "valid"
        self.verifier_address = "172.24.75.239:8001"
        #self.model = "gpt-4o"
        self.model = "gpt-4.1-mini"
        self.max_turns = 10 
        self.max_gen_tokens = 8000
        self.temperature = 1.0
        self.num_workers = 128
        self.max_problems = None
        self.output_dir = "./benchmark_results"
        self.debug = False

        self.batch_run = True  # Use batched processing for efficiency

    def debug_mode(self):
        self.debug = True

    def full_run(self):
        self.max_turns = 10

    

@main(BenchmarkConfig)
def main(args: BenchmarkConfig):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print(f"📊 Loading {args.dataset} {args.split} dataset...")
    dataset: List[Dict] = load_eval_dataset(args.dataset, args.split)
    
    # Apply debug mode or max_problems limit
    if args.debug:
        # Randomly sample 25 problems for debug mode
        random.seed(42)  # For reproducibility
        dataset = random.sample(dataset, min(25, len(dataset)))
        print(f"🐛 DEBUG MODE: Running on {len(dataset)} random problems")
    elif args.max_problems:
        dataset = dataset[:args.max_problems]
        print(f"🎯 Limited to first {len(dataset)} problems for testing")
    
    print(f"📝 Evaluating {len(dataset)} problems with max_turns={args.max_turns}")
    print(f"🤖 Using model: {args.model}")
    print(f"⚡ Using {args.num_workers} parallel workers")
    
    # Choose evaluation method based on batch_run flag
    start_time = time.time()
    
    if args.batch_run:
        print("🚀 Using BATCHED multi-turn processing (efficient)")
        # Use batched processing - much more efficient
        sorted_results = solve_batch_problems(
            dataset=dataset,
            verifier_address=args.verifier_address,
            model=args.model,
            max_turns=args.max_turns,
            max_gen_tokens=args.max_gen_tokens,
            temperature=args.temperature,
            num_workers=args.num_workers
        )
    else:
        print("🐌 Using INDIVIDUAL multi-turn processing (for comparison)")
        # Use individual subprocess processing - legacy method for comparison
        results = {}
        
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all tasks with named arguments
            future_to_idx = {}
            for i, problem in enumerate(dataset):
                future = executor.submit(
                    solve_single_problem,
                    problem_idx=i,
                    problem_data=problem,
                    verifier_address=args.verifier_address,
                    model=args.model,
                    max_turns=args.max_turns,
                    max_gen_tokens=args.max_gen_tokens,
                    temperature=args.temperature
                )
                future_to_idx[future] = i
            
            # Process results as they complete
            for future in tqdm(as_completed(future_to_idx), total=len(dataset), desc="Solving problems"):
                problem_idx, result = future.result()
                results[problem_idx] = result
        
        # Sort results by problem index
        sorted_results = [results[i] for i in range(len(dataset))]
    
    elapsed_time = time.time() - start_time
    print(f"⏱️  Completed evaluation in {elapsed_time:.1f} seconds")
    
    # Analyze results
    print("📈 Analyzing results...")
    analysis = analyze_results(sorted_results, args.max_turns)
    
    # Print summary
    print(f"\n🎯 BENCHMARK RESULTS SUMMARY")
    print(f"=" * 50)
    print(f"Total problems: {analysis['total_problems']}")
    print(f"Successfully solved: {analysis['successful_problems']}")
    print(f"Overall success rate: {analysis['overall_success_rate']:.1f}%")
    print(f"Average turns for success: {analysis['avg_turns_for_success']:.1f}")
    
    print(f"\n📊 Turn-by-turn performance:")
    for turn, perf in enumerate(analysis['turn_performance'], 1):
        print(f"  Turn {turn}: {perf:.1f}%")
    
    # Save detailed results

    # Create timestamp for the results dir and put everything in there, to the minute
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(results_dir, exist_ok=True)

    debug_suffix = "_debug" if args.debug else ""
    batch_suffix = "_batch" if args.batch_run else "_individual"
    results_filename = f"benchmark_{args.model.replace('-', '_')}_{args.max_turns}turns{debug_suffix}{batch_suffix}_{timestamp}.json"
    results_path = os.path.join(results_dir, results_filename)
    

    full_results = {
        "config": vars(args),
        "analysis": analysis,
        "detailed_results": sorted_results,
        "timestamp": timestamp,
        "elapsed_time": elapsed_time
    }
    
    with open(results_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"💾 Detailed results saved to: {results_path}")
    
    # Generate plots
    print("📈 Generating performance plots...")
    plot_results(analysis, args.max_turns, results_dir, args.model)
    print(f"📊 Plots saved to: {results_dir}")


if __name__ == "__main__":
    main()