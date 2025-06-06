#!/usr/bin/env python3
"""
Test script for the new tool calling functionality.
"""

from certainty_machine.querying.api import single_query_with_tools

def test_python_interpreter():
    """Test the Python interpreter tool with o3-mini"""
    prompt = """
    You must write and execute Python code to solve this problem.
    
    Write a Python function that:
    1. Creates a Fibonacci sequence up to the 15th number
    2. Squares each number in the sequence
    3. Calculates and prints:
       - The sum of the squared numbers
       - The largest squared number
       - The average of all squared numbers
    4. Creates a line plot showing the growth of squared Fibonacci numbers
    
    Execute the code and show both the numerical results and the plot.

    """
    
    print("Testing Python interpreter tool with o3-mini...")
    print("-" * 60)
    result = single_query_with_tools(
        prompt=prompt,
        model="gpt-4o",  # Use a model with strong tool-calling skills
        max_gen_tokens=4000,
        temperature=0.1  # Lower temperature encourages deterministic tool usage
    )
    
    print(f"\nToken Usage:")
    print(f"  Input tokens: {result.input_token_count}")
    print(f"  Output tokens: {result.output_token_count}")
    print(f"  Reasoning tokens: {result.reasoning_token_count}")
    print(f"  Tool calls made: {result.tool_calls_made}")
    print(f"  Error: {result.is_error}")
    if result.error_message:
        print(f"  Error message: {result.error_message}")


def test_simple_calculation():
    """Test a simpler calculation to verify basic functionality"""
    
    prompt = "Please calculate 15 * 27 + 100 / 4. You can use the calculator tool or write Python code, whichever you prefer."
    
    print("\n\nTesting simple calculation...")
    print("-" * 60)
    
    result = single_query_with_tools(
        prompt=prompt,
        model="gpt-4o-mini",
        max_gen_tokens=2000,
        temperature=0.7
    )
    
    print(f"\nToken Usage:")
    print(f"  Input tokens: {result.input_token_count}")
    print(f"  Output tokens: {result.output_token_count}")
    print(f"  Reasoning tokens: {result.reasoning_token_count}")
    print(f"  Tool calls made: {result.tool_calls_made}")
    print(f"  Error: {result.is_error}")
    if result.error_message:
        print(f"  Error message: {result.error_message}")


if __name__ == "__main__":
    test_simple_calculation()
    test_python_interpreter() 