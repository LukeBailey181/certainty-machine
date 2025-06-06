#!/usr/bin/env python3
"""
Test script for the iterative agentic mathlib retrieval tool.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to Python path
sys.path.append('.')

from certainty_machine.querying.tools import agentic_mathlib_retrieval, execute_custom_tool, get_or_create_rag_instance


def test_direct_function_call():
    """Test calling the agentic retrieval function directly with iterative refinement"""
    print("=" * 80)
    print("TESTING DIRECT FUNCTION CALL - ITERATIVE REFINEMENT")
    print("=" * 80)
    
    test_description = "continuous functions on compact sets"
    
    print(f"Testing description: '{test_description}'")
    print("Parameters: depth=2, k=3 (for faster testing)")
    print("-" * 60)
    
    try:
        result = agentic_mathlib_retrieval(
            description=test_description,
            depth=2,  # Use fewer iterations for faster testing
            k=3      # Get top 3 results
        )
        
        print("✅ Function executed successfully!")
        print("\nResult:")
        print(result)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def test_via_tool_execution():
    """Test calling the tool via the execute_custom_tool function"""
    print("\n" + "=" * 80)
    print("TESTING VIA TOOL EXECUTION - ITERATIVE REFINEMENT")
    print("=" * 80)
    
    test_description = "Banach fixed point theorem"
    
    print(f"Testing description: '{test_description}'")
    print("Parameters: depth=2, k=4")
    print("-" * 60)
    
    try:
        result = execute_custom_tool(
            tool_name="agentic_mathlib_retrieval",
            arguments={
                "description": test_description,
                "depth": 2,
                "k": 4
            }
        )
        
        print("✅ Tool execution successful!")
        print("\nResult:")
        print(result)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def test_with_full_api():
    """Test using the full API with tool calling"""
    print("\n" + "=" * 80)
    print("TESTING WITH FULL API - ITERATIVE REFINEMENT")
    print("=" * 80)
    
    try:
        from certainty_machine.querying.api import single_query_with_tools
        
        prompt = """
        I need to understand the mathematical concept of topological spaces and their properties. 
        Please use the agentic retrieval tool to find relevant theorems, definitions, and properties 
        related to topological spaces, continuity, and compactness. Use depth=3 and return the top 5 results.
        """
        
        print(f"Testing with prompt: {prompt[:100]}...")
        print("-" * 60)
        
        result = single_query_with_tools(
            prompt=prompt,
            model="gpt-4o-mini",
            max_gen_tokens=3000,
            temperature=0.7
        )
        
        print("✅ API call successful!")
        print(f"\nToken Usage:")
        print(f"  Input tokens: {result.input_token_count}")
        print(f"  Output tokens: {result.output_token_count}")
        print(f"  Tool calls made: {result.tool_calls_made}")
        print(f"  Error: {result.is_error}")
        
        if result.error_message:
            print(f"  Error message: {result.error_message}")
        
        print(f"\nResponse:")
        print(result.response_text)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def test_different_depths():
    """Test the iterative refinement with different depth values"""
    print("\n" + "=" * 80)
    print("TESTING DIFFERENT REFINEMENT DEPTHS")
    print("=" * 80)
    
    test_description = "group theory homomorphisms"
    
    depths_to_test = [1, 2, 3]
    
    for depth in depths_to_test:
        print(f"\n--- Testing with depth={depth} ---")
        
        try:
            result = agentic_mathlib_retrieval(
                description=test_description,
                depth=depth,
                k=2  # Keep k small for comparison
            )
            
            print(f"✅ Depth {depth} executed successfully!")
            
            # Extract just the search history part for comparison
            lines = result.split('\n')
            history_start = False
            for line in lines:
                if "Search Refinement History:" in line:
                    history_start = True
                elif history_start and line.startswith("## Top"):
                    break
                elif history_start and line.strip():
                    print(line)
                    
        except Exception as e:
            print(f"❌ Error with depth {depth}: {str(e)}")


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "=" * 80)
    print("TESTING EDGE CASES")
    print("=" * 80)
    
    # Test with very specific query
    print("--- Test 1: Very specific mathematical query ---")
    try:
        result = agentic_mathlib_retrieval(
            description="Cauchy-Schwarz inequality in inner product spaces",
            depth=2,
            k=3
        )
        print("✅ Specific query test passed")
    except Exception as e:
        print(f"❌ Specific query test failed: {str(e)}")
    
    # Test with broad query
    print("\n--- Test 2: Very broad mathematical query ---")
    try:
        result = agentic_mathlib_retrieval(
            description="algebra",
            depth=2,
            k=3
        )
        print("✅ Broad query test passed")
    except Exception as e:
        print(f"❌ Broad query test failed: {str(e)}")
    
    # Test with maximum parameters
    print("\n--- Test 3: Maximum parameters ---")
    try:
        result = agentic_mathlib_retrieval(
            description="limit theorems in probability theory",
            depth=1,  # Keep depth low for speed but test high k
            k=10
        )
        print("✅ Maximum parameters test passed")
    except Exception as e:
        print(f"❌ Maximum parameters test failed: {str(e)}")


def test_prerequisites():
    """Test if all prerequisites are available"""
    print("=" * 80)
    print("CHECKING PREREQUISITES")
    print("=" * 80)
    
    # Check OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    print(f"OpenAI API Key: {'✅ Set' if api_key else '❌ Not set'}")
    
    # Check if mathlib files exist
    embeddings_file = "mathlib_embeddings.pkl"
    tree_file = "mathlib_tree.json"
    
    print(f"Embeddings file ({embeddings_file}): {'✅ Exists' if os.path.exists(embeddings_file) else '❌ Missing'}")
    print(f"Tree file ({tree_file}): {'✅ Exists' if os.path.exists(tree_file) else '❌ Missing'}")
    
    # Try importing and initializing MathlibRAG (using shared instance)
    try:
        print("MathlibRAG import: ✅ Success")
        
        # Try initializing using the shared instance (this might take a while)
        rag = get_or_create_rag_instance()
        info = rag.get_file_info()
        print(f"RAG initialization: ✅ Success ({info['total_chunks']} chunks loaded)")
        
    except Exception as e:
        print(f"MathlibRAG: ❌ Error - {str(e)}")
    
    print()


if __name__ == "__main__":
    print("Testing Iterative Agentic Mathlib Retrieval Tool")
    print("This test will check prerequisites and run various tests of the iterative refinement tool.")
    print()
    
    # Run tests in order
    test_prerequisites()
    
    # Only run the actual tests if prerequisites are met
    if os.getenv('OPENAI_API_KEY') and os.path.exists("mathlib_embeddings.pkl"):
        test_direct_function_call()
        test_via_tool_execution()
        test_different_depths()
        test_edge_cases()
        test_with_full_api()
    else:
        print("⚠️  Skipping functional tests due to missing prerequisites.")
        print("Please ensure:")
        print("1. OPENAI_API_KEY environment variable is set")
        print("2. mathlib_embeddings.pkl file is present")
        print("3. mathlib_tree.json file is present") 