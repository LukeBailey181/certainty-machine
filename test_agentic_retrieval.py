#!/usr/bin/env python3
"""
Test script for the agentic mathlib retrieval tool.
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
    """Test calling the agentic retrieval function directly"""
    print("=" * 80)
    print("TESTING DIRECT FUNCTION CALL")
    print("=" * 80)
    
    test_description = "continuous functions on compact sets"
    
    print(f"Testing description: '{test_description}'")
    print("-" * 60)
    
    try:
        result = agentic_mathlib_retrieval(
            description=test_description,
            num_queries=2,  # Use fewer queries for faster testing
            results_per_query=2
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
    print("TESTING VIA TOOL EXECUTION")
    print("=" * 80)
    
    test_description = "Banach fixed point theorem"
    
    print(f"Testing description: '{test_description}'")
    print("-" * 60)
    
    try:
        result = execute_custom_tool(
            tool_name="agentic_mathlib_retrieval",
            arguments={
                "description": test_description,
                "num_queries": 2,
                "results_per_query": 2
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
    print("TESTING WITH FULL API")
    print("=" * 80)
    
    try:
        from certainty_machine.querying.api import single_query_with_tools
        
        prompt = """
        I need to understand the mathematical concept of topological spaces and their properties. 
        Please use the agentic retrieval tool to find relevant theorems, definitions, and properties 
        related to topological spaces, continuity, and compactness.
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
    print("Testing Agentic Mathlib Retrieval Tool")
    print("This test will check prerequisites and run various tests of the tool.")
    print()
    
    # Run tests in order
    test_prerequisites()
    
    # Only run the actual tests if prerequisites are met
    if os.getenv('OPENAI_API_KEY') and os.path.exists("mathlib_embeddings.pkl"):
        test_direct_function_call()
        test_via_tool_execution()
        test_with_full_api()
    else:
        print("⚠️  Skipping functional tests due to missing prerequisites.")
        print("Please ensure:")
        print("1. OPENAI_API_KEY environment variable is set")
        print("2. mathlib_embeddings.pkl file is present")
        print("3. mathlib_tree.json file is present") 