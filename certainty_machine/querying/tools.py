# Tools configuration for OpenAI function calling
# This module defines the tools available to the models and handles tool execution

from typing import Dict, Callable, Any, List
import json
import re
import os
from openai import OpenAI

import dotenv
dotenv.load_dotenv()

# Built-in OpenAI tools that don't require local execution
# Note: code_interpreter is passed separately in the API call, not in tools array
BUILTIN_TOOLS = []

# Global variable to cache the MathlibRAG instance
_rag_instance = None


def get_or_create_rag_instance():
    """
    Get or create the cached MathlibRAG instance.
    This ensures we only load the embeddings once across all tool calls.
    
    Returns:
        MathlibRAG instance
    """
    global _rag_instance
    
    if _rag_instance is None:
        import sys
        
        # Add the parent directory to the path to import query_mathlib 
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        
        from query_mathlib import MathlibRAG
        print("Initializing MathlibRAG (this may take time)...")
        _rag_instance = MathlibRAG()
        print("✅ MathlibRAG initialized successfully")
    
    return _rag_instance


def agentic_mathlib_retrieval(description: str, num_queries: int = 3, results_per_query: int = 3) -> str:
    """
    Agentic retrieval system that generates optimal search queries and retrieves mathematical context.
    Similar to OpenAI's deep research feature, but for mathematical theorems and concepts.
    
    Args:
        description: Text description of what theorem, lemma, or Lean context is needed
        num_queries: Number of search queries to generate (default: 3)
        results_per_query: Number of results to retrieve per query (default: 3)
        
    Returns:
        Formatted string containing retrieved context organized by query
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Generate optimal search queries using OpenAI
        query_generation_prompt = f"""You are an expert in mathematics and formal verification. Given a description of what mathematical content someone is looking for, generate {num_queries} optimal search queries that would help find the most relevant theorems, lemmas, definitions, or proofs.

The queries should be:
1. Specific and targeted to mathematical concepts
2. Use appropriate mathematical terminology
3. Cover different aspects or approaches to the topic
4. Be suitable for searching through mathematical literature and formal proofs

Description: {description}

Generate exactly {num_queries} search queries, one per line, without numbering or bullets:"""

        # Call OpenAI to generate queries
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a mathematical research assistant who excels at creating targeted search queries for finding specific mathematical content, theorems, and formal proofs."
                },
                {
                    "role": "user", 
                    "content": query_generation_prompt
                }
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        # Extract and clean the generated queries
        generated_text = response.choices[0].message.content.strip()
        queries = [q.strip() for q in generated_text.split('\n') if q.strip()]
        
        # Ensure we have the right number of queries
        queries = queries[:num_queries]
        
        # Initialize the RAG system (using cached instance if available)
        try:
            rag = get_or_create_rag_instance()
        except ImportError as e:
            return f"Error: Could not import MathlibRAG. Make sure query_mathlib.py is accessible. Error: {str(e)}"
        except Exception as e:
            return f"Error: Could not initialize MathlibRAG. Make sure the embedding files exist. Error: {str(e)}"
        
        # Perform retrieval for each query
        all_results = []
        
        for i, query in enumerate(queries, 1):
            try:
                results = rag.similarity_search(query, k=results_per_query)
                all_results.append({
                    'query': query,
                    'query_number': i,
                    'results': results
                })
            except Exception as e:
                all_results.append({
                    'query': query,
                    'query_number': i,
                    'results': [],
                    'error': str(e)
                })
        
        # Format the results nicely
        formatted_output = f"# Agentic Retrieval Results for: {description}\n\n"
        formatted_output += f"Generated {len(queries)} targeted search queries and retrieved relevant mathematical context.\n\n"
        
        for result_set in all_results:
            query_num = result_set['query_number']
            query = result_set['query']
            
            formatted_output += f"## Query {query_num}: \"{query}\"\n\n"
            
            if 'error' in result_set:
                formatted_output += f"❌ Error retrieving results: {result_set['error']}\n\n"
                continue
            
            results = result_set['results']
            if not results:
                formatted_output += "No results found for this query.\n\n"
                continue
            
            for j, (content, score, metadata) in enumerate(results, 1):
                file_name = metadata.get('file_name', 'Unknown')
                formatted_output += f"### Result {query_num}.{j} (Similarity: {score:.3f})\n"
                formatted_output += f"**Source:** `{file_name}`\n\n"
                formatted_output += f"```lean\n{content}\n```\n\n"
        
        formatted_output += "---\n\n"
        formatted_output += "*Results retrieved using agentic search with AI-generated queries optimized for mathematical content discovery.*"
        
        return formatted_output
        
    except Exception as e:
        return f"Error in agentic retrieval: {str(e)}\n\nPlease ensure:\n1. OpenAI API key is set\n2. MathlibRAG embeddings are available\n3. query_mathlib.py is accessible"


## html diagram generator
## given some text description, return html code for a diagram that can explain the model's reasoning. 

def html_diagram_generator(description: str) -> str:
    """
    Generate HTML code for a diagram based on text description using OpenAI.
    
    Args:
        description: Text description of the diagram to create
        
    Returns:
        HTML code that renders the requested diagram
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Create a prompt for the diagram generation
        prompt = f"""Generate HTML code for a diagram based on this description: "{description}"

Requirements:
- Return ONLY the HTML code, no explanations or markdown
- Use inline CSS styling for all elements
- Create a responsive, modern-looking diagram
- Use SVG, CSS, or HTML elements as appropriate
- Make it visually appealing with good colors and typography
- Ensure the diagram is self-contained (no external dependencies)
- The HTML should be ready to embed directly into a web page

Generate the HTML code for: {description}"""

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using a cost-effective model for HTML generation
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert web developer who creates beautiful, interactive diagrams using HTML, CSS, and SVG. Return only clean HTML code without any markdown formatting or explanations."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            max_tokens=2000,
            temperature=0.7
        )
        
        # Extract the HTML code from the response
        html_code = response.choices[0].message.content.strip()
        
        # Clean up any potential markdown formatting
        if html_code.startswith('```html'):
            html_code = html_code[7:]
        if html_code.endswith('```'):
            html_code = html_code[:-3]
        
        return html_code.strip()
        
    except Exception as e:
        return f"<div style='color: red; padding: 20px; border: 1px solid red; border-radius: 5px; margin: 20px; font-family: Arial, sans-serif;'>Error generating diagram: {str(e)}<br><br>Please check your OpenAI API key and ensure it's set in the OPENAI_API_KEY environment variable.</div>"

# Example calculator function for testing
def calculator(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")
        
    Returns:
        String result of the calculation
    """
    try:
        # Use eval safely for basic math operations
        # In production, you'd want a proper math expression parser
        allowed_names = {
            k: v for k, v in __builtins__.items() 
            if k in ['abs', 'round', 'min', 'max', 'sum', 'pow']
        }
        allowed_names.update({'__builtins__': {}})
        
        result = eval(expression, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


# Custom tools that require local execution
CUSTOM_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "agentic_mathlib_retrieval",
            "description": "Agentic retrieval system for mathematical content. Generates multiple optimal search queries using AI and retrieves relevant theorems, lemmas, definitions, or proofs from the Mathlib database. Similar to OpenAI's deep research feature but specialized for mathematics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Text description of what mathematical theorem, lemma, definition, or concept you need. Be specific about the mathematical area and what you're looking for (e.g., 'continuous functions on compact sets', 'Banach fixed point theorem', 'group homomorphism properties')"
                    },
                    "num_queries": {
                        "type": "integer",
                        "description": "Number of different search queries to generate (default: 3, range: 1-5)",
                        "minimum": 1,
                        "maximum": 5
                    },
                    "results_per_query": {
                        "type": "integer", 
                        "description": "Number of results to retrieve per query (default: 3, range: 1-10)",
                        "minimum": 1,
                        "maximum": 10
                    }
                },
                "required": ["description"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "calculator",
            "description": "Perform basic mathematical calculations. Use this for simple arithmetic operations like addition, subtraction, multiplication, division, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4', '(10 + 5) / 3')"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "html_diagram_generator",
            "description": "Generate HTML code for diagrams based on text descriptions. Supports flowcharts, mind maps, organizational charts, timelines, Venn diagrams, and general concept diagrams.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Text description of the diagram to create (e.g., 'flowchart showing user login process', 'mind map about machine learning concepts', 'timeline of project phases')"
                    }
                },
                "required": ["description"]
            }
        }
    }
]

# Mapping of custom tool names to their implementation functions
TOOL_MAPPING: Dict[str, Callable] = {
    "agentic_mathlib_retrieval": agentic_mathlib_retrieval,
    "calculator": calculator,
    "html_diagram_generator": html_diagram_generator
}


def get_all_tools() -> List[Dict[str, Any]]:
    """
    Returns all available tools (both built-in and custom).
    
    Returns:
        List of tool definitions in OpenAI format
    """
    tools = []
    
    # Add built-in tools
    for builtin_tool in BUILTIN_TOOLS:
        tools.append(builtin_tool)
    
    # Add custom tools
    tools.extend(CUSTOM_TOOLS)
    
    return tools


def execute_custom_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Execute a custom tool function.
    
    Args:
        tool_name: Name of the tool to execute
        arguments: Arguments to pass to the tool function
        
    Returns:
        String result from the tool execution
        
    Raises:
        ValueError: If tool_name is not found in TOOL_MAPPING
    """
    if tool_name not in TOOL_MAPPING:
        raise ValueError(f"Tool '{tool_name}' not found in tool mapping")
    
    tool_function = TOOL_MAPPING[tool_name]
    return tool_function(**arguments)


def is_builtin_tool(tool_call) -> bool:
    """
    Check if a tool call is for a built-in OpenAI tool.
    
    Args:
        tool_call: Tool call object from OpenAI response
        
    Returns:
        True if it's a built-in tool, False if it's a custom tool
    """
    # Built-in tools have different structure than function calls
    return hasattr(tool_call, 'type') and tool_call.type == 'code_interpreter'


# Example of how to add custom tools (for future reference):
#
# def example_custom_tool(query: str) -> str:
#     """Example custom tool implementation"""
#     return f"Processed query: {query}"
#
# # Add to CUSTOM_TOOLS:
# CUSTOM_TOOLS.append({
#     "type": "function",
#     "function": {
#         "name": "example_custom_tool",
#         "description": "An example custom tool",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "query": {"type": "string", "description": "Query to process"}
#             },
#             "required": ["query"]
#         }
#     }
# })
#
# # Add to TOOL_MAPPING:
# TOOL_MAPPING["example_custom_tool"] = example_custom_tool
