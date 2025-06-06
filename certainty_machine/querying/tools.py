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
        
        from certainty_machine.querying.rag.query_mathlib import MathlibRAG
        print("Initializing MathlibRAG (this may take time)...")
        _rag_instance = MathlibRAG()
        print("✅ MathlibRAG initialized successfully")
    
    return _rag_instance


def agentic_mathlib_retrieval(description: str, depth: int = 3, k: int = 5) -> str:
    """
    Iterative agentic retrieval system that refines search queries based on context quality.
    Generates an initial query, evaluates results, and iteratively refines the search.
    
    Args:
        description: Text description of what theorem, lemma, or Lean context is needed
        depth: Number of refinement iterations to perform (default: 3)
        k: Total number of highest-scoring results to return at the end (default: 5)
        
    Returns:
        Formatted string containing the k best retrieved results across all iterations
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Initialize the RAG system (using cached instance if available)
        try:
            rag = get_or_create_rag_instance()
        except ImportError as e:
            return f"Error: Could not import MathlibRAG. Make sure query_mathlib.py is accessible. Error: {str(e)}"
        except Exception as e:
            return f"Error: Could not initialize MathlibRAG. Make sure the embedding files exist. Error: {str(e)}"
        
        # Store all results across iterations
        all_results = []
        current_query = None
        search_history = []
        
        for iteration in range(depth):
            # Generate or refine query
            if iteration == 0:
                # Generate initial query
                initial_query_prompt = f"""You are an expert in mathematics and formal verification. Given a description of what mathematical content someone is looking for, generate ONE optimal search query that would help find the most relevant theorems, lemmas, definitions, or proofs.

The query should be:
1. Specific and targeted to mathematical concepts
2. Use appropriate mathematical terminology
3. Be suitable for searching through mathematical literature and formal proofs

Description: {description}

Generate exactly one search query (no numbering, bullets, or explanation):"""

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a mathematical research assistant who excels at creating targeted search queries for finding specific mathematical content, theorems, and formal proofs."
                        },
                        {
                            "role": "user", 
                            "content": initial_query_prompt
                        }
                    ],
                    max_tokens=150,
                    temperature=0.7
                )
                
                current_query = response.choices[0].message.content.strip()
                
            else:
                # Refine query based on previous results and quality assessment
                refinement_prompt = f"""You are refining a search query for mathematical content. 

Original request: {description}

Previous search query: {current_query}

Previous search results summary:
{chr(10).join([f"- {result['content'][:100]}..." for result in search_history[-1]['top_results'][:3]])}

Quality assessment of previous results: {search_history[-1]['quality_assessment']}

Based on the original request, previous query, results, and quality assessment, generate a refined search query that would better find the mathematical content being sought. The refined query should address any gaps or issues identified in the quality assessment.

Generate exactly one refined search query (no numbering, bullets, or explanation):"""

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a mathematical research assistant who excels at iteratively refining search queries based on result quality to find specific mathematical content."
                        },
                        {
                            "role": "user", 
                            "content": refinement_prompt
                        }
                    ],
                    max_tokens=150,
                    temperature=0.5
                )
                
                current_query = response.choices[0].message.content.strip()
            
            # Execute RAG search with current query
            try:
                results = rag.similarity_search(current_query, k=10)  # Get more results for quality assessment
                
                if not results:
                    search_history.append({
                        'iteration': iteration + 1,
                        'query': current_query,
                        'top_results': [],
                        'quality_assessment': "No results found",
                        'should_continue': iteration < depth - 1
                    })
                    continue
                
                # Add results to our collection
                for content, score, metadata in results:
                    all_results.append({
                        'content': content,
                        'score': score,
                        'metadata': metadata,
                        'query': current_query,
                        'iteration': iteration + 1
                    })
                
                # Assess quality of results
                top_results = results[:5]  # Assess top 5 results
                results_summary = "\n\n".join([
                    f"Result {i+1} (Score: {score:.3f}):\n{content[:300]}..."
                    for i, (content, score, metadata) in enumerate(top_results)
                ])
                
                quality_prompt = f"""Assess the quality and relevance of these search results for the following mathematical request:

Original request: {description}
Search query used: {current_query}

Retrieved results:
{results_summary}

Please provide a quality assessment that addresses:
1. How well do these results match the original request?
2. Are the mathematical concepts relevant and accurate?
3. What aspects of the original request are still missing or could be better addressed?
4. Overall quality score from 1-10 (where 10 is perfect match)

Provide a concise assessment (2-3 sentences):"""

                quality_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a mathematical expert who evaluates the quality and relevance of search results for mathematical queries."
                        },
                        {
                            "role": "user", 
                            "content": quality_prompt
                        }
                    ],
                    max_tokens=200,
                    temperature=0.3
                )
                
                quality_assessment = quality_response.choices[0].message.content.strip()
                
                # Store iteration results
                search_history.append({
                    'iteration': iteration + 1,
                    'query': current_query,
                    'top_results': [{'content': content, 'score': score, 'metadata': metadata} 
                                  for content, score, metadata in top_results],
                    'quality_assessment': quality_assessment,
                    'should_continue': iteration < depth - 1
                })
                
            except Exception as e:
                search_history.append({
                    'iteration': iteration + 1,
                    'query': current_query,
                    'top_results': [],
                    'quality_assessment': f"Error during search: {str(e)}",
                    'should_continue': iteration < depth - 1
                })
        
        # Select top k results by score across all iterations
        if not all_results:
            return f"# Iterative Retrieval Results for: {description}\n\nNo results found across {depth} iterations.\n\n## Search History:\n" + \
                   "\n".join([f"Iteration {h['iteration']}: {h['query']} - {h['quality_assessment']}" for h in search_history])
        
        # Sort all results by score (descending) and take top k
        all_results.sort(key=lambda x: x['score'], reverse=True)
        top_k_results = all_results[:k]
        
        # Format the final output
        formatted_output = f"# Iterative Agentic Retrieval Results for: {description}\n\n"
        formatted_output += f"Performed {depth} iterations of query refinement and selected top {k} results by relevance score.\n\n"
        
        # Add search history
        formatted_output += "## Search Refinement History:\n\n"
        for i, history in enumerate(search_history, 1):
            formatted_output += f"**Iteration {i}:** `{history['query']}`\n"
            formatted_output += f"*Quality Assessment:* {history['quality_assessment']}\n\n"
        
        # Add top k results
        formatted_output += f"## Top {k} Results (Highest Relevance Scores):\n\n"
        
        for i, result in enumerate(top_k_results, 1):
            file_name = result['metadata'].get('file_name', 'Unknown')
            formatted_output += f"### Result {i} (Score: {result['score']:.3f})\n"
            formatted_output += f"**Source:** `{file_name}`\n"
            formatted_output += f"**Found in iteration:** {result['iteration']}\n"
            formatted_output += f"**Query:** `{result['query']}`\n\n"
            formatted_output += f"```lean\n{result['content']}\n```\n\n"
        
        formatted_output += "---\n\n"
        formatted_output += f"*Results retrieved using iterative agentic search with {depth} refinement cycles, optimized for mathematical content discovery.*"
        
        return formatted_output
        
    except Exception as e:
        return f"Error in iterative agentic retrieval: {str(e)}\n\nPlease ensure:\n1. OpenAI API key is set\n2. MathlibRAG embeddings are available\n3. query_mathlib.py is accessible"


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
            "description": "Iterative agentic retrieval system for mathematical content. Generates an initial search query, evaluates result quality, and iteratively refines the search based on context assessment. Returns the k highest-scoring results across all iterations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Text description of what mathematical theorem, lemma, definition, or concept you need. Be specific about the mathematical area and what you're looking for (e.g., 'continuous functions on compact sets', 'Banach fixed point theorem', 'group homomorphism properties')"
                    }
                    ## hard code depth and k for now
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
