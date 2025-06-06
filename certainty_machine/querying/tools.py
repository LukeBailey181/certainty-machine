# Tools configuration for OpenAI function calling
# This module defines the tools available to the models and handles tool execution

from typing import Dict, Callable, Any, List
import json
import re
import os
from openai import OpenAI


# Built-in OpenAI tools that don't require local execution
# Note: code_interpreter is passed separately in the API call, not in tools array
BUILTIN_TOOLS = []

## problem finder 


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
