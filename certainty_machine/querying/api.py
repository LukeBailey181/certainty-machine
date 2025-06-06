import os
import tqdm
import threading
import time
import random  # Import random for jitter calculation
import json  # Add json import for tool handling
from typing import List, cast, Optional, Dict, Any, Set  # Add additional type imports
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import anthropic
import openai
from google.genai import types
from anthropic.resources.messages.messages import Messages as AnthropicMessages
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


# Import our tools module
from .tools import get_all_tools, execute_custom_tool, is_builtin_tool

# GOOGLE_CLIENT = genai.Client(vertexai=True, project="", location="us-central1")
GOOGLE_CLIENT = None
ANTHROPIC_CLIENT = anthropic.Anthropic()
OPENAI_CLIENT = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


GOOGLE_MODELS: List[str] = ["gemini-2.0-flash-001"]
OPENAI_MODELS = [
    "o4-mini",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
    "o3-mini",
    "o4-mini-2025-04-16",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano-2025-04-14",
]
OPENAI_MODELS_NO_TEMP = ["o1", "o3-mini", "o4-mini-2025-04-16"]
ANTHROPIC_MODELS = [
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
]


@dataclass
class QueryResult:
    response_text: str
    input_token_count: int
    output_token_count: int
    is_error: bool


@dataclass
class ToolQueryResult:
    """Extended result class for tool-enabled queries"""
    response_text: str
    input_token_count: int
    output_token_count: int
    reasoning_token_count: int  # For reasoning models like o3
    tool_calls_made: int
    is_error: bool
    error_message: Optional[str] = None


# ------------------------------
# Parsing functions
# ------------------------------


def parse_google_response(response: types.GenerateContentResponse) -> QueryResult:
    output = QueryResult(
        response_text=response.text,
        input_token_count=response.usage_metadata.prompt_token_count
        if response.usage_metadata
        else 0,
        output_token_count=response.usage_metadata.output_token_count
        if response.usage_metadata
        else 0,
        is_error=False,
    )

    return output


def parse_anthropic_response(response: AnthropicMessages) -> QueryResult:
    output = QueryResult(
        response_text=response.content[0].text,
        input_token_count=response.usage.input_tokens,
        output_token_count=response.usage.output_tokens,
        is_error=False,
    )
    return output


def parse_openai_response(response: openai.ChatCompletion) -> QueryResult:
    output = QueryResult(
        response_text=response.choices[0].message.content,
        input_token_count=response.usage.prompt_tokens,
        output_token_count=response.usage.completion_tokens,
        is_error=False,
    )
    return output


# ------------------------------
# Query functions
# ------------------------------


def single_query_google(
    prompt: str, model: str, max_gen_tokens: int, temperature: float
) -> QueryResult:
    raise NotImplementedError("Google API not supported right now")
    response = GOOGLE_CLIENT.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            maxOutputTokens=max_gen_tokens,
            temperature=temperature,
        ),
    )
    return parse_google_response(response)


def single_query_anthropic(
    prompt: str, model: str, max_gen_tokens: int, temperature: float
) -> QueryResult:
    response = ANTHROPIC_CLIENT.messages.create(
        model=model,
        max_tokens=max_gen_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return parse_anthropic_response(response)


def single_query_openai(
    prompt: str, model: str, max_gen_tokens: int, temperature: float
) -> QueryResult:
    temp: float | None
    if model in OPENAI_MODELS_NO_TEMP:
        # Some openai models don't support temperature ¯\_(ツ)_/¯
        temp = 1.0
    else:
        temp = temperature

    response = OPENAI_CLIENT.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_gen_tokens,
        temperature=temp,
    )

    return parse_openai_response(response)


def single_query_with_tools(
    prompt: str, 
    model: str, 
    max_gen_tokens: int, 
    temperature: float,
    tools: Optional[List[Dict[str, Any]]] = None
) -> ToolQueryResult:
    """
    Query an OpenAI model with tool calling support using the Assistants API.
    
    Args:
        prompt: The user prompt
        model: Model name (should be o3, o4-mini, etc. for best tool support)
        max_gen_tokens: Maximum tokens to generate (not directly supported by Assistants API)
        temperature: Temperature setting (ignored for reasoning models)
        tools: List of tools to make available. If None, uses all available tools.
        
    Returns:
        ToolQueryResult with response and metadata about tool usage
    """
    try:
        # Use all available tools if none specified
        if tools is None:
            tools = get_all_tools()
        
        # Set temperature appropriately
        temp: float | None
        if model in OPENAI_MODELS_NO_TEMP:
            temp = None  # Reasoning models ignore temperature
        else:
            temp = temperature
        
        # Track token usage across all calls
        total_input_tokens = 0
        total_output_tokens = 0
        reasoning_tokens = 0
        tool_calls_made = 0
        
        # Variables to keep track of the last cumulative counts so we only add the *delta*
        prev_prompt_tokens: int = 0
        prev_completion_tokens: int = 0
        prev_reasoning_tokens: int = 0  # For o-series models that expose this
        
        # Build tools array properly
        tools_array = []
        
        # Add code interpreter (always available for now)
        tools_array.append({"type": "code_interpreter"})
        
        # Add any custom function tools
        if tools:
            tools_array.extend(tools)
        
        # Step 1: Create an Assistant
        assistant_params = {
            "name": "Query Assistant",
            "instructions": prompt,
            "model": model,
            "tools": tools_array
        }
        
        # Add temperature if supported
        if temp is not None:
            assistant_params["temperature"] = temp
            
        assistant = OPENAI_CLIENT.beta.assistants.create(**assistant_params)
        
        try:
            # Step 2: Create a Thread
            thread = OPENAI_CLIENT.beta.threads.create()
            
            # Step 3: Add initial message to the Thread
            message = OPENAI_CLIENT.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=prompt
            )
            
            # Step 4: Run the Assistant
            run_params = {
                "thread_id": thread.id,
                "assistant_id": assistant.id
            }
            
            run = OPENAI_CLIENT.beta.threads.runs.create(**run_params)
            
            # Step 5: Wait for completion and handle tool calls
            max_iterations = 100  # Prevent infinite loops
            last_printed_content = ""  # Track what we've already printed
            printed_ci_call_ids: Set[str] = set()  # Track Code Interpreter calls we've already displayed
            downloaded_file_ids: Set[str] = set()  # Track file_ids already saved locally
            
            for iteration in range(max_iterations):
                # Poll for run completion
                run_status = OPENAI_CLIENT.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                
                # Get the latest assistant message and stream new content
                messages = OPENAI_CLIENT.beta.threads.messages.list(thread_id=thread.id)
                current_assistant_content = ""
                
                # Find the most recent assistant message
                for message in messages.data:
                    if message.role == "assistant":
                        for content in message.content:
                            if content.type == "text":
                                current_assistant_content = content.text.value
                                break
                        break
                
                # Print only the new content (streaming effect)
                if current_assistant_content and current_assistant_content != last_printed_content:
                    if iteration == 0:
                        print("Assistant: ", end="", flush=True)
                    
                    # Print only the new part
                    new_content = current_assistant_content[len(last_printed_content):]
                    print(new_content, end="", flush=True)
                    last_printed_content = current_assistant_content
                
                # Track token usage if available
                if hasattr(run_status, 'usage') and run_status.usage:
                    # The API returns cumulative counts.  Only add the *new* tokens since the last poll.
                    curr_prompt_tokens = getattr(run_status.usage, 'prompt_tokens', 0)
                    curr_completion_tokens = getattr(run_status.usage, 'completion_tokens', 0)

                    delta_prompt = curr_prompt_tokens - prev_prompt_tokens
                    delta_completion = curr_completion_tokens - prev_completion_tokens

                    if delta_prompt > 0:
                        total_input_tokens += delta_prompt
                    if delta_completion > 0:
                        total_output_tokens += delta_completion

                    prev_prompt_tokens = curr_prompt_tokens
                    prev_completion_tokens = curr_completion_tokens

                    # Track reasoning tokens (if the model provides them)
                    if (
                        hasattr(run_status.usage, 'completion_tokens_details') and
                        run_status.usage.completion_tokens_details is not None
                    ):
                        curr_reasoning = getattr(run_status.usage.completion_tokens_details, 'reasoning_tokens', 0)
                        delta_reasoning = curr_reasoning - prev_reasoning_tokens
                        if delta_reasoning > 0:
                            reasoning_tokens += delta_reasoning
                        prev_reasoning_tokens = curr_reasoning
                
                # --- Stream Code Interpreter code as soon as it's available ---
                try:
                    run_steps = OPENAI_CLIENT.beta.threads.runs.steps.list(
                        thread_id=thread.id,
                        run_id=run.id
                    )

                    # Iterate chronologically so multi-chunk scripts appear in order
                    for step in sorted(
                        run_steps.data,
                        key=lambda s: getattr(s, "created_at", 0)
                    ):
                        if (
                            getattr(step, "step_details", None) is not None and
                            step.step_details.type == "tool_calls"
                        ):
                            for tc in step.step_details.tool_calls:
                                if tc.type == "code_interpreter" and tc.id not in printed_ci_call_ids:
                                    code_input = getattr(tc.code_interpreter, "input", "")
                                    if code_input:
                                        print("\n[Code Interpreter]:\n")
                                        print(code_input)
                                        print("<End of code interpreter>\n", flush=True)
                                        printed_ci_call_ids.add(tc.id)

                                    # Attempt to download any image or file outputs (may become available later)
                                    try:
                                        for output in getattr(tc.code_interpreter, "outputs", []):
                                            file_id: Optional[str] = None

                                            if output.type == "image" and hasattr(output, "image"):
                                                file_id = getattr(output.image, "file_id", None)
                                            elif output.type == "file" and hasattr(output, "file"):
                                                file_id = getattr(output.file, "file_id", None)

                                            if file_id and file_id not in downloaded_file_ids:
                                                # Ensure output directory exists
                                                os.makedirs("ci_files", exist_ok=True)

                                                # Determine filename (attempt to use metadata filename if available)
                                                filename = f"{file_id}"
                                                try:
                                                    file_meta = OPENAI_CLIENT.files.retrieve(file_id)
                                                    # prefer display filename if field exists
                                                    filename_meta = getattr(file_meta, "filename", None)
                                                    if filename_meta:
                                                        filename = filename_meta
                                                except Exception:
                                                    pass

                                                # Infer extension for images if not present
                                                if output.type == "image" and not os.path.splitext(filename)[1]:
                                                    filename += ".png"

                                                dest_path = os.path.join("ci_files", filename)

                                                try:
                                                    file_content = OPENAI_CLIENT.files.content(file_id)
                                                    with open(dest_path, "wb") as f:
                                                        f.write(file_content.read())

                                                    print(f"[Saved file to {dest_path}]", flush=True)
                                                    downloaded_file_ids.add(file_id)
                                                except Exception as e:
                                                    print(f"[Failed to save file {file_id}: {e}]", flush=True)
                                    except Exception as e:
                                        # Non-fatal; just log
                                        print(f"[Error processing CI outputs: {e}]", flush=True)
                except Exception:
                    # Ignore errors (e.g., steps not ready yet) and continue polling
                    pass
                
                if run_status.status == "completed":
                    # Run completed successfully
                    print("\n")  # Add final newline when complete
                    break
                elif run_status.status == "failed":
                    # Run failed
                    error_msg = f"Run failed: {run_status.last_error.message if run_status.last_error else 'Unknown error'}"
                    return ToolQueryResult(
                        response_text="",
                        input_token_count=total_input_tokens,
                        output_token_count=total_output_tokens,
                        reasoning_token_count=reasoning_tokens,
                        tool_calls_made=tool_calls_made,
                        is_error=True,
                        error_message=error_msg
                    )
                elif run_status.status == "requires_action":
                    # Handle required actions (tool calls)
                    required_action = run_status.required_action
                    if required_action and required_action.type == "submit_tool_outputs":
                        tool_outputs = []
                        
                        # Show that we're executing tools
                        print("\n\n[Executing tools...]", end="", flush=True)
                        
                        for tool_call in required_action.submit_tool_outputs.tool_calls:
                            tool_calls_made += 1
                            
                            try:
                                if tool_call.type == "function":
                                    # This is a custom function call
                                    function_name = tool_call.function.name
                                    function_args = json.loads(tool_call.function.arguments)
                                    
                                    # Show which tool is being executed
                                    print(f"\n  ├─ Using {function_name}...", end="", flush=True)
                                    
                                    # Execute the custom tool
                                    tool_result = execute_custom_tool(function_name, function_args)
                                    print(" ✓", flush=True)
                                    
                                    tool_outputs.append({
                                        "tool_call_id": tool_call.id,
                                        "output": tool_result
                                    })
                                elif tool_call.type == "code_interpreter":
                                    # Show code interpreter usage
                                    print(f"\n  ├─ Running code...", end="", flush=True)
                                    # Note: Built-in tools like code_interpreter are handled automatically
                                    # by the Assistants API, so we don't need to handle them here
                                    print(" ✓", flush=True)
                                    
                            except Exception as e:
                                # Handle tool execution errors
                                print(f" ✗ (error: {str(e)})", flush=True)
                                error_msg = f"Error executing tool {tool_call.function.name if tool_call.type == 'function' else tool_call.type}: {str(e)}"
                                tool_outputs.append({
                                    "tool_call_id": tool_call.id,
                                    "output": error_msg
                                })
                        
                        print("\n[Tools completed, continuing response...]\n", end="", flush=True)
                        
                        # Submit tool outputs
                        if tool_outputs:
                            OPENAI_CLIENT.beta.threads.runs.submit_tool_outputs(
                                thread_id=thread.id,
                                run_id=run.id,
                                tool_outputs=tool_outputs
                            )
                        
                        # Continue polling
                        continue
                elif run_status.status in ["queued", "in_progress", "cancelling"]:
                    # Still running, wait a bit
                    time.sleep(1)
                    continue
                else:
                    # Unknown status
                    error_msg = f"Unknown run status: {run_status.status}"
                    return ToolQueryResult(
                        response_text="",
                        input_token_count=total_input_tokens,
                        output_token_count=total_output_tokens,
                        reasoning_token_count=reasoning_tokens,
                        tool_calls_made=tool_calls_made,
                        is_error=True,
                        error_message=error_msg
                    )
            else:
                # Hit max iterations
                return ToolQueryResult(
                    response_text="Error: Maximum polling iterations reached",
                    input_token_count=total_input_tokens,
                    output_token_count=total_output_tokens,
                    reasoning_token_count=reasoning_tokens,
                    tool_calls_made=tool_calls_made,
                    is_error=True,
                    error_message="Maximum iterations reached"
                )
            
            # Step 6: Retrieve the final response
            messages = OPENAI_CLIENT.beta.threads.messages.list(thread_id=thread.id)
            
            # Get the assistant's final response (most recent message)
            final_response = ""
            for message in messages.data:
                if message.role == "assistant":
                    # Get the text content from the message
                    for content in message.content:
                        if content.type == "text":
                            final_response = content.text.value
                            break
                    break
            
            # Return successful result
            return ToolQueryResult(
                response_text=final_response,
                input_token_count=total_input_tokens,
                output_token_count=total_output_tokens,
                reasoning_token_count=reasoning_tokens,
                tool_calls_made=tool_calls_made,
                is_error=False
            )
            
        finally:
            # Clean up: Delete the thread and assistant
            try:
                OPENAI_CLIENT.beta.threads.delete(thread_id=thread.id)
            except:
                pass  # Ignore cleanup errors
            
            try:
                OPENAI_CLIENT.beta.assistants.delete(assistant_id=assistant.id)
            except:
                pass  # Ignore cleanup errors
        
    except Exception as e:
        # Handle any API or other errors
        return ToolQueryResult(
            response_text="",
            input_token_count=0,
            output_token_count=0,
            reasoning_token_count=0,
            tool_calls_made=0,
            is_error=True,
            error_message=str(e)
        )


# ------------------------------
# Batch query functions
# ------------------------------


def query_model(
    prompt: str, model: str, max_gen_tokens: int, temperature: float
) -> QueryResult | None:
    max_retries = 5
    base_delay = 5.0

    for attempt in range(max_retries):
        try:
            if model in GOOGLE_MODELS:
                response: QueryResult = single_query_google(
                    prompt, model, max_gen_tokens, temperature
                )
            elif model in ANTHROPIC_MODELS:
                response: QueryResult = single_query_anthropic(  # type: ignore
                    prompt, model, max_gen_tokens, temperature
                )
            elif model in OPENAI_MODELS:
                response: QueryResult = single_query_openai(  # type: ignore
                    prompt, model, max_gen_tokens, temperature
                )
            else:
                raise ValueError(f"Model {model} not supported")
            return response  # Success, return response

        except Exception as e:
            print(f"Attempt {attempt + 1} failed {e}")
            if attempt < max_retries - 1:
                # Calculate delay with exponential backoff and jitter
                delay = base_delay * (2**attempt) + (
                    random.random() * 0.5
                )  # Add jitter
                print(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Giving up.")
                return None  # Failed after all retries

    raise ValueError("You should never get here.")


def query_model_batch(
    prompts: List[str],
    model: str,
    max_gen_tokens: int,
    temperature: float,
    num_workers: int = 16,
) -> List[QueryResult]:
    # Initialize results list with placeholders
    results: List[QueryResult | None] = [None] * len(prompts)
    lock = threading.Lock()  # Lock for thread-safe updates (might not be strictly needed now but good practice)

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks, storing future and original index
        futures = {
            executor.submit(
                query_model, prompt, model, max_gen_tokens, temperature
            ): index
            for index, prompt in enumerate(prompts)
        }

        # Process results as they complete, using tqdm for progress
        for future in tqdm.tqdm(
            as_completed(futures), total=len(prompts), desc="Processing samples"
        ):
            original_index = futures[future]
            original_prompt = prompts[
                original_index
            ]  # Get original prompt for error message if needed
            try:
                result = future.result()
                if result:
                    # Place result in the correct position
                    with lock:
                        results[original_index] = result
                else:
                    # Handle case where query_model returned None (max retries reached)
                    error_result = QueryResult(
                        response_text=f'Error: Max retries reached for prompt starting with "{original_prompt[:50]}..."',
                        input_token_count=0,
                        output_token_count=0,
                        is_error=True,
                    )
                    with lock:
                        results[original_index] = error_result

            except Exception as exc:
                # Print part of the prompt string to identify it
                print(
                    f'Prompt starting with "{original_prompt[:50]}..." generated an exception: {exc}'
                )

                # Create a dictionary to store the error result
                error_result = QueryResult(
                    response_text=f"Error: {exc}",
                    input_token_count=0,  # Indicate no successful API call
                    output_token_count=0,  # Indicate no successful API call
                    is_error=True,
                )
                # Place error result in the correct position
                results[original_index] = error_result

    assert all(x is not None for x in results)

    return cast(List[QueryResult], results)
