import tqdm
import threading
import time
import random  # Import random for jitter calculation
from typing import List, cast
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import anthropic
import openai
from google.genai import types
from anthropic.resources.messages.messages import Messages as AnthropicMessages

# GOOGLE_CLIENT = genai.Client(vertexai=True, project="", location="us-central1")
GOOGLE_CLIENT = None
ANTHROPIC_CLIENT = anthropic.Anthropic()
OPENAI_CLIENT = openai.OpenAI()


GOOGLE_MODELS: List[str] = ["gemini-2.0-flash-001"]
OPENAI_MODELS = [
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
