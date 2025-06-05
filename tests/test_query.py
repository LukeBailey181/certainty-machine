from typing import List

from big_stp.models.query import (
    query_model,
    single_query_google,
    query_model_batch,
    QueryResult,
)


def test_single_query_google():
    model = "gemini-2.0-flash-001"
    prompt = "What is the capital of France? Answer with a single word."
    response: QueryResult = single_query_google(prompt, model)

    assert response is not None
    assert response.response_text is not None
    assert not response.is_error
    assert response.input_token_count > 0
    assert response.total_token_count > 0
    # This is technically a bit flaky if your model is bad
    assert response.response_text.lower().strip() == "paris"


def test_query():
    model = "gemini-2.0-flash-001"
    prompt = "What is the capital of France? Answer with a single word."
    response: QueryResult = query_model(prompt, model)

    assert response is not None
    assert response.response_text is not None
    assert not response.is_error
    assert response.input_token_count > 0
    assert response.total_token_count > 0
    # This is technically a bit flaky if your model is bad
    assert response.response_text.lower().strip() == "paris"


def test_batch_query():
    model = "gemini-2.0-flash-001"
    prompts = [
        "What is the capital of France? Answer with a single word.",
        "What is the capital of Germany? Answer with a single word.",
        "What is the capital of Italy? Answer with a single word.",
        "What is the capital of the UK? Answer with a single word.",
    ]

    answers = [
        "paris",
        "berlin",
        "rome",
        "london",
    ]

    responses: List[QueryResult] = query_model_batch(prompts, model)

    for response, answer in zip(responses, answers):
        assert response is not None
        assert response.response_text is not None
        assert not response.is_error
        assert response.input_token_count > 0
        assert response.total_token_count > 0
        # This is technically a bit flaky if your model is bad
        assert response.response_text.lower().strip() == answer


if __name__ == "__main__":
    # If you want you can run single tests here
    # test_single_query_google()
    # test_batch_query()
    pass
