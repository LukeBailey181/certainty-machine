"""
demo_ui_server.py

FastAPI server for the multi-turn theorem proving demo UI.
Provides real-time streaming updates during the proving process.
"""

import json
import asyncio
from typing import AsyncGenerator, Dict, Any
from dataclasses import asdict

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from certainty_machine.querying.multi_turn import multi_turn_prompt, MultiTurnResult
from certainty_machine.querying.api import query_model, QueryResult
from certainty_machine.querying.prompts import get_prover_prompt, get_refinement_prompt, extract_proof, NO_CODE_FOUND_TAG
from certainty_machine.verification.verify_client import verify_lean_code
from certainty_machine.verification.types import VerificationOutput


# Default Lean header for demo
DEFAULT_HEADER = """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

"""

app = FastAPI(title="Certainty Machine Demo", version="1.0.0")


class ProofRequest(BaseModel):
    theorem: str
    thinking_time: int  # Number of turns (1-20)
    model: str = "gpt-4.1-mini"
    verifier_address: str = "172.24.75.239:8001"


async def stream_multi_turn_progress(
    theorem: str,
    max_turns: int,
    model: str,
    verifier_address: str
) -> AsyncGenerator[str, None]:
    """
    Stream progress updates during multi-turn theorem proving.
    
    Yields JSON-encoded progress messages for the frontend.
    """
    header = DEFAULT_HEADER
    turn_history = []
    current_proof = None
    
    # Send initial status
    yield f"data: {json.dumps({'type': 'status', 'message': f'Starting theorem proving with {max_turns} thinking turns...', 'turn': 0})}\n\n"
    await asyncio.sleep(0.5)
    
    for turn in range(max_turns):
        turn_num = turn + 1
        
        # Send turn start message
        yield f"data: {json.dumps({'type': 'turn_start', 'turn': turn_num, 'max_turns': max_turns})}\n\n"
        await asyncio.sleep(0.3)
        
        # Determine prompt for this turn
        if turn == 0:
            prompt = get_prover_prompt(header=header, theorem=theorem)
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating initial proof attempt...', 'turn': turn_num})}\n\n"
        else:
            prompt = get_refinement_prompt(header=header, theorem=theorem, turn_history=turn_history)
            yield f"data: {json.dumps({'type': 'status', 'message': 'Refining proof based on previous errors...', 'turn': turn_num})}\n\n"
        
        # Send querying status with animation
        yield f"data: {json.dumps({'type': 'querying', 'message': f'Querying {model}...', 'turn': turn_num})}\n\n"
        await asyncio.sleep(0.5)
        
        # Query the model
        try:
            response: QueryResult = query_model(
                prompt=prompt,
                model=model,
                max_gen_tokens=8000,
                temperature=1.0
            )
            
            if response is None:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Model query failed', 'turn': turn_num})}\n\n"
                return
            
            # Stream the model response
            yield f"data: {json.dumps({'type': 'model_response', 'content': response.response_text, 'turn': turn_num})}\n\n"
            await asyncio.sleep(1.0)
            
            # Extract proof
            proof = extract_proof(response.response_text)
            if proof == NO_CODE_FOUND_TAG:
                dummy_verification = VerificationOutput(
                    verdict=False,
                    output={"error": "No proof found in model response"}
                )
                turn_history.append((prompt, response.response_text, dummy_verification))
                yield f"data: {json.dumps({'type': 'extraction_error', 'message': 'No valid proof found in response', 'turn': turn_num})}\n\n"
                continue
            
            current_proof = proof
            full_lean_file = header + theorem + proof
            
            # Show extracted proof
            yield f"data: {json.dumps({'type': 'extracted_proof', 'content': full_lean_file, 'turn': turn_num})}\n\n"
            await asyncio.sleep(1.0)
            
            # Send verification status
            yield f"data: {json.dumps({'type': 'verifying', 'message': 'Verifying proof with Lean compiler...', 'turn': turn_num})}\n\n"
            await asyncio.sleep(0.5)
            
            # Verify the proof
            verification_outputs = verify_lean_code(
                verifier_address=verifier_address,
                lean_code=[full_lean_file]
            )
            
            verification = verification_outputs[0]
            turn_history.append((prompt, response.response_text, verification))
            
            if verification.verdict:
                # Success!
                yield f"data: {json.dumps({'type': 'success', 'proof': current_proof, 'full_file': full_lean_file, 'turn': turn_num})}\n\n"
                return
            else:
                # Failed verification
                error_info = verification.output.get('errors', str(verification.output))
                yield f"data: {json.dumps({'type': 'verification_error', 'errors': error_info, 'turn': turn_num})}\n\n"
                await asyncio.sleep(1.5)
                
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Error: {str(e)}', 'turn': turn_num})}\n\n"
            return
    
    # Reached max turns without success
    final_errors = turn_history[-1][2].output.get('errors', 'Unknown error') if turn_history else 'No attempts made'
    yield f"data: {json.dumps({'type': 'final_failure', 'errors': final_errors, 'turns': max_turns})}\n\n"


@app.get("/", response_class=HTMLResponse)
async def get_demo_page():
    """Serve the main demo page."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Certainty Machine - Theorem Proving Demo</title>
    <style>
        :root {
            --bg-primary: #0f0f0f;
            --bg-secondary: #1a1a1a;
            --bg-tertiary: #252525;
            --text-primary: #ffffff;
            --text-secondary: #b3b3b3;
            --text-muted: #666666;
            --accent-blue: #3b82f6;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-yellow: #f59e0b;
            --border-color: #333333;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .header {
            text-align: center;
            margin-bottom: 3rem;
        }
        
        .header h1 {
            font-size: 3rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-green));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }
        
        .header p {
            font-size: 1.2rem;
            color: var(--text-secondary);
        }
        
        .input-section {
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 2rem;
            border: 1px solid var(--border-color);
            margin-bottom: 2rem;
        }
        
        .form-group {
            margin-bottom: 1.5rem;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        .theorem-input {
            width: 100%;
            min-height: 120px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            color: var(--text-primary);
            font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
            font-size: 0.9rem;
            resize: vertical;
        }
        
        .theorem-input:focus {
            outline: none;
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }
        
        .slider-container {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .slider {
            flex: 1;
            height: 6px;
            background: var(--bg-tertiary);
            border-radius: 3px;
            outline: none;
            -webkit-appearance: none;
        }
        
        .slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            background: var(--accent-blue);
            border-radius: 50%;
            cursor: pointer;
        }
        
        .slider::-moz-range-thumb {
            width: 20px;
            height: 20px;
            background: var(--accent-blue);
            border-radius: 50%;
            cursor: pointer;
            border: none;
        }
        
        .slider-value {
            min-width: 60px;
            text-align: center;
            font-weight: 600;
            color: var(--accent-blue);
        }
        
        .submit-btn {
            width: 100%;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-green));
            color: white;
            border: none;
            padding: 1rem 2rem;
            border-radius: 8px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(59, 130, 246, 0.3);
        }
        
        .submit-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .progress-section {
            background: var(--bg-secondary);
            border-radius: 12px;
            border: 1px solid var(--border-color);
            padding: 2rem;
            display: none;
        }
        
        .progress-header {
            display: flex;
            justify-content: between;
            align-items: center;
            margin-bottom: 1.5rem;
        }
        
        .turn-indicator {
            background: var(--bg-tertiary);
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-weight: 600;
            color: var(--accent-blue);
        }
        
        .status-message {
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            border-left: 4px solid var(--accent-blue);
            background: rgba(59, 130, 246, 0.1);
        }
        
        .spinner {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid var(--bg-tertiary);
            border-top: 2px solid var(--accent-blue);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 0.5rem;
        }
        
        .cog {
            display: inline-block;
            width: 16px;
            height: 16px;
            margin-right: 0.5rem;
            animation: spin 2s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .response-box {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
            font-size: 0.85rem;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .success-message {
            background: rgba(16, 185, 129, 0.1);
            border-left: 4px solid var(--accent-green);
            color: var(--accent-green);
        }
        
        .error-message {
            background: rgba(239, 68, 68, 0.1);
            border-left: 4px solid var(--accent-red);
            color: var(--accent-red);
        }
        
        .proof-display {
            background: var(--bg-tertiary);
            border: 1px solid var(--accent-green);
            border-radius: 8px;
            padding: 1.5rem;
            font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
            font-size: 0.9rem;
            white-space: pre-wrap;
            margin: 1rem 0;
        }
        
        .section-title {
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: var(--text-primary);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Certainty Machine</h1>
            <p>AI-Powered Theorem Proving with Multi-Turn Refinement</p>
        </div>
        
        <div class="input-section">
            <form id="theoremForm">
                <div class="form-group">
                    <label for="theorem">Lean Theorem Statement:</label>
                    <textarea 
                        id="theorem" 
                        class="theorem-input" 
                        placeholder="Enter your theorem statement here, e.g.:&#10;theorem example (x y : ℕ) : x + y = y + x := by"
                        required
                    ></textarea>
                </div>
                
                <div class="form-group">
                    <label for="thinking_time">Thinking Time (Number of Refinement Turns):</label>
                    <div class="slider-container">
                        <input type="range" id="thinking_time" class="slider" min="1" max="20" value="5">
                        <div class="slider-value" id="slider-value">5</div>
                    </div>
                </div>
                
                <button type="submit" class="submit-btn" id="submit-btn">
                    Start Theorem Proving
                </button>
            </form>
        </div>
        
        <div class="progress-section" id="progress-section">
            <div class="progress-header">
                <h3>Proving Progress</h3>
                <div class="turn-indicator" id="turn-indicator">Turn 0/0</div>
            </div>
            <div id="progress-content"></div>
        </div>
    </div>

    <script>
        const form = document.getElementById('theoremForm');
        const submitBtn = document.getElementById('submit-btn');
        const progressSection = document.getElementById('progress-section');
        const progressContent = document.getElementById('progress-content');
        const turnIndicator = document.getElementById('turn-indicator');
        const slider = document.getElementById('thinking_time');
        const sliderValue = document.getElementById('slider-value');
        
        // Update slider value display
        slider.addEventListener('input', function() {
            sliderValue.textContent = this.value;
        });
        
        // Form submission
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const theorem = document.getElementById('theorem').value.trim();
            const thinkingTime = parseInt(document.getElementById('thinking_time').value);
            
            if (!theorem) {
                alert('Please enter a theorem statement');
                return;
            }
            
            // Reset and show progress section
            progressContent.innerHTML = '';
            progressSection.style.display = 'block';
            submitBtn.disabled = true;
            submitBtn.textContent = 'Proving...';
            
            try {
                const response = await fetch('/prove', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        theorem: theorem,
                        thinking_time: thinkingTime
                    })
                });
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                
                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\\n');
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                handleProgressUpdate(data);
                            } catch (e) {
                                console.error('Error parsing SSE data:', e);
                            }
                        }
                    }
                }
            } catch (error) {
                console.error('Error:', error);
                addErrorMessage('Connection error occurred');
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Start Theorem Proving';
            }
        });
        
        function handleProgressUpdate(data) {
            switch (data.type) {
                case 'status':
                    addStatusMessage(data.message, data.turn);
                    break;
                case 'turn_start':
                    turnIndicator.textContent = `Turn ${data.turn}/${data.max_turns}`;
                    break;
                case 'querying':
                    addStatusMessage(`<span class="spinner"></span>${data.message}`, data.turn);
                    break;
                case 'model_response':
                    addResponseBox('Model Response', data.content);
                    break;
                case 'extracted_proof':
                    addResponseBox('Extracted Lean File', data.content);
                    break;
                case 'verifying':
                    addStatusMessage(`<span class="cog">⚙️</span>${data.message}`, data.turn);
                    break;
                case 'success':
                    addSuccessMessage(data.proof, data.full_file);
                    break;
                case 'verification_error':
                    addVerificationError(data.errors, data.turn);
                    break;
                case 'final_failure':
                    addFinalFailure(data.errors, data.turns);
                    break;
                case 'error':
                    addErrorMessage(data.message);
                    break;
            }
        }
        
        function addStatusMessage(message, turn) {
            const div = document.createElement('div');
            div.className = 'status-message';
            div.innerHTML = `<strong>Turn ${turn}:</strong> ${message}`;
            progressContent.appendChild(div);
            progressContent.scrollTop = progressContent.scrollHeight;
        }
        
        function addResponseBox(title, content) {
            const div = document.createElement('div');
            div.innerHTML = `
                <div class="section-title">${title}:</div>
                <div class="response-box">${content}</div>
            `;
            progressContent.appendChild(div);
            progressContent.scrollTop = progressContent.scrollHeight;
        }
        
        function addSuccessMessage(proof, fullFile) {
            const div = document.createElement('div');
            div.className = 'success-message status-message';
            div.innerHTML = `
                <strong>🎉 Proof Successful!</strong>
                <div class="proof-display">${fullFile}</div>
            `;
            progressContent.appendChild(div);
            progressContent.scrollTop = progressContent.scrollHeight;
        }
        
        function addVerificationError(errors, turn) {
            const div = document.createElement('div');
            div.className = 'error-message status-message';
            div.innerHTML = `
                <strong>Turn ${turn}: Verification Failed</strong>
                <div class="response-box">${typeof errors === 'string' ? errors : JSON.stringify(errors, null, 2)}</div>
            `;
            progressContent.appendChild(div);
            progressContent.scrollTop = progressContent.scrollHeight;
        }
        
        function addFinalFailure(errors, turns) {
            const div = document.createElement('div');
            div.className = 'error-message status-message';
            div.innerHTML = `
                <strong>❌ Failed to prove after ${turns} turns</strong>
                <div class="response-box">${typeof errors === 'string' ? errors : JSON.stringify(errors, null, 2)}</div>
            `;
            progressContent.appendChild(div);
            progressContent.scrollTop = progressContent.scrollHeight;
        }
        
        function addErrorMessage(message) {
            const div = document.createElement('div');
            div.className = 'error-message status-message';
            div.innerHTML = `<strong>Error:</strong> ${message}`;
            progressContent.appendChild(div);
            progressContent.scrollTop = progressContent.scrollHeight;
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.post("/prove")
async def prove_theorem(request: ProofRequest):
    """
    Start theorem proving with streaming progress updates.
    
    Returns Server-Sent Events (SSE) stream with real-time progress.
    """
    return StreamingResponse(
        stream_multi_turn_progress(
            theorem=request.theorem,
            max_turns=request.thinking_time,
            model=request.model,
            verifier_address=request.verifier_address
        ),
        media_type="text/plain"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)