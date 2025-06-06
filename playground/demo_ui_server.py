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
            --bg-primary: #0c0c0c;
            --bg-secondary: #1a1a1a;
            --text-primary: #e5e5e5;
            --text-secondary: #a0a0a0;
            --text-muted: #666666;
            --accent-blue: #3b82f6;
            --accent-green: #22c55e;
            --accent-red: #ef4444;
            --accent-yellow: #f59e0b;
            --border-color: #2a2a2a;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'SF Mono', Monaco, 'Cascadia Code', 'JetBrains Mono', monospace;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.5;
            min-height: 100vh;
            font-size: 14px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 1.5rem;
        }
        
        .header {
            margin-bottom: 2rem;
        }
        
        .header h1 {
            font-size: 1.8rem;
            font-weight: 600;
            color: var(--accent-blue);
            margin-bottom: 0.5rem;
        }
        
        .header p {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }
        
        .input-section {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        .form-group {
            margin-bottom: 1rem;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 0.5rem;
            color: var(--text-primary);
            font-weight: 500;
        }
        
        .theorem-input {
            width: 100%;
            min-height: 100px;
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            padding: 0.75rem;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.85rem;
            resize: vertical;
        }
        
        .theorem-input:focus {
            outline: none;
            border-color: var(--accent-blue);
        }
        
        .slider-container {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .slider {
            flex: 1;
            height: 4px;
            background: var(--bg-primary);
            outline: none;
            -webkit-appearance: none;
        }
        
        .slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            background: var(--accent-blue);
            cursor: pointer;
        }
        
        .slider::-moz-range-thumb {
            width: 16px;
            height: 16px;
            background: var(--accent-blue);
            cursor: pointer;
            border: none;
        }
        
        .slider-value {
            min-width: 40px;
            color: var(--accent-blue);
            font-weight: 500;
        }
        
        .submit-btn {
            background: var(--accent-blue);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            font-family: inherit;
            font-size: 0.9rem;
            cursor: pointer;
        }
        
        .submit-btn:hover {
            background: #2563eb;
        }
        
        .submit-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        
        .terminal-output {
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            padding: 1.5rem;
            font-family: inherit;
            font-size: 0.85rem;
            display: none;
        }
        
        .terminal-line {
            margin-bottom: 0.5rem;
            word-wrap: break-word;
        }
        
        .terminal-line:last-child {
            margin-bottom: 0;
        }
        
        .turn-header {
            color: var(--accent-blue);
            font-weight: 600;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
        
        .turn-header:first-child {
            margin-top: 0;
        }
        
        .status-line {
            color: var(--text-secondary);
        }
        
        .success-line {
            color: var(--accent-green);
            font-weight: 500;
        }
        
        .error-line {
            color: var(--accent-red);
        }
        
        .code-block {
            background: var(--bg-secondary);
            border-left: 3px solid var(--accent-blue);
            padding: 0.75rem;
            margin: 0.5rem 0;
            color: var(--text-primary);
            white-space: pre-wrap;
            font-family: inherit;
        }
        
        .success-code {
            border-left-color: var(--accent-green);
        }
        
        .error-code {
            border-left-color: var(--accent-red);
        }
        
        .spinner {
            display: inline-block;
            width: 12px;
            height: 12px;
            border: 1px solid var(--text-muted);
            border-top: 1px solid var(--accent-blue);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 0.5rem;
        }
        
        .cog {
            margin-right: 0.5rem;
            animation: spin 2s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .complete {
            animation: none !important;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Certainty Machine</h1>
            <p>AI-powered theorem proving with multi-turn refinement</p>
        </div>
        
        <div class="input-section">
            <form id="theoremForm">
                <div class="form-group">
                    <label for="theorem">theorem statement:</label>
                    <textarea 
                        id="theorem" 
                        class="theorem-input" 
                        placeholder="theorem example (x y : ℕ) : x + y = y + x := by"
                        required
                    ></textarea>
                </div>
                
                <div class="form-group">
                    <label for="thinking_time">thinking time (turns):</label>
                    <div class="slider-container">
                        <input type="range" id="thinking_time" class="slider" min="1" max="20" value="5">
                        <div class="slider-value" id="slider-value">5</div>
                    </div>
                </div>
                
                <button type="submit" class="submit-btn" id="submit-btn">
                    start proving
                </button>
            </form>
        </div>
        
        <div class="terminal-output" id="terminal-output">
        </div>
    </div>

    <script>
        const form = document.getElementById('theoremForm');
        const submitBtn = document.getElementById('submit-btn');
        const terminalOutput = document.getElementById('terminal-output');
        const slider = document.getElementById('thinking_time');
        const sliderValue = document.getElementById('slider-value');
        
        let currentSpinner = null;
        
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
            
            // Reset and show terminal
            terminalOutput.innerHTML = '';
            terminalOutput.style.display = 'block';
            submitBtn.disabled = true;
            submitBtn.textContent = 'proving...';
            
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
                addTerminalLine('error: connection failed', 'error-line');
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = 'start proving';
            }
        });
        
        function handleProgressUpdate(data) {
            switch (data.type) {
                case 'status':
                    addTerminalLine(`${data.message}`, 'status-line');
                    break;
                case 'turn_start':
                    addTerminalLine(`\\n=== Turn ${data.turn}/${data.max_turns} ===`, 'turn-header');
                    break;
                case 'querying':
                    stopCurrentAnimation();
                    currentSpinner = addTerminalLine(`<span class="spinner"></span>${data.message}`, 'status-line');
                    break;
                case 'model_response':
                    stopCurrentAnimation();
                    addTerminalLine('✓ model response received', 'success-line');
                    addCodeBlock(data.content);
                    break;
                case 'extracted_proof':
                    addTerminalLine('✓ proof extracted', 'success-line');
                    addCodeBlock(data.content, 'success-code');
                    break;
                case 'verifying':
                    stopCurrentAnimation();
                    currentSpinner = addTerminalLine(`<span class="cog">⚙</span>${data.message}`, 'status-line');
                    break;
                case 'success':
                    stopCurrentAnimation();
                    addTerminalLine('\\n🎉 PROOF SUCCESSFUL!', 'success-line');
                    addCodeBlock(data.full_file, 'success-code');
                    break;
                case 'verification_error':
                    stopCurrentAnimation();
                    addTerminalLine('✗ verification failed', 'error-line');
                    addCodeBlock(typeof data.errors === 'string' ? data.errors : JSON.stringify(data.errors, null, 2), 'error-code');
                    break;
                case 'final_failure':
                    stopCurrentAnimation();
                    addTerminalLine(`\\n❌ Failed after ${data.turns} turns`, 'error-line');
                    addCodeBlock(typeof data.errors === 'string' ? data.errors : JSON.stringify(data.errors, null, 2), 'error-code');
                    break;
                case 'error':
                    stopCurrentAnimation();
                    addTerminalLine(`error: ${data.message}`, 'error-line');
                    break;
            }
        }
        
        function addTerminalLine(text, className = '') {
            const div = document.createElement('div');
            div.className = `terminal-line ${className}`;
            div.innerHTML = text;
            terminalOutput.appendChild(div);
            return div;
        }
        
        function addCodeBlock(content, className = '') {
            const div = document.createElement('div');
            div.className = `code-block ${className}`;
            div.textContent = content;
            terminalOutput.appendChild(div);
        }
        
        function stopCurrentAnimation() {
            if (currentSpinner) {
                const spinners = currentSpinner.querySelectorAll('.spinner, .cog');
                spinners.forEach(spinner => {
                    spinner.classList.add('complete');
                });
                currentSpinner = null;
            }
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