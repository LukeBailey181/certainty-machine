"""
verifier_server.py

Example launch command:
    python verifier_server.py --port 8001 --num_workers 32
"""

import argparse
import logging
import os
import socket
from typing import List

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from tqdm import tqdm
from certainty_machine.verification.prover.lean.verifier import Lean4ServerScheduler

DEFAULT_LEAN_WORKSPACE = "./mathlib4/"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verifier_server")

# ------------------------------------------------------------------------
# FastAPI app definition
# ------------------------------------------------------------------------
app = FastAPI(title="Lean Verifier Server", version="0.1.0")

# Global references persisted in app state
app.state.server = None  # Server instance
app.state.num_workers = None  # Number of worker processes


# ------------------------------------------------------------------------
# Pydantic request schemas
# ------------------------------------------------------------------------
class VerificationRequest(BaseModel):
    cleaned_proofs: List[str]
    timeout: float
    memory_limit: int


# ------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------
def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return False
        except socket.error:
            return True


# ------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------
@app.post("/verify")
async def verify(request: VerificationRequest):
    cleaned_proofs = request.cleaned_proofs
    total = len(cleaned_proofs)
    pbar = tqdm(
        total=total, leave=True, desc="Verifying Lean proofs", dynamic_ncols=True
    )

    # Use the Lean4ServerScheduler to verify proofs
    verification_outputs = []
    try:
        # Submit all proofs for verification
        request_id_list = app.state.verifier_scheduler.submit_all_request(
            cleaned_proofs
        )

        # Get all verification outputs
        outputs_list = app.state.verifier_scheduler.get_all_request_outputs(
            request_id_list
        )
        verification_outputs = outputs_list

        if pbar:
            pbar.update(total)
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        verification_outputs = [
            {"pass": False, "complete": False, "error": str(e)} for _ in range(total)
        ]
    finally:
        if pbar:
            pbar.close()

    return {"verification_outputs": verification_outputs}


@app.post("/ping")
async def ping():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/shutdown")
async def shutdown(request: Request):
    """
    Endpoint to gracefully shutdown the server
    """
    logger.info("Received shutdown request")

    # Schedule the server shutdown
    async def shutdown_server():
        # Wait a brief moment to allow the response to be sent
        import asyncio

        await asyncio.sleep(1)
        # Stop the server
        await request.app.state.server.shutdown()
        # Force exit the process
        os._exit(0)

    import asyncio

    asyncio.create_task(shutdown_server())
    return {"message": "Server is shutting down"}


# ------------------------------------------------------------------------
# Main entrypoint
# ------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Lean Verifier Server")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", type=int, default=8001, help="Port to bind the server to"
    )
    parser.add_argument(
        "--num_workers", type=int, default=32, help="Maximum number of worker processes"
    )
    parser.add_argument(
        "--lean_workspace",
        type=str,
        default=DEFAULT_LEAN_WORKSPACE,
        help="Path to mathlib4",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)

    if is_port_in_use(args.port):
        raise ValueError(f"Port {args.port} is already in use")

    logger.info(f"Starting server on {ip_address}:{args.port}")

    # Store configuration in app state
    app.state.num_workers = args.num_workers
    app.state.lean_workspace = args.lean_workspace
    app.state.verifier_scheduler = Lean4ServerScheduler(
        max_concurrent_requests=args.num_workers, timeout=50, memory_limit=-1
    )

    # Start the server
    config = uvicorn.Config(app=app, host=args.host, port=args.port, workers=1)
    server = uvicorn.Server(config=config)
    app.state.server = server

    try:
        server.run()
    finally:
        # Ensure we clean up the scheduler on shutdown
        if hasattr(app.state, "verifier_scheduler"):
            app.state.verifier_scheduler.close()


if __name__ == "__main__":
    main()
