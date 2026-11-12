"""Start the What Is That Smell? simulated electronic nose."""

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "what_is_that_smell.server:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8017")),
    )
