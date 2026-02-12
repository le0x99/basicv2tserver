import os
import uvicorn

if __name__ == "__main__":
    # Ensure GPU visible only here
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    uvicorn.run(
        "server.model_server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        timeout_keep_alive=300
    )
