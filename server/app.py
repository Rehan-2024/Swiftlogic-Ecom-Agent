import os
import sys
import uvicorn

# Add the parent directory to the path so it can find ecom_env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import create_fastapi_app
from ecom_env import EcomEnv, EcomAction, EcomObservation

# Build the official OpenEnv FastAPI server by passing the CLASS, not an instance
app = create_fastapi_app(EcomEnv, EcomAction, EcomObservation)

@app.get("/")
async def root():
    return {
        "status": "online",
        "environment": "Swiftlogic E-commerce Agent",
        "version": "1.0.0",
        "framework": "OpenEnv v0.2.3",
        "endpoints": ["/reset", "/step", "/state"]
    }

def main():
    # Bind to the port Hugging Face expects
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()