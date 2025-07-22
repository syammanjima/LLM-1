from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
from typing import List
from datetime import datetime

app = FastAPI()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "llama2"

class ChatResponse(BaseModel):
    response: str
    model: str
    created_at: str

@app.get("/models")
async def get_models():
    """Get available Ollama models"""
    try:
        # First check if Ollama is available
        models = ollama.list()
        if models and "models" in models:
            model_names = [model["name"] for model in models["models"]]
            return {"models": model_names}
        else:
            # Fallback to default models if Ollama doesn't respond properly
            return {"models": ["llama2", "mistral", "codellama", "gemma3"]}
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        # Return default models as fallback
        return {"models": ["llama2", "mistral", "codellama", "gemma3"]}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send chat request to Ollama"""
    try:
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        response = ollama.chat(
            model=request.model,
            messages=messages
        )
        
        return ChatResponse(
            response=response["message"]["content"],
            model=request.model,
            created_at=str(datetime.now())
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
