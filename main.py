# main.py
import ollama
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI(
    title="Ollama Medllama2 API",
    description="API que serve o modelo Medllama2 via Ollama.",
    version="1.0.0"
)

class OllamaRequest(BaseModel):
    model: str
    prompt: str
    stream: Optional[bool] = False

@app.post("/api/generate")
async def generate_response(request: OllamaRequest):
    
    print("="*25)
    try:
        response = ollama.chat(
            model=request.model,
            messages=[{
                'role': 'user',
                'content': request.prompt
            }],
            stream=request.stream,
        )
        return {"response": response['message']['content']}
    except Exception as e:
        return {"error": f"Erro ao conectar com Ollama ou processar a requisição: {e}"}, 500