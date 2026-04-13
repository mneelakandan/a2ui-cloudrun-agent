import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="A2UI Agent Backend")

# Initialize Gemini Client using Vertex AI for enterprise-grade, keyless IAM authentication.
# Ensure the Cloud Run service account has the 'Vertex AI User' IAM role.
client = genai.Client(vertexai=True, location="us-central1")

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate-ui")
async def generate_ui(request: PromptRequest):
    try:
        # Strict system instructions to force A2UI Bridge JSON output
        instruction = """
        You are a UI generator. Create A2UI Bridge JSON messages based on the user's request. 
        IMPORTANT: Return ONLY a valid JSON array of A2UI components (e.g., Card, Column, Text, TextField) 
        with no markdown formatting, no code blocks, and no conversational text.
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=request.prompt,
            config=types.GenerateContentConfig(
                system_instruction=instruction,
                response_mime_type="application/json",
            ),
        )
        
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}
