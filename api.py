import io
import time
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from ai_logic import analyze_leaf
from nlp_logic import generate_treatment_plan, pre_scan_image

# 1. Create the App FIRST
app = FastAPI(title="AI Crop Doctor API")

# 2. Setup Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Define the Front Door (The Home Page)
@app.get("/")
async def read_index():
    return FileResponse(
        'index.html',
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

# --- NEW: Premium App Pages ---

@app.get("/encyclopedia", response_class=HTMLResponse)
async def encyclopedia():
    with open("encyclopedia.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/how-it-works", response_class=HTMLResponse)
async def how_it_works():
    with open("how_it_works.html", "r", encoding="utf-8") as f:
        return f.read()

# ------------------------------

# 4. Define the AI Logic
@app.post("/api/analyze")
def analyze_crop(
    image: UploadFile = File(...),
    language: str = Form("English"),
    question: str = Form("")
):
    try:
        # Read the image safely into memory
        image_bytes = image.file.read()
        
        # Give a copy to the Universal Pre-Scanner (Gemini)
        scan_result = pre_scan_image(io.BytesIO(image_bytes))
        
        # Prevent the "High Traffic" 429 error
        time.sleep(2)

        # The Routing Logic (Dual-Brain Architecture)
        if "NOT_PLANT" in scan_result:
            return {"status": "error", "message": "Nice try! That doesn't look like a plant or a leaf."}

        if "GENERIC_HEALTHY" in scan_result:
            disease_result = "healthy crop"
        else:
            disease_result = analyze_leaf(io.BytesIO(image_bytes)) 
        
        # --- PROFESSIONAL ERROR HANDLING START ---
        
        # If the image is unsupported or an error occurred during leaf analysis
        if "ERROR" in disease_result or "Unsupported" in disease_result:
            # We provide professional generic advice instead of calling the Gemini API
            return {
                "status": "success",
                "diagnosis": disease_result,
                "treatment": (
                    "It seems the image provided is either unclear or not a supported crop leaf.\n\n"
                    "General Guidelines:\n"
                    "1. Ensure the leaf is well-lit and centered.\n"
                    "2. Avoid blurry or dark photos.\n"
                    "3. For specific diagnosis, please upload a Tomato, Potato, or Pepper leaf.\n"
                    "4. If symptoms persist, contact the National Farmers Helpline or your nearest expert mentioned below."
                )
            }

        # Run Gemini Treatment Plan only for valid diagnoses
        try:
            treatment_plan = generate_treatment_plan(
                disease_result, 
                user_text=question, 
                selected_lang=language, 
                audio_path=None 
            )
        except Exception:
            # Final fallback if the Gemini API itself is down/key is missing
            treatment_plan = "Please consult a local agricultural officer for a detailed treatment plan based on this diagnosis."
        
        # --- PROFESSIONAL ERROR HANDLING END ---

        return {
            "status": "success",
            "diagnosis": disease_result,
            "treatment": treatment_plan
        }
        
    except Exception as e:
        print(f"CRASH ERROR: {str(e)}")
        return {"status": "error", "message": "An internal error occurred. Please try again with a clearer image."}