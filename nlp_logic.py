import os
import google.generativeai as genai
from PIL import Image

# Gemini API key
api_key = os.environ.get("GOOGLE_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    print("WARNING: GOOGLE_API_KEY not found in environment variables!")


MODEL_NAME = 'gemini-2.0-flash'

def pre_scan_image(image_stream):
    """
    Acts as a universal pre-scanner. Detects if it's a plant, 
    and checks if it is universally healthy before bothering the CNN.
    """
    try:
        img = Image.open(image_stream)
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = (
            "Analyze this image. "
            "1. If it is NOT a plant, leaf, or crop, reply exactly with 'NOT_PLANT'. "
            "2. If it IS a plant, and it looks completely green and healthy with no visible signs of disease, reply exactly with 'GENERIC_HEALTHY'. "
            "3. If it IS a plant, but it shows signs of spots, wilting, yellowing, or disease, reply exactly with 'DISEASED'. "
            "Reply with ONLY one of those three words."
        )
        
        response = model.generate_content([img, prompt])
        result = response.text.strip().upper()
        return result
        
    except Exception as e:
        print(f"Pre-Scan Error: {e}")
        # Fail-open: If the API fails, assume it's a diseased plant so the CNN can try.
        return "DISEASED"

def generate_treatment_plan(disease_result, user_text="", selected_lang="English", audio_path=None):
    """
    Takes the CNN diagnosis and generates a localized, readable treatment plan or maintenance guide.
    """
    # 🚨 THE EXPERT FALLBACK 🚨
    if disease_result == "UNKNOWN_DISEASE" or "Error" in disease_result:
        return (
            "Diagnosis Result: Inconclusive\n\n"
            "This image contains a condition that is currently outside my primary training database, or the image is unclear. "
            "Because I cannot identify this with high confidence, I recommend consulting a local agricultural expert, "
            "botanist, or your nearest university extension office for a precise physical diagnosis."
        )

    # 1. THE HEALTHY VS. DISEASED LOGIC SPLIT
    is_healthy = "healthy" in disease_result.lower()
    
    # 2. THE STRICT PROMPT ENGINEERING
    if is_healthy:
        prompt = f"""
        You are an expert agricultural botanist. 
        The vision model has diagnosed this crop as: {disease_result}.

        Please provide a specific, actionable maintenance plan to keep it healthy.
        Format your response exactly like this:
        1. Watering: (Specific instructions)
        2. Sunlight & Soil: (Specific needs)
        3. Monitoring: (What to watch out for)

        Keep it under 100 words. Do not use generic advice. Do NOT use markdown, bolding, or asterisks.
        
        *** CRITICAL OVERRIDE ***
        You MUST translate your ENTIRE response into {selected_lang}. 
        If {selected_lang} is not English, you are strictly forbidden from outputting English words.
        """
    else:
        prompt = f"""
        You are an expert agricultural botanist. 
        A farmer has uploaded an image of a crop. The vision model has diagnosed it as: {disease_result}.

        Please provide a highly specific, actionable treatment plan.
        Format your response exactly like this:
        1. Immediate Action: (What to do today)
        2. Treatment: (Specific sprays or fertilizers)
        3. Prevention: (How to stop it next season)

        Keep it under 150 words and do not use generic advice. Do NOT use markdown, bolding, or asterisks.
        
        *** CRITICAL OVERRIDE ***
        You MUST translate your ENTIRE response into {selected_lang}. 
        If {selected_lang} is not English, you are strictly forbidden from outputting English words.
        """

    # 3. Append optional user context
    if user_text:
        prompt += f"\n\nUser Question: {user_text}\nAnswer this question briefly at the end of the plan."

    # --- DEBUGGING PRINT ---
    print(f"\n[DEBUG] Backend received Language: '{selected_lang}' | Disease: '{disease_result}'")

    try:
        # 4. Standard API Call
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        # Strip out any lingering markdown asterisks that Gemini tries to sneak in
        clean_text = response.text.replace("*", "")
        return clean_text

    except Exception as e:
        error_message = str(e).lower()
        
        # 🚨 DEBUG: This prints the real error to your terminal so we can see it
        print(f"\n[GEMINI API ERROR]: {e}\n")
        
        # 5. The "Presentation Saver" Offline Fallback
        if "429" in error_message or "exhausted" in error_message or "quota" in error_message:
            
            # If the CNN says it's healthy, give offline healthy advice
            if is_healthy:
                return (
                    "Standard Maintenance Plan:\n\n"
                    "1. Watering: Continue your current schedule to maintain consistent soil moisture. Water at the base of the plant.\n\n"
                    "2. Sunlight: Ensure the crop continues to receive adequate sunlight.\n\n"
                    "3. Monitoring: Check leaves weekly for any early signs of pests, spots, or discoloration."
                )
            
            # If the CNN says it's diseased, give offline treatment advice
            else:
                return (
                    "Emergency Treatment Plan:\n\n"
                    "1. Isolation: Immediately remove and destroy any severely affected leaves to prevent the disease from spreading to healthy crops.\n\n"
                    "2. Environment: Ensure proper soil drainage and avoid overhead watering to keep the foliage dry.\n\n"
                    "3. Treatment: Apply a standard, crop-appropriate organic fungicide or neem oil spray. Apply during the early morning or late evening.\n\n"
                    "(Note: Cloud diagnostics are running in offline mode. Standard safety practices apply.)"
                )
                
        return f"System Error generating treatment plan: {str(e)}"