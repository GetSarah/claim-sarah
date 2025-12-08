import os
import pytesseract
from fastapi import FastAPI, Request, HTTPException
from pdf2image import convert_from_bytes
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Optional
import json

# Initialize FastAPI
app = FastAPI()

# Initialize OpenAI (Set your API key in environment variables)
# export OPENAI_API_KEY="sk-..."
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ------------------------
# DATA MODELS (Strict Output)
# ------------------------
class LineItem(BaseModel):
    description: str
    quantity: float
    unit: str
    total_price: float

class ClaimData(BaseModel):
    carrier: str
    claim_number: Optional[str]
    insured_name: Optional[str]
    property_address: Optional[str]
    rcv: Optional[float]
    acv: Optional[float]
    deductible: Optional[float]
    depreciation: Optional[float]
    roof_squares: Optional[float]
    line_items: List[LineItem]
    missing_items: List[str]
    summary: str

# ------------------------
# HELPER: OCR
# ------------------------
def extract_text_from_pdf(binary: bytes) -> str:
    """
    Converts PDF bytes to images, preprocesses them for better contrast,
    and runs Tesseract OCR.
    """
    text_chunks = []
    try:
        # Convert PDF to images (300 DPI is standard for OCR)
        images = convert_from_bytes(binary, dpi=300)
        
        for i, img in enumerate(images):
            # Optional: Convert to grayscale for better OCR accuracy
            img = img.convert('L') 
            
            # Extract text
            page_text = pytesseract.image_to_string(img)
            text_chunks.append(f"--- PAGE {i+1} ---\n{page_text}")
            
        return "\n".join(text_chunks)
    except Exception as e:
        print(f"OCR Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process PDF file. Ensure Poppler is installed.")

# ------------------------
# HELPER: AI EXTRACTION
# ------------------------
def analyze_with_llm(raw_text: str) -> ClaimData:
    """
    Sends raw OCR text to GPT-4o with strict rules for distinguishing 
    Category Totals vs. Grand Totals.
    """
    
    system_prompt = """
    You are ClaimSarah, an expert roofing insurance adjuster AI. 
    You are processing OCR text from an insurance estimate (Xactimate/Symbility).

    YOUR GOAL: Extract strict JSON data.

    CRITICAL RULES FOR "MONEY" (RCV / ACV):
    1. HIERARCHY OF TRUTH: You will often see "Total Roofing" or "Total Slope" followed by a final "TOTALS" or "Net Claim" line.
    2. ALWAYS prefer the "Grand Total" / "TOTALS" / "Net Claim" at the very bottom of the document over specific category totals.
    3. The 'rcv' and 'acv' fields must represent the VALUE OF THE ENTIRE CLAIM, not just the roofing section.
    4. Example: If "Total Roofing" is $700 but "TOTALS" is $1,400, the RCV is $1,400.

    RULES FOR "QUANTITIES":
    1. Correct OCR noise (e.g., "3100" -> "31.00").
    2. Respect the Unit of Measure. 
       - If text says "31.00 EA" (Each), report 31.00 with unit "EA". 
       - If text says "31.00 SQ" (Squares), report 31.00 with unit "SQ".
    3. Do not convert units unless explicitly clear (e.g. don't change EA to SQ).

    RULES FOR "MISSING ITEMS":
    Check for these REQUIRED roofing items. If missing, list them in 'missing_items':
    - Drip Edge
    - Starter Strip
    - Ridge Cap
    - Ice & Water Shield
    - Pipe Jack Flashing
    - Ridge Vent / Box Vent
    
    (Note: If this is a small repair estimate—indicated by units of "EA" or "LF" instead of "SQ"—be lenient with missing items.)

    Output strictly as JSON matching the schema.
    """

    user_prompt = f"Here is the raw OCR text:\n\n{raw_text}"

    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=ClaimData,
    )

    return response.choices[0].message.parsed
# ------------------------
# API ENDPOINT
# ------------------------
@app.post("/parse-estimate")
async def parse_estimate(request: Request):
    # 1. Get Binary
    binary = await request.body()
    if not binary:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # 2. OCR (The "Eyes")
    raw_text = extract_text_from_pdf(binary)
    
    # 3. AI Processing (The "Brain")
    # We use AI because insurance formats vary wildly (Farmers vs Allstate vs State Farm).
    # Regex is too brittle for production.
    parsed_data = analyze_with_llm(raw_text)

    # 4. Return JSON
    return parsed_data
