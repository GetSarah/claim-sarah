import os
import pytesseract
from fastapi import FastAPI, Request, HTTPException
from pdf2image import convert_from_bytes
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import json

# Initialize FastAPI
app = FastAPI()

# Initialize OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ------------------------
# 1. UPDATED DATA MODELS
# ------------------------

class LineItem(BaseModel):
    description: str
    quantity: float = 0.0
    unit: str = "EA"
    total_price: float = 0.0

class AuditFlag(BaseModel):
    category: Literal["Code & Safety", "Flashing & Trim", "Waste Leak", "Labor & Overhead"]
    item_name: str
    issue_type: Literal["MISSING", "ZERO_VALUE"]
    severity: Literal["HIGH", "MEDIUM"]
    reasoning: str = Field(description="Why this is flagged based on the winning algorithm (e.g., 'Starter shingles cannot be cut from waste')")

class ClaimData(BaseModel):
    # Financials (The "Nuclear" Money Extraction)
    carrier: Optional[str] = None
    claim_number: Optional[str] = None
    insured_name: Optional[str] = None
    property_address: Optional[str] = None
    rcv: float = Field(0.0, description="Replacement Cost Value - usually Grand Total")
    acv: float = Field(0.0, description="Actual Cash Value")
    deductible: float = 0.0
    depreciation: float = 0.0
    roof_squares: float = 0.0
    
    # The Audit
    audit_flags: List[AuditFlag] = Field(description="List of specific missed or underpaid items based on the Sarah Algorithm")
    
    # Raw Line Items (for verification)
    line_items: List[LineItem]

    summary: str

# ------------------------
# 2. HELPER: OCR (Unchanged)
# ------------------------
def extract_text_from_pdf(binary: bytes) -> str:
    text_chunks = []
    try:
        images = convert_from_bytes(binary, dpi=300)
        for i, img in enumerate(images):
            img = img.convert('L') 
            page_text = pytesseract.image_to_string(img)
            text_chunks.append(f"--- PAGE {i+1} ---\n{page_text}")
        return "\n".join(text_chunks)
    except Exception as e:
        print(f"OCR Error: {e}")
        if text_chunks:
            return "\n".join(text_chunks)
        raise HTTPException(status_code=500, detail="Failed to process PDF file.")

# ------------------------
# 3. HELPER: SARAH'S WINNING ALGORITHM (PROMPT V2)
# ------------------------
def analyze_with_llm(raw_text: str) -> ClaimData:
    
    system_prompt = """
    You are 'Sarah', an elite Insurance Claim Audit Agent. Your job is to extract financial data and AUDIT the estimate for missed profitability items.

    ### PART 1: FINANCIAL EXTRACTION (The "Nuclear" Method)
    1. FIND THE SUMMARY TABLE: Usually at the end ("Totals", "Net Claim").
    2. RCV (Replacement Cost Value): The Grand Total / Largest number at the bottom.
    3. ACV (Actual Cash Value): The "Net Actual Cash Value Payment" or Total minus Depreciation.
    4. ROOF SQUARES: Look for "Total SQ" or calculate based on removal of shingles (SQ quantity).

    ### PART 2: THE "WINNING ALGORITHM" AUDIT
    Scan the line items specifically for the following 4 categories. If an item is NOT found, or found with a $0 value, create an 'AuditFlag'.

    CATEGORY 1: CODE & SAFETY (Non-Negotiable)
    - Ice & Water Shield: Look for "Ice & Water", "Leak Barrier". (Flag if MISSING or $0).
    - Drip Edge: Look for "Drip Edge", "Gutter Apron". (Flag if MISSING).
    - Ventilation: Look for "Ridge Vent", "Box Vent", "Turtle Vent". (Flag if MISSING - codes require upgrades).
    - Double Underlayment: Check if roof pitch is mentioned as low slope. (Flag if MISSING on low slope).

    CATEGORY 2: FLASHING & TRIM (Structural Integrity)
    - Step Flashing: Look for "Step Flash". (Flag if MISSING - Reuse is bad practice).
    - Head/End Wall Flashing: Look for "Apron", "Head Wall". (Flag if MISSING).
    - Kickout Flashing: Look for "Kickout", "Diverter". (Flag if MISSING - Critical code item).
    - Gable Cornice/Returns: Look for "Cornice", "Returns". (Flag if MISSING).

    CATEGORY 3: WASTE LEAKS (Hidden Profit)
    - Starter Shingles: Look for "Starter". Adjusters try to include this in waste. (Flag if MISSING).
    - Ridge/Hip Caps: Look for "Ridge Cap", "Hip Cap". Adjusters try to include this in waste. (Flag if MISSING).

    CATEGORY 4: LABOR & IF-INCURRED (High Value)
    - Permits & Fees: Look for "Permit". (Flag if MISSING or $0).
    - Overhead & Profit (O&P): Look for "O&P", "Overhead". (Flag if MISSING on complex jobs).
    - Interior/Painting: Look for interior repairs. (Flag if MISSING).
    - Clean-up/Landscaping: Look for "Dumps", "Clean up", "Protect Landscaping". (Flag if MISSING).

    ### OUTPUT RULES
    - Extract ALL line items found in the OCR text.
    - Generate 'audit_flags' strictly based on the algorithm above.
    - If an item is present and paid correctly, DO NOT flag it.
    """

    user_prompt = f"Here is the raw OCR text from the estimate:\n\n{raw_text}"

    # Utilizing GPT-4o Structured Outputs
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
# 4. API ENDPOINT
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
    parsed_data = analyze_with_llm(raw_text)

    return parsed_data
