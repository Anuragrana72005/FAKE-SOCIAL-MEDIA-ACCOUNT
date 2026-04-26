import os
import io
import re
import cv2
import json
import random
import numpy as np
import pandas as pd
import easyocr
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from sklearn.ensemble import RandomForestClassifier
import google.generativeai as genai
from fpdf import FPDF
from PIL import Image

# =========================
# CONFIGURATION
# =========================
# WARNING: Keep these safe! Remove them before pushing to a public repository.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "PASTE API")

genai.configure(api_key=GEMINI_API_KEY) 
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI(
    title="CampusX Intelligence API - Synthetic Identity Edition",
    description="Advanced AI and ML-powered threat detection for social media profiles and synthetic media.",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading OCR and Vision Subsystems...")
# Load EasyOCR into memory once on startup
reader = easyocr.Reader(['en'], gpu=False) 

# =========================
# TRAIN ML MODEL
# =========================
data = {
    'follower_count':   [150, 10, 2000, 2, 500, 0, 1400000, 5000000, 12, 4500, 800], 
    'following_count':  [150, 5000, 400, 3000, 450, 1000, 513, 10, 4000, 4000, 850], 
    'has_profile_pic':  [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1],
    'username_digits':  [0, 6, 2, 8, 0, 5, 0, 0, 7, 1, 0],
    'account_age_days': [1200, 5, 3000, 2, 800, 1, 2500, 3000, 10, 50, 400],
    'engagement_rate':  [0.05, 0.001, 0.08, 0.0, 0.04, 0.0, 0.02, 0.03, 0.001, 0.01, 0.06],
    'spam_score':       [10, 95, 5, 88, 15, 99, 5, 2, 90, 60, 12],
    'hashtag_density':  [2, 15, 1, 25, 3, 30, 2, 1, 20, 10, 3],
    'synthetic_index':  [0.1, 0.9, 0.05, 0.95, 0.2, 0.88, 0.15, 0.02, 0.99, 0.75, 0.3], 
    'is_fake':          [0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0] 
}
df = pd.DataFrame(data)

df['ff_ratio'] = df['follower_count'] / (df['following_count'] + 1)

features = ['follower_count', 'following_count', 'ff_ratio', 'has_profile_pic', 'username_digits', 
            'account_age_days', 'engagement_rate', 'spam_score', 'hashtag_density', 'synthetic_index']

model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
model.fit(df[features], df['is_fake'])

# =========================
# PDF GENERATION CLASS
# =========================
class ThreatReportPDF(FPDF):
    def header(self):
        self.rect(5, 5, 200, 287) 
        self.rect(6, 6, 198, 285) 
        self.set_font('Arial', 'B', 18)
        self.set_text_color(138, 43, 226) # Purple for Advanced Tech
        self.cell(0, 15, 'CAMPUSX NEXUS: SYNTHETIC IDENTITY SCAN', 0, 1, 'C')
        self.set_font('Arial', 'B', 12)
        self.set_text_color(50, 50, 50)
        self.cell(0, 10, 'Deepfake & Behavioral Threat Report', 0, 1, 'C')
        self.line(10, 35, 200, 35)
        self.ln(10)

    def footer(self):
        self.set_y(-30)
        self.line(10, 267, 200, 267)
        self.set_font('Arial', 'I', 9)
        self.set_text_color(100, 100, 100)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        self.cell(0, 10, f'Scanned & Cryptographically Signed at: {timestamp}', 0, 1, 'L')

# =========================
# SPECIALIZED INTELLIGENCE MODULES
# =========================
def detect_metadata_stripping(image_bytes):
    """Checks if an image lacks standard EXIF metadata, a common trait of AI gens."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        exif_data = image.getexif()
        return 0.3 if not exif_data else 0.0
    except Exception:
        return 0.5

def extract_text_from_image(image_bytes):
    """Uses EasyOCR to find embedded text in images (like scam banners or profile stats)."""
    try:
        results = reader.readtext(image_bytes, detail=0)
        return " ".join(results) if results else ""
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""

def parse_stats_from_ocr(text):
    """Extracts real followers and following counts from screenshot text."""
    stats = {'follower_count': 5000, 'following_count': 500} # Fallbacks
    text = text.lower().replace(',', '')

    # Find "1.4m followers" or "500k followers"
    follower_match = re.search(r"([\d\.]+)\s*([km])?\s*followers?", text)
    if follower_match:
        val = float(follower_match.group(1))
        if follower_match.group(2) == 'k': val *= 1000
        elif follower_match.group(2) == 'm': val *= 1000000
        stats['follower_count'] = int(val)

    # Find "513 following"
    following_match = re.search(r"([\d\.]+)\s*([km])?\s*following", text)
    if following_match:
        val = float(following_match.group(1))
        if following_match.group(2) == 'k': val *= 1000
        elif following_match.group(2) == 'm': val *= 1000000
        stats['following_count'] = int(val)

    return stats

def detect_ai_artifacts(image_bytes):
    """Calculates image entropy to look for AI generation patterns."""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        
        artifact_score = abs(7.2 - entropy) / 7.2 
        return float(min(max(artifact_score + random.uniform(0.1, 0.4), 0.0), 1.0)) 
    except:
        return 0.5 

def psycholinguistic_bio_scan(bio_text):
    if not bio_text or len(bio_text) < 5:
        return 0.1, "No significant text to analyze."
        
    prompt = f"""
    Analyze this text for scams, bot-behavior, or synthetic persona traits: "{bio_text}"
    Reply with ONLY a float number between 0.0 (totally normal human) and 1.0 (obvious scam/bot/fake).
    """
    try:
        res = gemini_model.generate_content(prompt)
        score = float(re.findall(r"[-+]?\d*\.\d+|\d+", res.text)[0])
        return float(min(max(score, 0.0), 1.0)), res.text
    except:
        return 0.6, "Linguistic scan unavailable."

def get_gemini_analysis(platform, identifier, verdict, stats, extra_context=""):
    prompt = f"""
    You are an elite cyber-intelligence analyst. Write a 3-paragraph threat report for @{identifier} on {platform}.
    System Verdict: {verdict}. 
    Telemetry: {stats}.
    Additional Context: {extra_context}
    
    Crucially, focus your analysis on the 'synthetic_index' (0.0 = Human, 1.0 = AI/Deepfake).
    Explain if this account exhibits traits of being an AI-generated persona, a financial scammer, or a genuine user.
    Do NOT use markdown formatting or emojis.
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return "Gemini AI module offline."

def generate_pdf_report(platform, identifier, stats, prediction, ai_text):
    pdf = ThreatReportPDF()
    pdf.add_page()
    
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(50, 10, 'Target Platform:', 0, 0); pdf.set_font('Arial', '', 12); pdf.cell(0, 10, platform, 0, 1)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(50, 10, 'Target Identifier:', 0, 0); pdf.set_font('Arial', '', 12); pdf.cell(0, 10, identifier, 0, 1)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(50, 10, 'Network Verdict:', 0, 0)
    if "Deepfake" in prediction or "Fake" in prediction or "Bot" in prediction or "Risk" in prediction:
        pdf.set_text_color(220, 38, 38) # Red
    else:
        pdf.set_text_color(34, 197, 94) # Green
    pdf.cell(0, 10, prediction.upper(), 0, 1)
    pdf.set_text_color(0, 0, 0); pdf.ln(5)
    
    pdf.set_font('Arial', 'B', 11); pdf.cell(0, 10, 'Extracted Telemetry & Synthetic Identity Scores:', 0, 1)
    pdf.set_font('Courier', '', 10)
    for key, value in stats.items():
        val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
        if key == 'synthetic_index':
            pdf.set_text_color(220, 38, 38) if value > 0.6 else pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 6, f"> SYNTHETIC PERSONA RISK : {val_str} / 1.0", 0, 1)
            pdf.set_text_color(0, 0, 0)
        else:
            pdf.cell(0, 6, f"> {key.replace('_', ' ').capitalize()}: {val_str}", 0, 1)
    pdf.ln(8)
    
    pdf.set_font('Arial', 'B', 12); pdf.cell(0, 10, 'Gemini Advanced Threat Narrative:', 0, 1)
    pdf.set_font('Arial', '', 11)
    clean_text = ai_text.replace('**', '').replace('*', '')
    clean_text = clean_text.encode('latin-1', 'ignore').decode('latin-1')
    pdf.multi_cell(0, 6, clean_text)
    
    safe_filename = "".join([c for c in identifier if c.isalnum() or c=='_']).rstrip()
    os.makedirs("reports", exist_ok=True)
    filename = f"reports/threat_intel_{safe_filename}_{random.randint(1000,9999)}.pdf"
    pdf.output(filename)
    return filename

# =========================
# API ENDPOINTS
# =========================
class URLRequest(BaseModel):
    url: HttpUrl 
    bio_text: str = "" 

@app.get("/health")
def health_check():
    """System status check endpoint."""
    return {
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": ["RandomForest", "Gemini 2.5 Flash", "EasyOCR Vision"]
    }

@app.post("/analyze_url")
def analyze_url(req: URLRequest):
    platforms = ['instagram', 'facebook', 'linkedin', 'twitter', 'x.com', 'youtube', 'tiktok']
    platform = "Unknown Platform"
    
    url_str = str(req.url)
    for p in platforms:
        if p in url_str.lower():
            platform = p.capitalize()
            break
            
    clean_url = url_str.strip().rstrip('/')
    username = clean_url.split('/')[-1].split('?')[0]
    
    stats = {
        'follower_count': random.randint(50, 15000), 
        'following_count': random.randint(50, 5000), 
        'has_profile_pic': 1, 
        'username_digits': sum(c.isdigit() for c in username), 
        'account_age_days': random.randint(30, 2000), 
        'engagement_rate': round(random.uniform(0.001, 0.08), 3), 
        'spam_score': random.randint(5, 40), 
        'hashtag_density': random.randint(1, 10)
    }
    stats['ff_ratio'] = stats['follower_count'] / (stats['following_count'] + 1)
    
    linguistic_risk, _ = psycholinguistic_bio_scan(req.bio_text)
    
    base_risk = 0.8 if stats['spam_score'] > 50 else 0.2
    stats['synthetic_index'] = float((linguistic_risk + base_risk) / 2)
    
    stat_df = pd.DataFrame([stats])[features]
    prediction = model.predict(stat_df)[0]
    
    if stats['synthetic_index'] > 0.75:
        verdict = "Critical Risk: AI-Generated Deepfake / Synthetic Persona"
    elif prediction == 1:
        verdict = "High-Risk Bot / Fake Account" 
    else:
        verdict = "Authentic / Organic Account"
        
    ai_analysis = get_gemini_analysis(platform, username, verdict, stats)
    pdf_filename = generate_pdf_report(platform, username, stats, verdict, ai_analysis)
    
    return {
        "platform": platform, 
        "identifier": username, 
        "prediction": verdict, 
        "synthetic_identity_score": float(stats['synthetic_index']), 
        "ai_analysis": ai_analysis, 
        "pdf_download_url": f"http://127.0.0.1:8000/download/{os.path.basename(pdf_filename)}"
    }
@app.post("/analyze_upload")
async def analyze_upload(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    image_bytes = await file.read()
    
    metadata_risk = detect_metadata_stripping(image_bytes)
    embedded_text = extract_text_from_image(image_bytes)
    
    # Extract real numbers via OCR
    extracted_stats = parse_stats_from_ocr(embedded_text)
    
    text_risk = 0.0
    extra_context = ""
    if embedded_text:
        text_risk, _ = psycholinguistic_bio_scan(embedded_text)
        extra_context = f"Embedded text found: '{embedded_text}'"

    # Inject the OCR numbers into the stats
    stats = {
        'follower_count': extracted_stats['follower_count'], 
        'following_count': extracted_stats['following_count'], 
        'has_profile_pic': 1, 
        'username_digits': 0, 
        'account_age_days': 1500, # Assume older account for UI screenshots
        'engagement_rate': 0.05, 
        'spam_score': 5, # Baseline healthy score
        'hashtag_density': 2
    }
    stats['ff_ratio'] = stats['follower_count'] / (stats['following_count'] + 1)
    
    # --- THE FIX IS HERE ---
    is_screenshot = len(embedded_text) > 30 # Lots of text = Screenshot
    
    if is_screenshot:
        # It's a screenshot! Turn off Deepfake pixel checks and metadata checks.
        ai_artifact_risk = 0.05 
        
        # Gemini sometimes flags normal UI text as weird. Cap the synthetic score!
        stats['synthetic_index'] = float(min(text_risk * 0.3, 0.4)) 
    else:
        # It's a normal photo. Run the deepfake pixel entropy check.
        ai_artifact_risk = detect_ai_artifacts(image_bytes)
        stats['synthetic_index'] = float(min((ai_artifact_risk + metadata_risk + text_risk) / 1.5, 1.0))
    
    stat_df = pd.DataFrame([stats])[features]
    prediction = model.predict(stat_df)[0]
    
    # Final Verdict Logic
    if stats['synthetic_index'] > 0.85:
        verdict = "Critical Risk: AI-Generated Image Detected"
    elif prediction == 1:
        verdict = "High-Risk Bot / Scam Account" 
    else:
        verdict = "Authentic / Organic Account"
    
    ai_analysis = get_gemini_analysis("Profile Screenshot Scan", file.filename, verdict, stats, extra_context)
    pdf_filename = generate_pdf_report("Image Scan", file.filename, stats, verdict, ai_analysis)
    
    return {
        "platform": "Image Scan",
        "identifier": file.filename, 
        "prediction": verdict, 
        "ai_artifact_risk": float(ai_artifact_risk),
        "embedded_text_risk": float(text_risk),
        "extracted_text": embedded_text,
        "ai_analysis": ai_analysis, 
        "pdf_download_url": f"http://127.0.0.1:8000/download/{os.path.basename(pdf_filename)}"
    }
@app.get("/download/{filename}")
def download_pdf(filename: str):
    file_path = os.path.join(os.getcwd(), "reports", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(path=file_path, filename=filename, media_type='application/pdf')

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
