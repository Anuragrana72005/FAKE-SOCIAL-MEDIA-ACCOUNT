import io
import re
import cv2
import random
import numpy as np
import pandas as pd
import easyocr
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="Fake Account Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading EasyOCR AI models...")
reader = easyocr.Reader(['en'], gpu=False) 

# --- ML Model Training ---
data = {
    'follower_count': [150, 10, 2000, 2, 500, 0, 1400000, 5000000, 2000000], 
    'following_count': [150, 5000, 400, 3000, 450, 1000, 513, 10, 0], 
    'has_profile_pic': [1, 0, 1, 0, 1, 0, 1, 1, 1],
    'username_digits': [0, 6, 2, 8, 0, 5, 0, 0, 0],
    'account_age_days': [1200, 5, 3000, 2, 800, 1, 2500, 3000, 4000],
    'is_fake': [0, 1, 0, 1, 0, 1, 0, 0, 0] 
}
df = pd.DataFrame(data)
X = df[['follower_count', 'following_count', 'has_profile_pic', 'username_digits', 'account_age_days']]
y = df['is_fake']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# --- Core Logic ---
def make_prediction(stats: dict, badge_detected: bool = False):
    if badge_detected:
        return {"prediction": "Real Account", "fake_probability": 0.0, "scanned_stats": stats, "badge_detected": "Yes"}
        
    input_data = pd.DataFrame([stats])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    return {
        "prediction": "Fake Account" if prediction == 1 else "Real Account",
        "fake_probability": round(probability * 100, 2),
        "scanned_stats": stats,
        "badge_detected": "No"
    }

def fetch_api_data(username: str):
    """Simulates API fetch and generates a visual profile card"""
    is_likely_fake = random.choice([True, False])
    
    if is_likely_fake:
        # Generate bot profile UI data
        profile = {
            "name": "User_" + str(random.randint(1000, 9999)),
            "bio": "No bio yet.",
            "pic_url": f"https://ui-avatars.com/api/?name={username}&background=random" # Generates a text avatar
        }
        stats = {'follower_count': random.randint(0, 50), 'following_count': random.randint(1000, 7000), 'has_profile_pic': 0, 'username_digits': sum(c.isdigit() for c in username), 'account_age_days': random.randint(1, 10)}
    else:
        # Generate real user profile UI data
        profile = {
            "name": username.capitalize() + " Official",
            "bio": "Welcome to my official page! ✨",
            "pic_url": f"https://picsum.photos/seed/{username}/200/200" # Generates a random person's face based on username
        }
        stats = {'follower_count': random.randint(500, 50000), 'following_count': random.randint(100, 800), 'has_profile_pic': 1, 'username_digits': sum(c.isdigit() for c in username), 'account_age_days': random.randint(500, 3000)}
        
    return stats, profile

def detect_blue_tick(cv2_image):
    hsv = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([90, 100, 100]), np.array([130, 255, 255]))
    return cv2.countNonZero(mask) > 50

def extract_stats_from_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    is_verified = detect_blue_tick(img)
    
    try:
        results = reader.readtext(img)
        text = " ".join([res[1] for res in results]).lower()
        
        f_match = re.search(r'([\d\.,]+[km]?)\s*followers?', text)
        fw_match = re.search(r'([\d\.,]+[km]?)\s*following?', text)

        def parse_number(match_str):
            if not match_str: return 0
            clean = match_str.replace(',', '').strip()
            mult = 1000 if 'k' in clean else (1000000 if 'm' in clean else 1)
            try: return int(float(clean.replace('k','').replace('m','')) * mult)
            except: return 0

        stats = {'follower_count': parse_number(f_match.group(1) if f_match else None), 'following_count': parse_number(fw_match.group(1) if fw_match else None), 'has_profile_pic': 1, 'username_digits': 0, 'account_age_days': 1000}
    except:
        stats = {'follower_count': 0, 'following_count': 0, 'has_profile_pic': 0, 'username_digits': 0, 'account_age_days': 0}
    return stats, is_verified

# --- API Endpoints ---
class AccountData(BaseModel):
    follower_count: int
    following_count: int
    has_profile_pic: int
    username_digits: int
    account_age_days: int

@app.post("/predict_manual")
def predict_manual(data: AccountData):
    return make_prediction(data.model_dump())

@app.post("/predict_search")
def predict_search(username: str = Form(...)):
    stats_dict, profile_dict = fetch_api_data(username)
    result = make_prediction(stats_dict)
    # Attach the profile UI data to the response
    result["profile_details"] = profile_dict
    result["searched_username"] = username
    return result

@app.post("/predict_upload")
async def predict_upload(file: UploadFile = File(...)):
    image_bytes = await file.read()
    stats_dict, is_verified = extract_stats_from_image(image_bytes)
    return make_prediction(stats_dict, badge_detected=is_verified)