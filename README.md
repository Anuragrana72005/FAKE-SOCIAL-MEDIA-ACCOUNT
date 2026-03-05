# FAKE-SOCIAL-MEDIA-ACCOUNT
🛡️ Sentinel: AI-Powered Social Media Verification
Sentinel is an elite, full-stack intelligence tool designed to scan and verify the authenticity of social media accounts. By combining Random Forest Machine Learning with Deep Learning Vision (EasyOCR), Sentinel identifies bots and fraudulent profiles with institutional-grade accuracy.
Elite Features
Premium Visual Interface: A dark-luxury dashboard featuring frosted glassmorphism, golden accents, and high-fidelity animations.

Neural Classification: A Random Forest model that analyzes account telemetry (follower ratios, naming patterns, and account age) to detect anomalies.

Vision-Based Intelligence: Uses EasyOCR to extract text directly from screenshots, making it highly effective for manual profile audits.

Verified Badge Detection: OpenCV-powered color filtering that automatically identifies official "Blue Ticks," bypassing the ML model for 100% verified accounts.

Encrypted Threat Log: Integrated SQLite database that records every scan, timestamp, and metadata result for future reference.

Telemetry Spinner: A custom golden orbital loading animation that provides visual feedback during deep-scan processing.

🏗️ Technical Stack
Intelligence Backend: FastAPI (Python)

Machine Learning: Scikit-Learn

Computer Vision: EasyOCR & OpenCV

Data Management: SQLite3

Frontend Branding: Modern HTML5, CSS3 (Ultra-Rich Theme), and Vanilla JavaScript

🚀 Deployment Guide
1. Repository Setup
Bash
git clone https://github.com/your-username/sentinel-detector.git
cd sentinel-detector
2. Environment Installation
Install the required AI and backend libraries:

Bash
python -m pip install fastapi uvicorn scikit-learn pandas easyocr opencv-python-headless numpy python-multipart
3. Initialize the Backend
Run the server using the Python module command to avoid PATH errors:

Bash
python -m uvicorn main:app --reload
4. Client Access
Launch index.html in your browser. Ensure the backend is active to process live requests.

📊 Evaluation Logic
Ingestion: Data is received via search, manual input, or image upload.

Vision Pre-Processing: Images are scanned for verified blue pixels and text data.

Neural Scoring: The Random Forest model outputs a "Fake Probability" percentage.

Database Commit: The scan result is saved to sentinel_history.db.

📜 License
This project is licensed under the MIT License.
