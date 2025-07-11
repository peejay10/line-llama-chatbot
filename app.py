import os
import json
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from flask import Flask, request, abort
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

load_dotenv()

LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')
GOOGLE_SHEET_URL = os.getenv('GOOGLE_SHEET_URL')

app = Flask(__name__)

@app.route("/")
def index():
    return "OK", 200

# setup Google Sheet
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
client = gspread.authorize(creds)
sheet_file = client.open_by_url(GOOGLE_SHEET_URL)
sheet_general = sheet_file.worksheet("General")
sheet_by_term = sheet_file.worksheet("ByTerm")
sheet_by_semester = sheet_file.worksheet("BySemester")

# load model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

waiting_for_term = {}
waiting_for_semester = {}

def generate_with_ollama(prompt):
    url = "http://localhost:11434/api/generate"
    data = {"model": "gemma", "prompt": prompt, "stream": False}
    try:
        res = requests.post(url, json=data)
        return res.json().get("response", "ขออภัย ระบบไม่สามารถตอบได้ครับ")
    except:
        return "ขออภัย ระบบไม่สามารถติดต่อ AI ได้ครับ"

def semantic_search(user_question, sheet, question_key='คำถาม'):
    data = sheet.get_all_records()
    questions = [row[question_key] for row in data if row[question_key]]
    embeddings = model.encode(questions, convert_to_tensor=True)
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(user_embedding, embeddings)[0]
    idx = scores.argmax().item()
    score = scores[idx].item()
    if score > 0.7:
        return data[idx], sheet.title
    return None, None

def reply_message(reply_token, message):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {LINE_CHANNEL_ACCESS_TOKEN}',
    }
    body = {
        'replyToken': reply_token,
        'messages': [{'type': 'text', 'text': message}],
    }
    requests.post('https://api.line.me/v2/bot/message/reply', headers=headers, json=body)

@app.route("/callback", methods=["POST"])
def callback():
    data = request.get_json()
    if not data:
        abort(400)
    events = data.get("events", [])
    for event in events:
        if event.get("type") == "message" and event["message"].get("type") == "text":
            user_message = event["message"]["text"].strip()
            reply_token = event["replyToken"]
            reply_message(reply_token, "ระบบกำลังพัฒนาครับ (Render ทดสอบ)")
    return 'OK', 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
