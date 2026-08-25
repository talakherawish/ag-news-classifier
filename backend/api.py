from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import pandas as pd
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
from datasets import load_dataset
import threading
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import FileResponse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOWED_FILES = {"AICourseProject2_Report.pdf", "COMP338_Project2.pdf"}

@app.get("/files/{file_name}")
async def get_file(file_name: str):
    if file_name not in ALLOWED_FILES:
        raise HTTPException(status_code=404, detail="File not found")
    file_path = os.path.join(BASE_DIR, file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")

state = {
    "is_training": False,
    "logs": [],
    "results": None,
    "lr_model": None,
    "dt_model": None,
    "vectorizer": None,
    "plot_url": None,
}

class PredictRequest(BaseModel):
    text: str

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_training():
    global state
    try:
        state["is_training"] = True
        state["logs"] = []
        state["results"] = None
        
        state["logs"].append("Loading dataset...")

        dataset = load_dataset("wangrongsheng/ag_news") #loading news dataset
        df = pd.DataFrame(dataset["train"])
        
        state["logs"].append("Preprocessing text...")
        df["text"] = df["text"].apply(clean_text) #cleanng text
        
        state["logs"].append("Vectorizing text with TF-IDF...") # vectorization
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        X = vectorizer.fit_transform(df["text"])
        y = df["label"]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        state["logs"].append("Training Logistic Regression model...")
        # Logistic Regression
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        lr_acc = accuracy_score(y_test, y_pred_lr)
        lr_report = classification_report(y_test, y_pred_lr, output_dict=True)
        
        state["logs"].append("Training Decision Tree model...")
        # Decision Tree
        dt = DecisionTreeClassifier(max_depth=20, random_state=42)
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test)
        dt_acc = accuracy_score(y_test, y_pred_dt)
        dt_report = classification_report(y_test, y_pred_dt, output_dict=True)
        
        state["logs"].append("Rendering Decision Tree (top 3 levels)...")
        plt.figure(figsize=(20, 10))
        plot_tree(
            dt,
            max_depth=3,
            feature_names=vectorizer.get_feature_names_out(),
            class_names=["World", "Sports", "Business", "Sci/Tech"],
            filled=True,
            rounded=True
        )
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        state["lr_model"] = lr
        state["dt_model"] = dt
        state["vectorizer"] = vectorizer
        state["plot_url"] = f"data:image/png;base64,{plot_base64}"
        state["results"] = {
            "logistic_regression": {
                "accuracy": lr_acc,
                "report": lr_report
            },
            "decision_tree": {
                "accuracy": dt_acc,
                "report": dt_report
            }
        }
        state["logs"].append("Training Complete! View results below.")
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        state["is_training"] = False

@app.post("/train")
async def train_model():
    if state["is_training"]:
        return {"message": "Training already in progress"}
    
    # Run in background
    thread = threading.Thread(target=run_training)
    thread.start()
    return {"message": "Training started"}

@app.post("/reset")
async def reset_state():
    global state
    state = {
        "is_training": False,
        "logs": [],
        "results": None,
        "lr_model": None,
        "dt_model": None,
        "vectorizer": None,
        "plot_url": None,
    }
    return {"message": "State reset successfully"}

@app.get("/status")
async def get_status():
    return {
        "is_training": state["is_training"],
        "logs": state["logs"],
        "has_results": state["results"] is not None,
        "results": state["results"],
        "plot_url": state["plot_url"]
    }

@app.post("/predict")
async def predict(request: PredictRequest):
    if not state["lr_model"] or not state["vectorizer"]:
        raise HTTPException(status_code=400, detail="Models not trained yet")
    
    cleaned = clean_text(request.text)
    vec = state["vectorizer"].transform([cleaned])
    
    lr_pred = int(state["lr_model"].predict(vec)[0])
    dt_pred = int(state["dt_model"].predict(vec)[0])
    
    categories = ["World", "Sports", "Business", "Sci/Tech"]
    
    return {
        "logistic_regression": categories[lr_pred],
        "decision_tree": categories[dt_pred]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
