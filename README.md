# AG News Classifier

A web-based dashboard for classifying news articles into four categories — **World**, **Sports**, **Business**, and **Sci/Tech** — using the AG News dataset. Built as part of the Artificial Intelligence course (COMP338).

## Overview

The application trains two machine learning models (Logistic Regression and Decision Tree) on the AG News dataset and displays real-time training progress, accuracy metrics, classification reports, and a decision tree visualization — all through an interactive React dashboard.

## Tech Stack

- **Frontend:** React 19, Vite, Framer Motion, Lucide Icons
- **Backend:** FastAPI, scikit-learn, pandas, matplotlib
- **Dataset:** [AG News](https://huggingface.co/datasets/wangrongsheng/ag_news) (via HuggingFace Datasets)

## Features

- One-click model training with live terminal-style logs
- Logistic Regression and Decision Tree classifiers
- Per-class precision, recall, and F1 scores
- Decision tree visualization (top 3 levels)
- PDF document viewer for project description and report
- Team "About Us" modal

## Project Structure

```
├── backend/
│   └── api.py            # FastAPI backend with training & prediction endpoints
├── frontend/
│   ├── src/
│   │   ├── App.jsx       # Main React application
│   │   └── App.css       # Styles
│   ├── public/           # Static assets (team photos)
│   └── vite.config.js    # Vite configuration
├── main.py               # Standalone training script (CLI)
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+

### Backend

```bash
pip install fastapi uvicorn scikit-learn pandas matplotlib datasets
cd backend
uvicorn api:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```
