# Arcade Chat App

A Streamlit-based chat application that uses the Arcade Python SDK to create an interactive chat experience.

## Setup

1. Install dependencies using `uv`:

```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
uv pip install -r requirements.txt
```

2. Set up your environment variables:

```bash
cp .env.example .env
```

Then edit `.env` and add your Arcade API key.

## Running the Application

To run the chat application:

```bash
streamlit run src/chat_app.py
```

The application will open in your default web browser. You can start chatting with the Arcade AI assistant right away!

## Features

- Clean, modern chat interface
- Real-time responses from Arcade AI
- Message history persistence during session
- Responsive design that works on desktop and mobile

## Requirements

- Python 3.8+
- Arcade API key
- Dependencies listed in `requirements.txt`
