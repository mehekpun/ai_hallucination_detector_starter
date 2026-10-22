AI HALLUCINATION DETECTOR - BEGINNER SETUP

1. Make sure Python is installed.
2. Open this folder in VS Code.
3. Open Terminal > New Terminal.
4. Run: python -m venv .venv
5. Activate the environment:
   Windows PowerShell: .venv\Scripts\Activate.ps1
   If PowerShell blocks activation, run: .venv\Scripts\python.exe -m pip install -r requirements.txt
6. Install dependencies:
   python -m pip install -r requirements.txt
   (If activation did not work, use .venv\Scripts\python.exe -m pip install -r requirements.txt)
7. Rename .env.example to .env.
8. Open .env and replace both placeholder values with your own API keys.
9. Run: python -m uvicorn main:app --reload
   (If activation did not work, run: .venv\Scripts\python.exe -m uvicorn main:app --reload)
10. In your browser, open http://127.0.0.1:8000

Do not share the .env file or push it to GitHub. .gitignore excludes it.

This is a learning prototype. The verdicts can be wrong; always inspect the linked evidence.
