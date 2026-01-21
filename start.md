cd backend
python -m venv backend\.venv
python.exe -m pip install --upgrade pip setuptools wheel
requirements.txt
python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8000

cd frontend
npm ci

npm start