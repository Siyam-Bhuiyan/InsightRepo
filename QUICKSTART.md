# InsightRepo - Quick Start Guide

## Prerequisites Verification

You already have Ollama installed with the required models. Let's verify:

```bash
ollama list
```

You should see:

- `qwen2.5-coder:7b`
- `nomic-embed-text`

## Backend Setup

### 1. Create Python Virtual Environment

```bash
cd backend
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows (PowerShell):**

```powershell
venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create Environment Configuration

Copy the example environment file:

```bash
copy .env.example .env
```

The `.env` file should contain:

```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5-coder:7b
EMBEDDING_MODEL=nomic-embed-text
HF_API_TOKEN=your_hf_token_here
MAX_CHUNKS=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

**Note:** HF_API_TOKEN is optional - only needed if you want to use Hugging Face cloud mode.

### 5. Verify Ollama is Running

```bash
ollama list
```

If Ollama isn't running, start it (it usually starts automatically on Windows).

### 6. Start Backend Server

```bash
uvicorn main:app --reload
```

The backend will be available at: http://localhost:8000
API documentation: http://localhost:8000/docs

---

## Frontend Setup

### 1. Install Node Dependencies

Open a **new terminal** and navigate to the frontend directory:

```bash
cd frontend
npm install
```

### 2. Start Development Server

```bash
npm run dev
```

The frontend will be available at: http://localhost:5173

---

## Testing the Application

### Test 1: Health Check

1. Open http://localhost:8000/docs
2. Try the `/health` endpoint
3. Verify `ollama_connected: true` and models are listed

### Test 2: Index a Small Repository

1. Open http://localhost:5173
2. Use the GitHub URL input
3. Try a small repository, e.g., `https://github.com/pallets/flask`
4. Wait for indexing to complete

### Test 3: Ask Questions

Try these questions:

- "What is the main entry point of this application?"
- "Where is routing handled?"
- "How are HTTP requests processed?"
- "What design patterns are used?"

### Test 4: Check Citations

Verify that:

- Each answer includes source code references
- File paths are shown
- Similarity scores are displayed
- Code snippets are properly formatted

---

## Project Structure

```
InsightRepo/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── rag_engine.py        # RAG pipeline
│   ├── repo_ingestion.py    # Repository cloning/extraction
│   ├── embedding_service.py # Ollama embeddings
│   ├── llm_service.py       # LLM inference
│   ├── models.py            # Pydantic models
│   ├── config.py            # Configuration
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── services/        # API client
│   │   ├── App.tsx          # Main application
│   │   └── main.tsx         # Entry point
│   └── package.json         # Node dependencies
└── README.md
```

---

## Troubleshooting

### Ollama Connection Issues

If you see "Ollama Disconnected" in the UI:

```bash
# Check if Ollama is running
ollama list

# Test Ollama directly
ollama run qwen2.5-coder:7b "Hello"
```

### Port Conflicts

If port 8000 or 5173 is in use:

**Backend:**

```bash
uvicorn main:app --reload --port 8001
```

**Frontend:**
Update `vite.config.ts` to change the port.

### Import Errors

If you get import errors in Python:

```bash
# Ensure virtual environment is activated
pip install --upgrade -r requirements.txt
```

### CORS Errors

The backend is configured to allow requests from `http://localhost:5173` and `http://localhost:3000`. If using a different port, update `config.py`:

```python
allowed_origins: list[str] = ["http://localhost:5173", "http://localhost:3000", "http://localhost:YOUR_PORT"]
```

---

## Next Steps

1. **Try Different Repositories**: Test with various codebases
2. **Experiment with Questions**: See how RAG retrieval works
3. **Adjust Parameters**: Modify chunk size, overlap, and max chunks in `.env`
4. **Add Hugging Face**: Get an API token from https://huggingface.co/settings/tokens
5. **Deploy**: Consider containerizing with Docker

---

## Development Tips

### Backend Development

```bash
# Run with auto-reload
uvicorn main:app --reload

# Check logs
# Logs appear in the terminal where uvicorn is running
```

### Frontend Development

```bash
# Development mode with hot reload
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Testing Embeddings

```bash
# Test Ollama embeddings directly
curl http://localhost:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "test code snippet"
}'
```

---

## Performance Considerations

- **Large Repositories**: Indexing can take several minutes for repos with 1000+ files
- **Query Time**: First query may be slower as models load into memory
- **Memory Usage**: Ollama models require ~8GB RAM
- **Storage**: FAISS indexes are stored in `backend/vector_stores/`

---

## Features Implemented

✅ GitHub repository cloning
✅ ZIP file upload
✅ Language-aware code chunking (Python, JavaScript, TypeScript, etc.)
✅ Vector embeddings with Ollama (nomic-embed-text)
✅ FAISS vector similarity search
✅ LLM inference (Ollama + Hugging Face)
✅ RAG-based question answering
✅ Source code citations with similarity scores
✅ Real-time indexing progress
✅ React UI with Tailwind CSS
✅ RESTful API with FastAPI

---

Enjoy exploring your codebases with AI! 🚀
