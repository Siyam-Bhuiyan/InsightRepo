# 🎉 InsightRepo - System is Running!

## ✅ Current Status

Both servers are running successfully:

- **Backend API**: http://localhost:8000
- **Frontend UI**: http://localhost:5173
- **API Documentation**: http://localhost:8000/docs

## 🚀 How to Use InsightRepo

### Step 1: Load a Repository

You have two options to load a codebase:

#### Option A: GitHub URL

1. Open http://localhost:5173
2. Click the "GitHub URL" tab
3. Enter a repository URL, for example:
   - `https://github.com/pallets/flask` (Small Python web framework)
   - `https://github.com/fastapi/fastapi` (Python async framework)
   - `https://github.com/vercel/next.js` (React framework)
4. Click "Clone Repository"

#### Option B: Upload ZIP

1. Click the "Upload ZIP" tab
2. Select a ZIP file containing source code
3. Upload will automatically start indexing

### Step 2: Wait for Indexing

- Watch the progress bar as files are processed
- Indexing time depends on repository size:
  - Small repos (50-100 files): 30-60 seconds
  - Medium repos (500 files): 2-5 minutes
  - Large repos (1000+ files): 5-15 minutes
- You'll see "Repository indexed successfully!" when ready

### Step 3: Ask Questions

Once indexing is complete, you can ask questions like:

**Architecture Questions:**

- "What is the overall architecture of this application?"
- "How are the main components organized?"
- "What design patterns are used?"

**Implementation Questions:**

- "Where is authentication handled?"
- "How does data flow from the API to the database?"
- "Which files implement the user service?"
- "Where are API routes defined?"

**Code Understanding:**

- "How is error handling implemented?"
- "What are the main entry points?"
- "Which files handle database queries?"
- "How are requests validated?"

### Step 4: Review Citations

Each answer includes:

- **File Path**: Exact location of relevant code
- **Language**: Programming language detected
- **Similarity Score**: How relevant the code is to your question
- **Code Snippet**: Actual code that was used to generate the answer

You can verify every answer by checking the cited source code!

## 🔧 Advanced Features

### LLM Mode Selection

Choose between two inference modes:

**Ollama (Local) - RECOMMENDED**

- Uses your local qwen2.5-coder:7b model
- Fast response times
- Fully private - no data leaves your machine
- Best for development and testing

**Hugging Face (Cloud)**

- Uses cloud-based inference API
- Requires HF_API_TOKEN in backend/.env
- Useful for deployment scenarios
- Get token at: https://huggingface.co/settings/tokens

### RAG Configuration

You can adjust retrieval settings in `backend/.env`:

```env
MAX_CHUNKS=5         # Number of code snippets to retrieve
CHUNK_SIZE=1000      # Size of each code chunk
CHUNK_OVERLAP=200    # Overlap between chunks
```

**Recommendations:**

- Increase MAX_CHUNKS (to 7-10) for more comprehensive answers
- Decrease CHUNK_SIZE (to 500-800) for more focused results
- Increase CHUNK_OVERLAP (to 300) for better context preservation

## 📊 Example Workflow

### Example: Analyzing Flask

1. **Load Repository**

   ```
   URL: https://github.com/pallets/flask
   ```

2. **Wait for Indexing**

   ```
   Processed Files: 87 / 87
   Status: Completed ✓
   ```

3. **Ask Questions**

   ```
   Q: "Where is routing handled in Flask?"
   A: Routing in Flask is handled in the flask/app.py file through the Flask class...

   Citations:
   - File: flask/app.py (Similarity: 94.3%)
   - File: flask/helpers.py (Similarity: 87.6%)
   ```

4. **Follow-up Questions**
   ```
   Q: "How does Flask handle request context?"
   Q: "Where are blueprints implemented?"
   Q: "How is the application factory pattern used?"
   ```

## 🛠️ Troubleshooting

### Backend Issues

**Ollama Not Connected:**

```bash
# Check if Ollama is running
ollama list

# Restart Ollama service
# On Windows, restart the Ollama application
```

**Import Errors:**

```bash
cd backend
C:/Users/User/Documents/GitHub/InsightRepo/.venv/Scripts/python.exe -m pip install --upgrade -r requirements.txt
```

**Port Already in Use:**

```bash
# Change backend port
C:/Users/User/Documents/GitHub/InsightRepo/.venv/Scripts/uvicorn.exe main:app --reload --port 8001
```

### Frontend Issues

**Build Errors:**

```bash
cd frontend
npm install
npm run dev
```

**API Connection Failed:**

- Check that backend is running at http://localhost:8000
- Try accessing http://localhost:8000/health
- Check browser console for CORS errors

### Performance Issues

**Slow Indexing:**

- Normal for large repositories
- Check CPU/memory usage
- Try with a smaller repository first

**Slow Query Responses:**

- First query is slower (model loading)
- Subsequent queries are faster
- Consider reducing MAX_CHUNKS

## 📝 Development Commands

### Backend

```bash
# Start backend
cd backend
C:/Users/User/Documents/GitHub/InsightRepo/.venv/Scripts/uvicorn.exe main:app --reload

# Test with curl (PowerShell)
Invoke-WebRequest http://localhost:8000/health

# View API docs
# Open: http://localhost:8000/docs
```

### Frontend

```bash
# Start frontend
cd frontend
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## 🎯 Tips for Best Results

1. **Start Small**: Test with smaller repositories first to understand the system

2. **Be Specific**: Ask focused questions rather than broad queries
   - Good: "Where is user authentication validated?"
   - Avoid: "Tell me everything about this code"

3. **Follow the Citations**: Always check the cited code to verify answers

4. **Iterate Questions**: Start broad, then drill down into specific files/functions

5. **Use Context**: Reference previous answers in follow-up questions

## 🌟 Next Steps

### Enhance Your System

1. **Add More Models**

   ```bash
   ollama pull mistral:7b
   ollama pull codellama:7b
   ```

2. **Increase Context Window**
   - Adjust CHUNK_SIZE and MAX_CHUNKS
   - Experiment with different values

3. **Add Custom Filters**
   - Modify `repo_ingestion.py` to filter specific file types
   - Add language-specific chunking strategies

4. **Deploy to Cloud**
   - Containerize with Docker
   - Deploy backend to Railway, Render, or AWS
   - Deploy frontend to Vercel or Netlify

### Share Your System

- Demo it to colleagues
- Use it for onboarding new developers
- Create documentation from Q&A sessions
- Build a knowledge base of common questions

## 🎓 Learning Resources

**RAG Systems:**

- LangChain Documentation: https://python.langchain.com/docs
- FAISS Guide: https://github.com/facebookresearch/faiss/wiki

**Ollama:**

- Ollama Documentation: https://ollama.ai/docs
- Model Library: https://ollama.ai/library

**FastAPI:**

- FastAPI Tutorial: https://fastapi.tiangolo.com/tutorial
- Async Programming: https://fastapi.tiangolo.com/async

---

## 🎊 Congratulations!

You now have a fully functional RAG-powered code repository analysis system!

**What makes this special:**

- ✅ Fully local inference (privacy-first)
- ✅ Source code citations (explainable AI)
- ✅ Language-aware chunking (smart parsing)
- ✅ Hybrid deployment (local + cloud)
- ✅ Full-stack implementation (learning showcase)

**Start asking questions and exploring codebases!** 🚀
