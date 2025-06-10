from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, HttpUrl
from sqlalchemy.orm import Session
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from datetime import datetime

from app.core.scraper import AsyncWebScraper
from app.db.models import ScrapedContent, ScrapingJob, Base
from app.db.database import get_db, engine

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Web Scraping with LLMs and LangChain")

# Create static directory if it doesn't exist
os.makedirs("app/static", exist_ok=True)
os.makedirs("app/templates", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="app/templates")

# Initialize LLM for topic extraction
llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo"
)

TOPIC_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a topic extraction assistant. Your task is to extract a single Wikipedia topic from the user's query.
    If the query is ambiguous (e.g., multiple possible topics or unclear reference), return 'AMBIGUOUS_QUERY' followed by a clarification question.
    For example:
    - Input: "tell me about ai" -> Output: "artificial intelligence"
    - Input: "what is machine learning" -> Output: "machine learning"
    - Input: "can you explain quantum computing" -> Output: "quantum computing"
    - Input: "tell me about Meryl" -> Output: "AMBIGUOUS_QUERY: Which Meryl are you interested in? Please specify (e.g., Meryl Streep, Meryl Davis, etc.)"
    - Input: "tell me about John" -> Output: "AMBIGUOUS_QUERY: There are many notable people named John. Could you please specify which John you're interested in?"
    - Input: "tell me about robert pattinson" -> Output: "Robert Pattinson"
    - Input: "who is tom cruise" -> Output: "Tom Cruise"
    Return ONLY the topic or the AMBIGUOUS_QUERY message, nothing else."""),
    ("user", "{query}")
])

def get_wikipedia_url(topic: str) -> str:
    """Convert a topic to a Wikipedia URL with proper capitalization."""
    # Clean and format the topic
    topic = topic.strip()
    
    # Split into words and capitalize each word
    words = topic.split()
    capitalized_words = [word.capitalize() for word in words]
    
    # Join with underscores
    formatted_topic = '_'.join(capitalized_words)
    
    # Create Wikipedia URL
    return f"https://en.wikipedia.org/wiki/{formatted_topic}"

async def extract_topic_from_query(query: str) -> tuple[str, bool, str | None]:
    """Extract Wikipedia topic from user query using LLM.
    Returns a tuple of (topic, is_ambiguous, clarification_question)"""
    chain = TOPIC_EXTRACTION_PROMPT | llm
    result = await chain.ainvoke({"query": query})
    response = result.content.strip()
    
    if response.startswith("AMBIGUOUS_QUERY:"):
        clarification = response.replace("AMBIGUOUS_QUERY:", "").strip()
        return "", True, clarification
    
    if not response:
        raise HTTPException(
            status_code=400,
            detail="No topic found in the query. Please provide a query containing a topic to search on Wikipedia."
        )
    
    return response, False, None

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("app/templates/index.html", "r") as f:
        return f.read()

@app.get("/favicon.png")
async def favicon():
    return FileResponse("app/static/favicon.png")

class ScrapeRequest(BaseModel):
    query: str

class ScrapeResponse(BaseModel):
    job_id: int | None = None
    status: str
    message: str
    needs_clarification: bool = False
    clarification_question: str | None = None
    error_type: str | None = None

@app.post("/api/scrape", response_model=ScrapeResponse)
async def create_scraping_job(
    request: ScrapeRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    try:
        # Extract topic from query
        topic, is_ambiguous, clarification = await extract_topic_from_query(request.query)
        
        if is_ambiguous:
            return ScrapeResponse(
                job_id=None,
                status="needs_clarification",
                message="Query needs clarification",
                needs_clarification=True,
                clarification_question=clarification
            )
            
        # Convert topic to Wikipedia URL
        url = get_wikipedia_url(topic)
        
        # Check if URL already exists in database
        existing_content = db.query(ScrapedContent).filter(
            ScrapedContent.url == url
        ).order_by(ScrapedContent.created_at.desc()).first()

        if existing_content:
            # Create a job to represent the existing content
            job = ScrapingJob(
                status="completed",
                url=url,
                results={
                    'url': existing_content.url,
                    'title': existing_content.title,
                    'text': existing_content.text,
                    'summary': existing_content.summary,
                    'timestamp': existing_content.extra_metadata.get('timestamp')
                }
            )
            db.add(job)
            db.commit()
            db.refresh(job)
            
            return ScrapeResponse(
                job_id=job.id,
                status="completed",
                message="Content retrieved from database"
            )

        # If URL doesn't exist, create new job
        job = ScrapingJob(
            status="pending",
            url=url,
            results={}
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        # Start scraping in background
        background_tasks.add_task(process_scraping_job, job.id, url)
        
        return ScrapeResponse(
            job_id=job.id,
            status="pending",
            message="Scraping job created successfully"
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up database connection
        db.close()

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: int, db: Session = Depends(get_db)):
    job = db.query(ScrapingJob).filter(ScrapingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # If the job failed with a not found error, return a clarification response
    if job.status == "failed" and job.error_type == "not_found":
        return {
            "status": "needs_clarification",
            "message": job.error,
            "needs_clarification": True,
            "clarification_question": f"The article was not found. {job.error}",
            "error_type": "not_found"
        }
    
    return job

@app.get("/api/content")
async def get_scraped_content(
    url: str = None,
    db: Session = Depends(get_db)
):
    if not url:
        raise HTTPException(status_code=400, detail="URL parameter is required")
    
    print(f"Searching for content with URL: {url}")  # Debug log
    
    content = db.query(ScrapedContent).filter(
        ScrapedContent.url == url
    ).order_by(ScrapedContent.created_at.desc()).first()
    
    if not content:
        print(f"No content found for URL: {url}")  # Debug log
        raise HTTPException(status_code=404, detail=f"No content found for URL: {url}")
    
    print(f"Found content with title: {content.title}")  # Debug log
    return content

async def process_scraping_job(job_id: int, url: str):
    db = next(get_db())
    try:
        # Update job status
        job = db.query(ScrapingJob).filter(ScrapingJob.id == job_id).first()
        job.status = "running"
        db.commit()

        print(f"Starting scraping job for URL: {url}")  # Debug log

        # Initialize scraper
        async with AsyncWebScraper() as scraper:
            result = await scraper.scrape_url(url)
            print(f"Scraping completed for URL: {url}")  # Debug log

            if result and 'error' not in result:
                # Save to database
                content = ScrapedContent(
                    url=url,  # Use the Wikipedia URL
                    title=result.get('title', ''),
                    text=result.get('text', ''),
                    summary=result.get('summary', ''),
                    extra_metadata={
                        'timestamp': result.get('timestamp'),
                        'source': 'wikipedia'
                    }
                )
                db.add(content)
                db.commit()
                print(f"Saved content to database with title: {content.title}")  # Debug log

                # Update job status with complete results
                job.status = "completed"
                job.results = {
                    'url': url,
                    'title': content.title,
                    'text': content.text,
                    'summary': content.summary,
                    'timestamp': content.extra_metadata.get('timestamp')
                }
                db.commit()
                print(f"Job {job_id} completed successfully")  # Debug log
            else:
                error_msg = result.get('error', 'Unknown error occurred') if result else 'No content found'
                error_type = result.get('error_type', 'general_error')
                print(f"Scraping failed: {error_msg}")  # Debug log
                job.status = "failed"
                job.error = error_msg
                job.error_type = error_type
                db.commit()
                raise Exception(error_msg)

    except Exception as e:
        print(f"Error in scraping job: {str(e)}")  # Debug log
        job.status = "failed"
        job.error = str(e)
        job.error_type = 'general_error'
        db.commit()
        raise
    finally:
        # Clean up database connection
        db.close()
        print(f"Scraping job {job_id} completed and cleaned up")  # Debug log

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint that verifies database connectivity and returns application status."""
    try:
        # Check database connectivity
        db.execute("SELECT 1")
        
        # Get basic stats
        total_content = db.query(ScrapedContent).count()
        total_jobs = db.query(ScrapingJob).count()
        pending_jobs = db.query(ScrapingJob).filter(ScrapingJob.status == "pending").count()
        running_jobs = db.query(ScrapingJob).filter(ScrapingJob.status == "running").count()
        
        return JSONResponse({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": {
                "connected": True,
                "stats": {
                    "total_content": total_content,
                    "total_jobs": total_jobs,
                    "pending_jobs": pending_jobs,
                    "running_jobs": running_jobs
                }
            }
        })
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
        ) 