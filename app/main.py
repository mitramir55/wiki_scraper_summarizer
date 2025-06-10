from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, HttpUrl
from sqlalchemy.orm import Session
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

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
    For example:
    - Input: "tell me about ai" -> Output: "artificial intelligence"
    - Input: "what is machine learning" -> Output: "machine learning"
    - Input: "can you explain quantum computing" -> Output: "quantum computing"
    Return ONLY the topic, nothing else."""),
    ("user", "{query}")
])

def get_wikipedia_url(topic: str) -> str:
    """Convert a topic to a Wikipedia URL."""
    # Clean and format the topic
    topic = topic.strip().lower()
    # Replace spaces with underscores
    topic = topic.replace(" ", "_")
    # Create Wikipedia URL
    return f"https://en.wikipedia.org/wiki/{topic}"

async def extract_topic_from_query(query: str) -> str:
    """Extract Wikipedia topic from user query using LLM."""
    chain = TOPIC_EXTRACTION_PROMPT | llm
    result = await chain.ainvoke({"query": query})
    topic = result.content.strip()
    
    if not topic:
        raise HTTPException(
            status_code=400,
            detail="No topic found in the query. Please provide a query containing a topic to search on Wikipedia."
        )
    
    return topic

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
    job_id: int
    status: str
    message: str

@app.post("/api/scrape", response_model=ScrapeResponse)
async def create_scraping_job(
    request: ScrapeRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    try:
        # Extract topic from query
        topic = await extract_topic_from_query(request.query)
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

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: int, db: Session = Depends(get_db)):
    job = db.query(ScrapingJob).filter(ScrapingJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
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
            print(f"Scraping result: {result}")  # Debug log

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
                print(f"Updated job status to completed with results: {job.results}")  # Debug log
            else:
                error_msg = result.get('error', 'Unknown error occurred') if result else 'No content found'
                print(f"Scraping failed: {error_msg}")  # Debug log
                job.status = "failed"
                job.error = error_msg
                db.commit()
                raise Exception(error_msg)

    except Exception as e:
        print(f"Error in scraping job: {str(e)}")  # Debug log
        job.status = "failed"
        job.error = str(e)
        db.commit()
        raise 