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

# Initialize LLM for URL extraction
llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo"
)

URL_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a URL extraction assistant. Your task is to extract a single URL from the user's query.
    If there are multiple URLs, extract the most relevant one.
    If there is no URL, return 'NO_URL_FOUND'.
    Return ONLY the URL or 'NO_URL_FOUND', nothing else."""),
    ("user", "{query}")
])

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

async def extract_url_from_query(query: str) -> str:
    """Extract URL from user query using LLM."""
    chain = URL_EXTRACTION_PROMPT | llm
    result = await chain.ainvoke({"query": query})
    url = result.content.strip()
    
    if url == "NO_URL_FOUND":
        raise HTTPException(
            status_code=400,
            detail="No URL found in the query. Please provide a query containing a URL."
        )
    
    # Validate the extracted URL
    try:
        return str(HttpUrl(url))
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid URL extracted: {url}. Please provide a query with a valid URL."
        )

@app.post("/api/scrape", response_model=ScrapeResponse)
async def create_scraping_job(
    request: ScrapeRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    try:
        # Extract URL from query
        url = await extract_url_from_query(request.query)
        
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
    query = db.query(ScrapedContent)
    if url:
        query = query.filter(ScrapedContent.url == url)
    content = query.order_by(ScrapedContent.created_at.desc()).first()
    return content

async def process_scraping_job(job_id: int, url: str):
    db = next(get_db())
    try:
        # Update job status
        job = db.query(ScrapingJob).filter(ScrapingJob.id == job_id).first()
        job.status = "running"
        db.commit()

        # Initialize scraper
        async with AsyncWebScraper() as scraper:
            result = await scraper.scrape_url(url)

            if result and 'error' not in result:
                # Save to database
                content = ScrapedContent(
                    url=result.get('url', url),
                    title=result.get('title', ''),
                    text=result.get('text', ''),
                    summary=result.get('summary', ''),
                    extra_metadata={'timestamp': result.get('timestamp')}
                )
                db.add(content)

                # Update job status
                job.status = "completed"
                job.results = result
                db.commit()
            else:
                job.status = "failed"
                job.error = result.get('error', 'Unknown error occurred') if result else 'No content found'
                db.commit()
                raise Exception(job.error)

    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        db.commit()
        raise 