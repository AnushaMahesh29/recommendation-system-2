from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from sentence_transformers import SentenceTransformer, util
import json
from typing import Optional, List, Dict, Any
import logging
import numpy as np
from pydantic import BaseModel, Field
from pydantic.json import pydantic_encoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Luxury Recommendation API",
    description="API for personalized luxury product recommendations using ML",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
# Product model with proper serialization
class Product(BaseModel):
    id: str
    name: str
    price: float
    image: str
    description: str
    category: str
    similarity_score: Optional[float] = Field(None, exclude=True)  # Don't include in response
    
    class Config:
        json_encoders = {
            np.ndarray: lambda v: v.tolist(),
        }

# Global variables
model = None
products: List[Product] = []

def numpy_to_list(array):
    """Convert numpy arrays to lists for JSON serialization"""
    if isinstance(array, np.ndarray):
        return array.tolist()
    return array

async def startup_event():
    """Initialize the application"""
    global model, products
    
    try:
        # Load ML model
        logger.info("Loading ML model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded successfully")
        
        # Load products
        logger.info("Loading products...")
        with open("products.json") as f:
            products_data = json.load(f)
            products.extend([Product(**p) for p in products_data["products"]])
        logger.info(f"Loaded {len(products)} products")
        
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise

app.add_event_handler("startup", startup_event)

@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "all-MiniLM-L6-v2",
        "products_loaded": len(products)
    }
@app.get("/recommend")
async def recommend(
    bio: str = Query(..., min_length=10, description="User description for recommendations"),
    category: Optional[str] = Query(None, description="Filter by product category"),
    name_weight: float = Query(0.3, description="Weight for product name similarity (0-1)")
):
    """Get personalized product recommendations considering both names and descriptions"""
    try:
        if not bio.strip():
            raise HTTPException(status_code=400, detail="Bio cannot be empty")
        if not 0 <= name_weight <= 1:
            raise HTTPException(status_code=400, detail="name_weight must be between 0 and 1")
            
        logger.info(f"Getting recommendations for bio: {bio[:50]}...")
        
        # Encode user bio once
        bio_vec = model.encode(bio)
        
        # Filter by category if specified
        filtered_products = products
        if category:
            category = category.lower()
            filtered_products = [p for p in products if p.category.lower() == category]
            if not filtered_products:
                raise HTTPException(
                    status_code=404,
                    detail=f"No products found in category: {category}"
                )
        
        # Calculate similarities and prepare response
        recommendations = []
        for product in filtered_products:
            # Encode both name and description separately
            name_vec = model.encode(product.name)
            desc_vec = model.encode(product.description)
            
            # Calculate weighted similarity
            name_sim = float(util.cos_sim(bio_vec, name_vec)[0][0])
            desc_sim = float(util.cos_sim(bio_vec, desc_vec)[0][0])
            combined_sim = (name_weight * name_sim) + ((1 - name_weight) * desc_sim)
            
            # Create response product
            rec_product = product.dict()
            rec_product.update({
                "similarity_score": combined_sim,
                "name_similarity": name_sim,  # Optional: include breakdown
                "desc_similarity": desc_sim   # Optional: include breakdown
            })
            recommendations.append(rec_product)
        
        # Sort by combined similarity and get top 3
        recommendations = sorted(
            recommendations,
            key=lambda x: x["similarity_score"],
            reverse=True
        )[:3]
        
        logger.info(f"Returning {len(recommendations)} recommendations")
        return recommendations
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate recommendations. Please try again later."
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )