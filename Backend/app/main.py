from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import user_routes, auth_routes, vessel_routes
from .database import engine, Base
from .utils.logging_config import setup_logging

# Create all tables in the database
Base.metadata.create_all(bind=engine)

# Configure logging (includes memory logging)
setup_logging()

app = FastAPI()

# CORS configuration
origins = [
    "http://localhost:3000",  # Frontend URL
    "http://127.0.0.1:3000",  # You can add other allowed origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the routes
app.include_router(auth_routes.router)
app.include_router(user_routes.router)
app.include_router(vessel_routes.router)

# Run the app using: uvicorn app.main:app --reload
