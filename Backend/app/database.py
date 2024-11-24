from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_HOST = os.getenv("DB_HOST")

# MySQL Database URL - Correctly formatted
SQLALCHEMY_DATABASE_URL = f"mysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

# Create the SQLAlchemy engine with pool settings to avoid idle disconnects
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_recycle=280,       # Recycle connections older than 280 seconds
    pool_pre_ping=True      # Check if the connection is alive before using it
)

# Create a session for database operations
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Dependency for getting the DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
