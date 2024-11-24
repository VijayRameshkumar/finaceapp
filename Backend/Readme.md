- Database
1. Log in to MySQL as the root user:
    sudo mysql -u root -p

2. Create database using database.sql

- Setup backend
1.  python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt
2. Updating the .env
3. Start redis server
4. Run the applications
    uvicorn app.main:app --reload
