from fastapi import FastAPI
from src.routers import foodscholar, foodchat  # Import the new router

app = FastAPI()

# Include the routers
app.include_router(foodscholar.router)
app.include_router(foodchat.router)  # Add this line

@app.get("/")
def root():
    return {"message": "Welcome to WiseFood API"}