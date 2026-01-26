from fastapi import FastAPI
from src.routers import foodchat_router  # Import the new router

app = FastAPI()

# Include the routers
app.include_router(foodchat_router.router)  # Add this line

@app.get("/")
def root():
    return {"message": "Welcome to WiseFood API"}