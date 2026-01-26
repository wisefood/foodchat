from fastapi import FastAPI
from src.routers import foodchat_router
import os

app = FastAPI()

# Include the routers
app.include_router(foodchat_router.router)

@app.get("/")
def root():
    return {"message": "Welcome to WiseFood API"}

if __name__ == "__main__":
    from user_interface import UI, launch_interface
    from foodchat_init import initialize_foodchat
    
    foodchat, config = initialize_foodchat()
    # rag_chain = foodchat.create_chain(args, ask_user_fn = None)
    # run_interactive_chat(foodchat, rag_chain, args)
    ui = UI(foodchat, config)
    launch_interface(ui, config)