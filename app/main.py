from fastapi import FastAPI
from demand.routers import router as demand_router

app = FastAPI(title="AzureStay AI Services")

app.include_router(demand_router)

@app.get("/health")
def health_check():
    return {"status": "healthy"}