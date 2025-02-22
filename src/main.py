import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.endpoints import router as api_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/")
def read_root():
    return {
        "message": (
            "Welcome to the High-Dimensional "
            "Image Processing Microservice"
        )
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
