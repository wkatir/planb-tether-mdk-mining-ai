from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, kpi, telemetry, control


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="MDK Mining AI",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(telemetry.router, prefix="/telemetry", tags=["telemetry"])
app.include_router(kpi.router, prefix="/kpi", tags=["kpi"])
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(control.router, prefix="/control", tags=["control"])


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "MDK Mining AI API"}


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy"}
