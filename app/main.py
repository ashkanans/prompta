from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings, apply_environment
from app.core.logging import configure_logging
from app.core.errors import register_exception_handlers
from app.core.middleware import RequestContextMiddleware
from app.api import health, batch, completions
from fastapi.routing import APIRoute

apply_environment(settings)
configure_logging()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

app.add_middleware(
CORSMiddleware,
allow_origins=settings.CORS_ORIGINS,
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)

app.add_middleware(RequestContextMiddleware)

register_exception_handlers(app)

app.include_router(health.router, prefix="")
app.include_router(batch.router, prefix="/batch", tags=["batch"])
app.include_router(completions.router, prefix="")

@app.get("/")
async def root():
    return {"name": settings.APP_NAME, "version": settings.APP_VERSION}

@app.get("/routes")
async def list_routes():
    route_list = []
    for route in app.routes:
        if isinstance(route, APIRoute):  # only include API routes, skip static/middleware
            route_list.append({
                "path": route.path,
                "name": route.name,
                "methods": list(route.methods)
            })
    return {"routes": route_list}