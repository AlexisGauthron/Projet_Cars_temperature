# -*- coding: utf-8 -*-
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import frame, vlm
from app.config import Settings

app = FastAPI(
    title=Settings.API_TITLE,
    description=Settings.API_DESCRIPTION,
    version=Settings.API_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=Settings.CORS_ORIGINS,
    allow_credentials=Settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=Settings.CORS_ALLOW_METHODS,
    allow_headers=Settings.CORS_ALLOW_HEADERS,
)

app.include_router(frame.router, prefix="/api")
app.include_router(vlm.router, prefix="/api")


@app.get("/")
async def root():
    return {
        "message": Settings.API_TITLE,
        "version": Settings.API_VERSION,
    }
