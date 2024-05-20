from fastapi import APIRouter

# from api.api_v1.endpoints.asr import asr
from api.api_v1 import end2end


api_router = APIRouter()

api_router.include_router(end2end.router, tags=["end2end"], prefix="/v1/end2end")
