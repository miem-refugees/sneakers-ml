from fastapi import APIRouter
from fastapi.responses import Response

router = APIRouter(tags=["common"])


@router.get("/ping")
async def ping():
    return Response(content="OK⛑", status_code=200)


@router.get("/")
async def root():
    return Response(content="Welcome to the club, buddy", status_code=200)
