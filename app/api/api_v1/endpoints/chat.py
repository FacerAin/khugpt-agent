from fastapi import APIRouter

router = APIRouter()


@router.post("/completion")
def make_chat():
    return "api/v1/chat/completion"
