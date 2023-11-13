from fastapi import APIRouter

from app.agent import ChatAgent
from app.agent.retriever import PineconeRetriever
from app.schemas.chat import ResponseAnswer, ReuqestQuery

router = APIRouter()

chat_agent = ChatAgent()


@router.post("/completion", response_model=ResponseAnswer)
def make_chat(req: ReuqestQuery):
    answer = chat_agent.run(req.query)
    return {"answer": answer}


@router.get("/similarity-search")
def get_similarity_search(query: str):
    retreiver = PineconeRetriever(index_name="khugpt")
    return retreiver.get_relevant_doc_string(query)
