from pydantic import BaseModel


class ReuqestQuery(BaseModel):
    query: str


class ResponseAnswer(BaseModel):
    answer: str
