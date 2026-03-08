from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: str | None = None
    similarity_score: float | None = None
    result: str
    dominant_cluster: int