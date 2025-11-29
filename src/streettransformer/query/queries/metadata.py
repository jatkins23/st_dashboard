from pydantic import BaseModel, Field
from datetime import datetime

class QueryMetadata(BaseModel):
    query_type: str
    execution_time_ms: float
    search_method: str
    timestamp: datetime = Field(default_factory=datetime.now)