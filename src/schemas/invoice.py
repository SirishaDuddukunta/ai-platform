from pydantic import BaseModel, Field
from typing import List

class LineItem(BaseModel):
    item: str
    qty: int = Field(gt=0)
    price: float

class Invoice(BaseModel):
    vendor: str
    items: List[LineItem]
    total: float
    # ADDITION: The Reasoning Log
    reasoning: str = Field(
        description="A step-by-step breakdown of how you calculated the total, "
                    "handled corrections, and identified the final vendor."
    )
    is_urgent: bool = False
    