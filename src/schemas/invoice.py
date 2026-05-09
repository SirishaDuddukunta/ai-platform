from pydantic import BaseModel, Field, field_validator
from typing import List

class LineItem(BaseModel):
    item: str
    price: float
    qty: int # Removed gt=0 to allow the data to enter our logic

    @field_validator('qty')
    @classmethod
    def validate_qty(cls, v):
        if v <= 0:
            # This error message is sent back to the LLM to trigger a retry
            raise ValueError("Quantity must be at least 1. Please correct 0 to 1.")
        return v

class Invoice(BaseModel):
    vendor: str
    items: List[LineItem]
    total: float
    reasoning: str