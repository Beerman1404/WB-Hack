from pydantic import BaseModel
from typing import Literal


class InputData(BaseModel):
    user_id: int
    created_date: str
    nm_id: int
    total_ordered: int
    payment_type: str
    is_paid: bool
    count_items: int
    unique_items: int
    avg_unique_purchase: float
    is_courier: bool
    nm_age: int
    Distance: float
    days_after_registration: int
    number_of_orders: int
    number_of_ordered_items: int
    mean_number_of_ordered_items: float
    min_number_of_ordered_items: int
    max_number_of_ordered_items: int
    mean_percent_of_ordered_items: float
    service: str


class PredictionResult(BaseModel):
    label: Literal[0, 1]
    probability: float
