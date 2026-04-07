from datetime import datetime

from sqlmodel import SQLModel, Field


class Item(SQLModel, table=True):
    model_config = {"table": True}

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=500)
    price: float = Field(gt=0)
    quantity: int = Field(ge=0, default=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ItemCreate(SQLModel):
    name: str = Field(min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=500)
    price: float = Field(gt=0)
    quantity: int = Field(ge=0, default=0)


class ItemRead(SQLModel):
    id: int
    name: str
    description: str | None
    price: float
    quantity: int
    created_at: datetime
    updated_at: datetime


class ItemUpdate(SQLModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=500)
    price: float | None = Field(default=None, gt=0)
    quantity: int | None = Field(default=None, ge=0)
