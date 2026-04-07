from typing import Annotated, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select

from app.database import get_session
from app.models import Item, ItemCreate, ItemRead, ItemUpdate

router = APIRouter(prefix="/items", tags=["items"])


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with get_session() as session:
        yield session


@router.get("/", response_model=list[ItemRead])
async def list_items(
    session: Annotated[AsyncSession, Depends(get_db_session)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
) -> list[Item]:
    result = await session.execute(select(Item).offset(skip).limit(limit))
    items = result.scalars().all()
    return list(items)


@router.get("/{item_id}", response_model=ItemRead)
async def get_item(
    item_id: Annotated[int, Path(gt=0)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> Item:
    result = await session.execute(select(Item).where(Item.id == item_id))
    item = result.scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found")
    return item


@router.post("/", response_model=ItemRead, status_code=status.HTTP_201_CREATED)
async def create_item(
    item_data: ItemCreate,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> Item:
    item = Item(**item_data.model_dump())
    session.add(item)
    await session.flush()
    await session.refresh(item)
    return item


@router.patch("/{item_id}", response_model=ItemRead)
async def update_item(
    item_id: Annotated[int, Path(gt=0)],
    item_data: ItemUpdate,
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> Item:
    result = await session.execute(select(Item).where(Item.id == item_id))
    item = result.scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found")

    update_data = item_data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(item, key, value)

    await session.flush()
    await session.refresh(item)
    return item


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(
    item_id: Annotated[int, Path(gt=0)],
    session: Annotated[AsyncSession, Depends(get_db_session)],
) -> None:
    result = await session.execute(select(Item).where(Item.id == item_id))
    item = result.scalar_one_or_none()
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found")
    await session.delete(item)
