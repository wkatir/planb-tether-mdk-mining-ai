# FastAPI Enterprise Template

A production-ready FastAPI project template with Docker, PostgreSQL, and modern Python tooling.

## Tech Stack

- **FastAPI** - Modern Python web framework
- **SQLModel** - SQL ORM with Pydantic integration
- **PostgreSQL** - Relational database
- **Alembic** - Database migrations
- **Docker** - Containerization
- **uv** - Python package manager

## Quick Start

```bash
# Start with Docker Compose
docker-compose up --build

# Run migrations
uv run alembic upgrade head

# Access API docs
open http://localhost:8000/docs
```

## Features

- Async SQLModel with PostgreSQL
- Alembic migrations
- Docker multi-stage build with uv
- Health check endpoint
- CRUD API structure
- Environment-based configuration

## TODO

- [ ] Add authentication (JWT)
- [ ] Add API documentation
- [ ] Add tests
- [ ] Add CI/CD pipeline

---

*Template created by [@wkatir](https://github.com/wkatir) - More details coming soon*
