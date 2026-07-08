"""
app/main.py
───────────
FastAPI application entry-point.

Startup:
  1. Load JAX checkpoint and JIT-warm the agent.
  2. Attach UNOGameWrapper to app.state.
  3. Initialise database tables (development only).
  4. Register all API routers.

Shutdown:
  1. Close Redis connection pool.
  2. Dispose database engine.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.redis_client import close_redis, redis_ping
from app.db.base import close_db, create_tables
from app.game.agent import get_agent
from app.game.wrapper import UNOGameWrapper
from app.services.multiplayer_service import MultiplayerService


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan – runs startup and shutdown logic
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    On startup:
      1. Load the JAX checkpoint and JIT-warm the agent.
      2. Attach shared services to app.state.
      3. Create database tables (auto-migrate in development).

    On shutdown:
      1. Close Redis pool.
      2. Dispose database engine.
    """
    settings = get_settings()

    # This blocks the event loop briefly while JAX compiles – acceptable at
    # startup.  In production you could run it in a ProcessPoolExecutor.
    agent   = get_agent()                   # loads checkpoint, warms JIT
    wrapper = UNOGameWrapper(agent)

    app.state.game_wrapper = wrapper
    app.state.multiplayer_service = MultiplayerService()

    # Create database tables (development convenience).
    # In production, use Alembic migrations instead.
    try:
        await create_tables()
    except Exception as e:
        # Don't crash on DB errors at startup – Redis-only mode still works
        import logging
        logging.getLogger("uvicorn").warning(f"DB table creation skipped: {e}")

    yield

    # ── Cleanup ───────────────────────────────────────────────────────────
    await close_redis()
    await close_db()


# ─────────────────────────────────────────────────────────────────────────────
# Application factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="Production-ready UNO AI platform with real-time gameplay, simulations, and multiplayer.",
        debug=settings.debug,
        lifespan=lifespan,
    )

    # CORS – allow configured origins.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────
    from app.api.auth import router as auth_router
    from app.api.game import router as game_router
    from app.api.simulation import router as sim_router
    from app.api.multiplayer import router as mp_router
    from app.api.stats import router as stats_router
    from app.api.replay import router as replay_router

    application.include_router(auth_router, prefix="/api")
    application.include_router(game_router, prefix="/api")
    application.include_router(sim_router, prefix="/api")
    application.include_router(mp_router, prefix="/api")
    application.include_router(stats_router, prefix="/api")
    application.include_router(replay_router, prefix="/api")

    # ── Health endpoint ───────────────────────────────────────────────────
    @application.get("/health", tags=["ops"])
    async def health():
        redis_ok = await redis_ping()
        return {
            "status": "ok" if redis_ok else "degraded",
            "redis": redis_ok,
            "version": "1.0.0",
        }

    return application


app = create_app()
