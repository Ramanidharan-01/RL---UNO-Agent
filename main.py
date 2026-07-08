"""
app/main.py
───────────
FastAPI application entry-point.

Phase 1: startup/shutdown lifecycle + a /health endpoint so the stack can
         be verified end-to-end before the real routes are added in Phase 2.

The ``game_wrapper`` application state attribute is the single ``UNOGameWrapper``
instance shared across all WebSocket and REST handlers.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.redis_client import close_redis, redis_ping
from app.game.agent import get_agent
from app.game.wrapper import UNOGameWrapper


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan – runs startup and shutdown logic
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    On startup:
      1. Load the JAX checkpoint and JIT-warm the agent (can take a few seconds
         on cold start – done once, not per-request).
      2. Attach the shared UNOGameWrapper to app.state.

    On shutdown:
      1. Gracefully close the Redis connection pool.
    """
    settings = get_settings()

    # This blocks the event loop briefly while JAX compiles – acceptable at
    # startup.  In production you could run it in a ProcessPoolExecutor.
    agent   = get_agent()                   # loads checkpoint, warms JIT
    wrapper = UNOGameWrapper(agent)
    app.state.game_wrapper = wrapper

    yield

    # ── Cleanup ───────────────────────────────────────────────────────────
    await close_redis()


# ─────────────────────────────────────────────────────────────────────────────
# Application factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        debug=settings.debug,
        lifespan=lifespan,
    )

    # CORS – allow the Next.js / Vite dev server during development.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers (added in Phase 2) ────────────────────────────────────────
    # from app.api.game   import router as game_router
    # from app.api.auth   import router as auth_router
    # from app.api.sim    import router as sim_router
    # application.include_router(game_router, prefix="/api")
    # application.include_router(auth_router, prefix="/api")
    # application.include_router(sim_router,  prefix="/api")

    # ── Health endpoint (available now) ───────────────────────────────────
    @application.get("/health", tags=["ops"])
    async def health():
        redis_ok = await redis_ping()
        return {
            "status": "ok" if redis_ok else "degraded",
            "redis": redis_ok,
        }

    return application


app = create_app()
