"""
greet_only_bot.py
Minimal TwitchIO 3.x bot with only the !greet command.

Install:
    pip install -U "twitchio[starlette]" asqlite python-dotenv aiohttp

Required Twitch app redirect URL:
    http://localhost:4343/oauth/callback

Authorize bot account, logged in as kneadbot:
    http://localhost:4343/oauth?scopes=user:read:chat%20user:write:chat%20user:bot&force_verify=true

Authorize broadcaster account, logged in as fromcollin:
    http://localhost:4343/oauth?scopes=channel:bot&force_verify=true

Expected local file:
    vibe_voice_client.py

Expected client API:
    TTSClient().synthesize_greet(twitch_user_id)
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import asqlite
import twitchio
from dotenv import load_dotenv
from twitchio import eventsub
from twitchio.ext import commands

from vibe_voice_client import TTSClient

load_dotenv()

LOGGER = logging.getLogger("greetbot")

CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")
CLIENT_SECRET = os.getenv("TWITCH_CLIENT_SECRET")

OWNER_ID = os.getenv("TWITCH_OWNER_ID", "NUMBER")  # owner
BOT_ID = os.getenv("TWITCH_BOT_ID", "NUMBER")     # bot

DATABASE_NAME = os.getenv("TWITCH_TOKEN_DB", "tokens.db")
PREFIX = os.getenv("TWITCH_PREFIX", "!")

VOICE_RECORD_URL = os.getenv(
    "VOICE_RECORD_URL",
    "https://your-web-site.com/voiceclone",
)


@dataclass
class RuntimeState:
    # Prevents one viewer from spamming !greet repeatedly in a single bot session.
    greeters: set[str] = field(default_factory=set)


STATE = RuntimeState()


def chatter_id(ctx: commands.Context) -> str:
    chatter = getattr(ctx, "chatter", None)
    return str(getattr(chatter, "id", "") or "")


def badge_names(ctx: commands.Context) -> list[str]:
    chatter = getattr(ctx, "chatter", None)
    return [str(getattr(badge, "set_id", "")) for badge in getattr(chatter, "badges", [])]


def has_badge(ctx: commands.Context, badge_name: str) -> bool:
    return badge_name in badge_names(ctx)


def is_subscriber(ctx: commands.Context) -> bool:
    # founder covers founders; broadcaster/OWNER_ID lets you test the command yourself.
    return (
        has_badge(ctx, "subscriber")
        or has_badge(ctx, "founder")
        or has_badge(ctx, "broadcaster")
        or chatter_id(ctx) == OWNER_ID
    )


async def require_subscriber(ctx: commands.Context) -> bool:
    if is_subscriber(ctx):
        return True
    await ctx.reply("subscriber-only command.")
    return False


def build_tts() -> TTSClient | None:
    try:
        return TTSClient()
    except Exception as exc:
        LOGGER.exception("Failed to initialize VibeVoice TTSClient: %s", exc)
        return None


TTS = build_tts()


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def parse_greet_result(result: Any) -> tuple[bool, str | None]:
    """
    Supports common client return shapes:
      - (ok, err)
      - True / False
      - {"ok": bool, "error": str}
      - None means success for clients that raise exceptions on failure.
    """
    if result is None:
        return True, None

    if isinstance(result, tuple):
        ok = bool(result[0]) if len(result) >= 1 else False
        err = str(result[1]) if len(result) >= 2 and result[1] else None
        return ok, err

    if isinstance(result, dict):
        ok = bool(result.get("ok", result.get("success", False)))
        err = result.get("error") or result.get("err") or result.get("message")
        return ok, str(err) if err else None

    if isinstance(result, bool):
        return result, None if result else "unknown error"

    return True, None


class GreetBot(commands.AutoBot):
    def __init__(self, *, token_database: asqlite.Pool, subs: list[eventsub.SubscriptionPayload]) -> None:
        self.token_database = token_database
        super().__init__(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            bot_id=BOT_ID,
            owner_id=OWNER_ID,
            prefix=PREFIX,
            subscriptions=subs,
            force_subscribe=True,
        )

    async def setup_hook(self) -> None:
        await self.add_component(GreetCommands(self))

    async def event_ready(self) -> None:
        LOGGER.info("Bot is ready. bot_id=%s owner_id=%s", self.bot_id, self.owner_id)

    async def event_oauth_authorized(self, payload: twitchio.authentication.UserTokenPayload) -> None:
        await self.add_token(payload.access_token, payload.refresh_token)

        # Bot token does not need a broadcaster chat subscription.
        if not payload.user_id or payload.user_id == self.bot_id:
            return

        subs = [
            eventsub.ChatMessageSubscription(
                broadcaster_user_id=payload.user_id,
                user_id=self.bot_id,
            )
        ]

        resp = await self.multi_subscribe(subs)
        if resp.errors:
            LOGGER.warning("Failed to subscribe after OAuth: %r", resp.errors)

    async def add_token(self, token: str, refresh: str) -> twitchio.authentication.ValidateTokenPayload:
        resp = await super().add_token(token, refresh)

        query = """
        INSERT INTO tokens (user_id, token, refresh)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id)
        DO UPDATE SET token = excluded.token, refresh = excluded.refresh;
        """

        async with self.token_database.acquire() as connection:
            await connection.execute(query, (resp.user_id, token, refresh))

        LOGGER.info("Saved token for user_id=%s", resp.user_id)
        return resp


class GreetCommands(commands.Component):
    def __init__(self, bot: GreetBot) -> None:
        self.bot = bot

    @commands.command()
    async def greet(self, ctx: commands.Context) -> None:
        if not await require_subscriber(ctx):
            return

        if TTS is None:
            await ctx.reply("voice clone client is offline.")
            return

        uid = chatter_id(ctx)
        if not uid:
            await ctx.reply("could not read your Twitch user id.")
            return

        if uid in STATE.greeters:
            await ctx.reply("you already used !greet this stream.")
            return

        try:
            result = await maybe_await(TTS.synthesize_greet(uid))
            ok, err = parse_greet_result(result)
        except Exception as exc:
            LOGGER.exception("!greet failed")
            await ctx.reply(f"greet failed: {type(exc).__name__}")
            return

        if not ok:
            err = str(err or "unknown error")
            if "Voice file not found" in err:
                await ctx.reply(f"I don’t have your voice yet. Record here: {VOICE_RECORD_URL}")
            else:
                await ctx.reply(f"greet failed: {err}")
            return

        STATE.greeters.add(uid)


async def setup_database(db: asqlite.Pool) -> tuple[list[tuple[str, str]], list[eventsub.SubscriptionPayload]]:
    async with db.acquire() as connection:
        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tokens (
                user_id TEXT PRIMARY KEY,
                token TEXT NOT NULL,
                refresh TEXT NOT NULL
            );
            """
        )
        rows = await connection.fetchall("SELECT user_id, token, refresh FROM tokens")

    tokens: list[tuple[str, str]] = []
    subs: list[eventsub.SubscriptionPayload] = []

    for row in rows:
        tokens.append((row["token"], row["refresh"]))

        # Only broadcaster/user tokens get chat subscriptions.
        if str(row["user_id"]) == BOT_ID:
            continue

        subs.append(
            eventsub.ChatMessageSubscription(
                broadcaster_user_id=str(row["user_id"]),
                user_id=BOT_ID,
            )
        )

    return tokens, subs


def validate_config() -> None:
    missing = []
    if not CLIENT_ID:
        missing.append("TWITCH_CLIENT_ID")
    if not CLIENT_SECRET:
        missing.append("TWITCH_CLIENT_SECRET")
    if not OWNER_ID:
        missing.append("TWITCH_OWNER_ID")
    if not BOT_ID:
        missing.append("TWITCH_BOT_ID")
    if missing:
        raise RuntimeError(f"Missing config: set {', '.join(missing)} in .env")


async def runner() -> None:
    validate_config()

    async with asqlite.create_pool(DATABASE_NAME) as token_db:
        tokens, subs = await setup_database(token_db)
        async with GreetBot(token_database=token_db, subs=subs) as bot:
            for token, refresh in tokens:
                await bot.add_token(token, refresh)
            await bot.start(load_tokens=False)


def main() -> None:
    twitchio.utils.setup_logging(level=logging.INFO)
    try:
        asyncio.run(runner())
    except KeyboardInterrupt:
        LOGGER.warning("Shutting down.")


if __name__ == "__main__":
    main()
