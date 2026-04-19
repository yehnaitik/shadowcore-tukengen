import os
import asyncio
import aiohttp
import discord
from discord import app_commands
from discord.ext import commands
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from groq import Groq
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

load_dotenv()

AI_INTEGRATIONS_OPENAI_API_KEY = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
AI_INTEGRATIONS_OPENAI_BASE_URL = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
AI_INTEGRATIONS_ANTHROPIC_API_KEY = os.environ.get("AI_INTEGRATIONS_ANTHROPIC_API_KEY")
AI_INTEGRATIONS_ANTHROPIC_BASE_URL = os.environ.get("AI_INTEGRATIONS_ANTHROPIC_BASE_URL")
AI_INTEGRATIONS_GEMINI_API_KEY = os.environ.get("AI_INTEGRATIONS_GEMINI_API_KEY")
AI_INTEGRATIONS_GEMINI_BASE_URL = os.environ.get("AI_INTEGRATIONS_GEMINI_BASE_URL")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

OPENAI_MODEL = "gpt-5"
ANTHROPIC_MODEL = "claude-sonnet-4-5"
GEMINI_MODEL = "gemini-2.5-pro-preview-03-25"
GROQ_MODEL = "llama-3.3-70b-versatile"

openai_client = OpenAI(api_key=AI_INTEGRATIONS_OPENAI_API_KEY, base_url=AI_INTEGRATIONS_OPENAI_BASE_URL)
anthropic_client = Anthropic(api_key=AI_INTEGRATIONS_ANTHROPIC_API_KEY, base_url=AI_INTEGRATIONS_ANTHROPIC_BASE_URL)
gemini_client = genai.Client(
    api_key=AI_INTEGRATIONS_GEMINI_API_KEY,
    http_options={'api_version': '', 'base_url': AI_INTEGRATIONS_GEMINI_BASE_URL}
)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# ── AI helpers ─────────────────────────────────────────────────────────────────

def is_rate_limit_error(exception: BaseException) -> bool:
    error_msg = str(exception)
    return (
        "429" in error_msg or "RATELIMIT_EXCEEDED" in error_msg
        or "quota" in error_msg.lower() or "rate limit" in error_msg.lower()
        or (hasattr(exception, "status_code") and getattr(exception, "status_code", None) == 429)
        or (hasattr(exception, "status") and getattr(exception, "status", None) == 429)
    )


def get_backup_response(prompt: str) -> str:
    if not groq_client:
        return "Backup brain not configured (no GROQ_API_KEY set)."
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a backup AI coding assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content or "Backup brain failed."
    except Exception as e:
        return f"Backup brain error: {str(e)}"


@retry(stop=stop_after_attempt(7), wait=wait_exponential(multiplier=1, min=2, max=128),
       retry=retry_if_exception(is_rate_limit_error), reraise=True)
def get_gpt5_response(prompt: str) -> str:
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert AI coding agent. Always provide high-quality, production-ready code snippets."},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=8192
    )
    return response.choices[0].message.content or "Error with GPT-5"


@retry(stop=stop_after_attempt(7), wait=wait_exponential(multiplier=1, min=2, max=128),
       retry=retry_if_exception(is_rate_limit_error), reraise=True)
def get_claude_response(prompt: str) -> str:
    message = anthropic_client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=8192,
        system="You are an expert at code logic and debugging. Focus on fixing errors and improving structural integrity.",
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text if message.content[0].type == "text" else "Error with Claude"


@retry(stop=stop_after_attempt(7), wait=wait_exponential(multiplier=1, min=2, max=128),
       retry=retry_if_exception(is_rate_limit_error), reraise=True)
def get_gemini_response(prompt: str) -> str:
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={"system_instruction": "You are a design and UI expert. Provide beautiful CSS, UI components, and design guidance."}
    )
    return response.text or "Error with Gemini"


async def handle_code_prompt(prompt: str) -> str:
    p_lower = prompt.lower()
    if any(k in p_lower for k in ["design", "ui", "css", "html", "style", "frontend"]):
        try:
            response = get_gemini_response(prompt)
            provider = "Gemini"
        except Exception as e:
            if "FREE_CLOUD_BUDGET_EXCEEDED" in str(e):
                response = get_backup_response(prompt)
                provider = "Llama-3 (Backup)"
            else:
                raise e
    elif any(k in p_lower for k in ["fix", "error", "bug", "logic", "reasoning", "refactor"]):
        try:
            response = get_claude_response(prompt)
            provider = "Claude"
        except Exception as e:
            if "FREE_CLOUD_BUDGET_EXCEEDED" in str(e):
                response = get_backup_response(prompt)
                provider = "Llama-3 (Backup)"
            else:
                raise e
    else:
        try:
            response = get_gpt5_response(prompt)
            provider = "GPT-5"
        except Exception as e:
            if "FREE_CLOUD_BUDGET_EXCEEDED" in str(e):
                response = get_backup_response(prompt)
                provider = "Llama-3 (Backup)"
            else:
                raise e
    return f"**[AI: {provider}]**\n{response}"


# ── Discord API helpers ────────────────────────────────────────────────────────

BASE = "https://discord.com/api/v10"


async def api_get(session: aiohttp.ClientSession, token: str, path: str):
    headers = {"Authorization": token, "Content-Type": "application/json"}
    async with session.get(f"{BASE}{path}", headers=headers) as r:
        if r.status == 403:
            raise PermissionError(f"No permission: {path}")
        if r.status != 200:
            text = await r.text()
            raise Exception(f"GET {path} failed ({r.status}): {text}")
        return await r.json()


async def api_post(session: aiohttp.ClientSession, token: str, path: str, payload: dict):
    headers = {"Authorization": token, "Content-Type": "application/json"}
    async with session.post(f"{BASE}{path}", headers=headers, json=payload) as r:
        if r.status not in (200, 201):
            text = await r.text()
            raise Exception(f"POST {path} failed ({r.status}): {text}")
        return await r.json()


async def api_post_multipart(session: aiohttp.ClientSession, token: str, path: str,
                              data: aiohttp.FormData):
    headers = {"Authorization": token}
    async with session.post(f"{BASE}{path}", headers=headers, data=data) as r:
        if r.status not in (200, 201):
            text = await r.text()
            raise Exception(f"POST multipart {path} failed ({r.status}): {text}")
        return await r.json()


async def api_delete(session: aiohttp.ClientSession, token: str, path: str):
    headers = {"Authorization": token}
    async with session.delete(f"{BASE}{path}", headers=headers) as r:
        return r.status


# ── Server cloner ──────────────────────────────────────────────────────────────

async def clone_server(user_token: str, source_id: str, target_id: str, progress_cb) -> str:
    async with aiohttp.ClientSession() as session:

        # Validate token
        try:
            await api_get(session, user_token, "/users/@me")
        except Exception as e:
            return f"❌ Invalid token or API error: {e}"

        await progress_cb("✅ Token verified — starting clone...")

        # Fetch source guild data
        try:
            src_roles = await api_get(session, user_token, f"/guilds/{source_id}/roles")
            src_channels = await api_get(session, user_token, f"/guilds/{source_id}/channels")
        except Exception as e:
            return f"❌ Could not read source server: {e}"

        # Fetch target guild channels to delete
        try:
            tgt_channels = await api_get(session, user_token, f"/guilds/{target_id}/channels")
        except Exception as e:
            return f"❌ Could not read target server: {e}"

        # Delete existing channels in target
        await progress_cb(f"🗑️ Deleting {len(tgt_channels)} existing channels...")
        for ch in tgt_channels:
            try:
                await api_delete(session, user_token, f"/channels/{ch['id']}")
            except Exception:
                pass
            await asyncio.sleep(0.4)

        # Clone roles (skip @everyone)
        src_roles_sorted = sorted(
            [r for r in src_roles if r["name"] != "@everyone"],
            key=lambda r: r["position"]
        )
        await progress_cb(f"🎭 Cloning {len(src_roles_sorted)} roles...")
        role_map: dict[str, str] = {}
        for role in src_roles_sorted:
            try:
                new_role = await api_post(session, user_token, f"/guilds/{target_id}/roles", {
                    "name": role["name"],
                    "permissions": role["permissions"],
                    "color": role["color"],
                    "hoist": role["hoist"],
                    "mentionable": role["mentionable"],
                })
                role_map[role["id"]] = new_role["id"]
                await asyncio.sleep(0.4)
            except Exception as e:
                await progress_cb(f"⚠️ Skipped role `{role['name']}`: {e}")

        # Clone categories first
        categories = sorted(
            [c for c in src_channels if c["type"] == 4],
            key=lambda c: c["position"]
        )
        await progress_cb(f"📁 Cloning {len(categories)} categories...")
        category_map: dict[str, str] = {}
        for cat in categories:
            try:
                new_cat = await api_post(session, user_token, f"/guilds/{target_id}/channels", {
                    "name": cat["name"],
                    "type": 4,
                    "position": cat["position"],
                })
                category_map[cat["id"]] = new_cat["id"]
                await asyncio.sleep(0.4)
            except Exception as e:
                await progress_cb(f"⚠️ Skipped category `{cat['name']}`: {e}")

        # Clone text and voice channels
        channels = sorted(
            [c for c in src_channels if c["type"] in (0, 2)],
            key=lambda c: c["position"]
        )
        await progress_cb(f"💬 Cloning {len(channels)} channels...")
        text_channel_map: dict[str, str] = {}
        for ch in channels:
            try:
                payload: dict = {
                    "name": ch["name"],
                    "type": ch["type"],
                    "position": ch["position"],
                }
                if ch.get("topic"):
                    payload["topic"] = ch["topic"]
                if ch.get("nsfw"):
                    payload["nsfw"] = ch["nsfw"]
                if ch.get("bitrate") and ch["type"] == 2:
                    payload["bitrate"] = ch["bitrate"]
                if ch.get("user_limit") and ch["type"] == 2:
                    payload["user_limit"] = ch["user_limit"]
                parent = ch.get("parent_id")
                if parent and parent in category_map:
                    payload["parent_id"] = category_map[parent]
                new_ch = await api_post(session, user_token, f"/guilds/{target_id}/channels", payload)
                if ch["type"] == 0:
                    text_channel_map[ch["id"]] = new_ch["id"]
                await asyncio.sleep(0.4)
            except Exception as e:
                await progress_cb(f"⚠️ Skipped channel `{ch['name']}`: {e}")

        # Copy last 15 messages from each text channel
        await progress_cb(f"📨 Copying messages from {len(text_channel_map)} text channels...")
        total_msgs = 0
        for src_ch_id, tgt_ch_id in text_channel_map.items():
            try:
                messages = await api_get(
                    session, user_token,
                    f"/channels/{src_ch_id}/messages?limit=15"
                )
            except PermissionError:
                continue  # Skip private/inaccessible channels silently
            except Exception:
                continue

            for msg in reversed(messages):
                try:
                    author = msg.get("author", {})
                    username = author.get("username", "Unknown")
                    discriminator = author.get("discriminator", "0")
                    tag = f"{username}#{discriminator}" if discriminator not in ("0", "0000") else username
                    content = msg.get("content", "")
                    timestamp = msg.get("timestamp", "")[:10]
                    avatar_hash = author.get("avatar")
                    author_id = author.get("id", "0")
                    avatar_url = (
                        f"https://cdn.discordapp.com/avatars/{author_id}/{avatar_hash}.png"
                        if avatar_hash else
                        "https://cdn.discordapp.com/embed/avatars/0.png"
                    )

                    # Handle forwarded messages (message_reference with snapshot)
                    ref_prefix = ""
                    if msg.get("message_reference") and msg.get("referenced_message"):
                        ref = msg["referenced_message"]
                        ref_author = ref.get("author", {}).get("username", "Unknown")
                        ref_content = ref.get("content", "")[:100]
                        ref_prefix = f"> **{ref_author}:** {ref_content}\n"

                    description = f"{ref_prefix}{content}" if (ref_prefix or content) else None

                    # Main embed (preserves author + text)
                    main_embed: dict = {
                        "color": 0x5865F2,
                        "author": {
                            "name": f"{tag}  •  {timestamp}",
                            "icon_url": avatar_url
                        }
                    }
                    if description:
                        main_embed["description"] = description[:4096]

                    # Attachments: images go into embed image/fields; other files as links
                    attachments = msg.get("attachments", [])
                    image_attachments = [
                        a for a in attachments
                        if a.get("content_type", "").startswith("image/")
                        or a.get("url", "").lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp"))
                    ]
                    file_attachments = [a for a in attachments if a not in image_attachments]

                    # First image goes into embed image
                    if image_attachments:
                        main_embed["image"] = {"url": image_attachments[0]["url"]}

                    # File links as fields
                    if file_attachments:
                        main_embed["fields"] = [
                            {"name": "📎 File", "value": f"[{a.get('filename', 'file')}]({a['url']})", "inline": False}
                            for a in file_attachments[:5]
                        ]

                    embeds = [main_embed]

                    # Additional images as extra embeds (Discord shows them stacked)
                    for extra_img in image_attachments[1:4]:
                        embeds.append({"url": "https://discord.com", "image": {"url": extra_img["url"]}})

                    # Original embeds from the message (bot embeds, link previews, etc.)
                    for orig_embed in msg.get("embeds", [])[:5]:
                        cleaned = {k: v for k, v in orig_embed.items()
                                   if k in ("title", "description", "url", "color", "image",
                                            "thumbnail", "fields", "author", "footer")}
                        if cleaned:
                            embeds.append(cleaned)

                    # Skip entirely empty messages
                    if not description and not attachments and not msg.get("embeds"):
                        continue

                    await api_post(session, user_token, f"/channels/{tgt_ch_id}/messages",
                                   {"embeds": embeds[:10]})
                    total_msgs += 1
                    await asyncio.sleep(0.5)
                except Exception:
                    continue

        return (
            f"✅ **Clone complete!**\n"
            f"• {len(src_roles_sorted)} roles\n"
            f"• {len(categories)} categories\n"
            f"• {len(channels)} channels\n"
            f"• {total_msgs} messages copied\n"
            f"cloned from `{source_id}` → `{target_id}`"
        )


# ── Promote helper ─────────────────────────────────────────────────────────────

async def run_promotion(user_token: str, invite_code: str,
                        promo_text: str, image_urls: list[str],
                        progress_cb) -> str:
    # Strip full URL to get just the code
    invite_code = invite_code.strip().rstrip("/").split("/")[-1]

    async with aiohttp.ClientSession() as session:
        # Validate token
        try:
            await api_get(session, user_token, "/users/@me")
        except Exception as e:
            return f"❌ Invalid token: {e}"

        # Resolve invite to get guild id
        try:
            invite_data = await api_get(session, user_token, f"/invites/{invite_code}?with_counts=true")
            guild_id = invite_data["guild"]["id"]
            guild_name = invite_data["guild"].get("name", guild_id)
        except Exception as e:
            return f"❌ Could not resolve invite: {e}"

        await progress_cb(f"🔍 Found server: **{guild_name}** — fetching members...")

        # Fetch members (up to 1000)
        try:
            members = []
            after = "0"
            while True:
                batch = await api_get(session, user_token,
                                      f"/guilds/{guild_id}/members?limit=1000&after={after}")
                if not batch:
                    break
                members.extend(batch)
                if len(batch) < 1000:
                    break
                after = batch[-1]["user"]["id"]
                await asyncio.sleep(0.5)
        except Exception as e:
            return f"❌ Could not fetch members: {e}"

        # Filter out bots
        humans = [m for m in members if not m.get("user", {}).get("bot", False)]
        await progress_cb(f"📤 Sending promotion to {len(humans)} members...")

        sent = 0
        failed = 0
        for member in humans:
            user = member.get("user", {})
            uid = user.get("id")
            if not uid:
                continue
            try:
                # Open DM channel
                dm = await api_post(session, user_token, "/users/@me/channels",
                                    {"recipient_id": uid})
                dm_id = dm["id"]

                # Build embeds: images first, then text
                embeds = []
                for i, url in enumerate(image_urls[:3]):
                    img_embed: dict = {"color": 0x5865F2, "image": {"url": url}}
                    if i == 0:
                        img_embed["url"] = "https://discord.com"  # group images
                    embeds.append(img_embed)

                if promo_text:
                    text_embed = {"description": promo_text, "color": 0x5865F2}
                    embeds.append(text_embed)

                await api_post(session, user_token, f"/channels/{dm_id}/messages",
                               {"embeds": embeds[:10]})
                sent += 1
                await asyncio.sleep(1.2)  # Avoid rate limits on DMs
            except Exception:
                failed += 1
                continue

        return (
            f"✅ **Promotion sent!**\n"
            f"• ✉️ Sent: {sent}\n"
            f"• ❌ Failed (DMs closed): {failed}\n"
            f"• Server: {guild_name}"
        )


# ── Bot setup ──────────────────────────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Tracks ongoing promote conversations: {user_id: {step, token, invite, text, images}}
promote_sessions: dict[int, dict] = {}


@bot.event
async def on_ready():
    for guild in bot.guilds:
        try:
            await bot.tree.sync(guild=guild)
            print(f'Slash commands synced to guild: {guild.name}')
        except Exception as e:
            print(f'Failed to sync to {guild.name}: {e}')
    print(f'Logged in as {bot.user.name} (ID: {bot.user.id})')
    print('Bot is online! Prefix (!) and slash (/) commands ready.')
    print('------')


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        await bot.process_commands(message)
        return

    uid = message.author.id

    if uid in promote_sessions:
        session = promote_sessions[uid]

        if session["step"] == "awaiting_text":
            session["text"] = message.content
            session["step"] = "awaiting_images"
            await message.channel.send(
                "🖼️ **Send up to 3 images** you want to include in the promotion.\n"
                "Or type `skip` to send text only."
            )
            return

        elif session["step"] == "awaiting_images":
            image_urls: list[str] = []
            if message.content.strip().lower() != "skip":
                for att in message.attachments:
                    ct = att.content_type or ""
                    if ct.startswith("image/") or att.url.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                        image_urls.append(att.url)
                    if len(image_urls) >= 3:
                        break

            token = session["token"]
            invite = session["invite"]
            promo_text = session["text"]
            del promote_sessions[uid]

            status_msg = await message.channel.send("⏳ Starting promotion...")

            async def progress(text: str):
                await status_msg.edit(content=text)

            result = await run_promotion(token, invite, promo_text, image_urls, progress)
            await status_msg.edit(content=result)
            return

    await bot.process_commands(message)


@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        raw = ctx.message.content.lstrip("!").strip()
        if raw:
            async with ctx.typing():
                try:
                    reply = await handle_code_prompt(raw)
                    await ctx.send(reply)
                except Exception:
                    await ctx.send(
                        "❓ Unknown command. Available: `!ping`, `!code`, `!clone`, `!promote`"
                    )
        else:
            await ctx.send("❓ Unknown command. Available: `!ping`, `!code`, `!clone`, `!promote`")

    elif isinstance(error, commands.MissingRequiredArgument):
        cmd = ctx.command.name if ctx.command else "unknown"
        hints = {
            "code": "`!code <your question>`",
            "clone": "`!clone <user_token> <source_id> <target_id>`",
            "promote": "`!promote <user_token> <invite_link>`",
            "ping": "`!ping`",
        }
        hint = hints.get(cmd, f"`!{cmd} <arguments>`")
        await ctx.send(f"⚠️ Missing arguments for `!{cmd}`. Usage: {hint}")
    else:
        await ctx.send(f"⚠️ Error: {str(error)}")


# ── Ping ───────────────────────────────────────────────────────────────────────

@bot.command(name="ping")
async def prefix_ping(ctx):
    latency = round(bot.latency * 1000)
    await ctx.send(f"Pong! 🏓 `{latency}ms`")


@bot.tree.command(name="ping", description="Check the bot's latency")
async def slash_ping(interaction: discord.Interaction):
    latency = round(bot.latency * 1000)
    await interaction.response.send_message(f"Pong! 🏓 `{latency}ms`")


# ── Code ───────────────────────────────────────────────────────────────────────

@bot.command(name="code")
async def prefix_code(ctx, *, prompt: str):
    async with ctx.typing():
        try:
            full_response = await handle_code_prompt(prompt)
            if len(full_response) > 2000:
                for i in range(0, len(full_response), 2000):
                    await ctx.send(full_response[i:i+2000])
            else:
                await ctx.send(full_response)
        except Exception as e:
            await ctx.send(f"An error occurred: {str(e)}")


@bot.tree.command(name="code", description="Ask the AI coding assistant a question")
@app_commands.describe(prompt="Your coding question or request")
async def slash_code(interaction: discord.Interaction, prompt: str):
    await interaction.response.defer()
    try:
        full_response = await handle_code_prompt(prompt)
        if len(full_response) > 2000:
            await interaction.followup.send(full_response[:2000])
            for i in range(2000, len(full_response), 2000):
                await interaction.followup.send(full_response[i:i+2000])
        else:
            await interaction.followup.send(full_response)
    except Exception as e:
        await interaction.followup.send(f"An error occurred: {str(e)}")


# ── Clone ──────────────────────────────────────────────────────────────────────

@bot.command(name="clone")
async def prefix_clone(ctx, user_token: str, source_id: str, target_id: str):
    msg = await ctx.send("⏳ Starting server clone...")

    async def progress(text: str):
        await msg.edit(content=text)

    result = await clone_server(user_token, source_id, target_id, progress)
    await msg.edit(content=result)


@bot.tree.command(name="clone", description="Clone a server's structure into another server")
@app_commands.describe(
    user_token="Your Discord user token",
    source_id="ID of the server to clone FROM",
    target_id="ID of the server to clone INTO"
)
async def slash_clone(interaction: discord.Interaction, user_token: str, source_id: str, target_id: str):
    await interaction.response.defer(ephemeral=True)
    status_msg = await interaction.followup.send("⏳ Starting server clone...", ephemeral=True)

    async def progress(text: str):
        await status_msg.edit(content=text)

    result = await clone_server(user_token, source_id, target_id, progress)
    await status_msg.edit(content=result)


# ── Promote ────────────────────────────────────────────────────────────────────

@bot.command(name="promote")
async def prefix_promote(ctx, user_token: str, invite_link: str):
    promote_sessions[ctx.author.id] = {
        "step": "awaiting_text",
        "token": user_token,
        "invite": invite_link,
        "text": "",
        "images": []
    }
    await ctx.send(
        "📢 **What do you want to promote?**\n"
        "Reply with your promotional message below."
    )


@bot.tree.command(name="promote", description="DM all members of a server with a promotional message")
@app_commands.describe(
    user_token="Your Discord user token",
    invite_link="Invite link of the target server"
)
async def slash_promote(interaction: discord.Interaction, user_token: str, invite_link: str):
    await interaction.response.send_message(
        "📢 **What do you want to promote?**\n"
        "Reply in this channel with your promotional message.",
        ephemeral=False
    )
    promote_sessions[interaction.user.id] = {
        "step": "awaiting_text",
        "token": user_token,
        "invite": invite_link,
        "text": "",
        "images": []
    }


if __name__ == "__main__":
    token = os.environ.get("DISCORD_TOKEN")
    if token:
        bot.run(token)
    else:
        print("Error: DISCORD_TOKEN not set.")
