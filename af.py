

import os
import io
import asyncio
import traceback
from datetime import datetime, timezone
from html import escape
from typing import Optional, Dict, List, Any, Tuple
import asyncpg
import numpy as np
import torch
import clip
from PIL import Image
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, FSInputFile
import logging
logger = logging.getLogger("close_view")
# ---------------- Config ----------------
TOKEN = os.getenv("tg_bot_token")
DATABASE_URL = os.getenv("DATABASE_URL")

if not TOKEN:
    raise RuntimeError("Установите tg_bot_token")
if not DATABASE_URL:
    raise RuntimeError("Установите DATABASE_URL (Postgres DSN)")

# ---------------- Bot init ----------------
bot = Bot(token=TOKEN)
dp = Dispatcher()

# ---------------- CLIP ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading CLIP on", device)
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

# ---------------- Constants ----------------
CLOTHING_CATEGORIES = [
    "t-shirt", "top", "shirt", "blouse", "hoodie", "sweater", "cardigan",
    "jeans", "trousers", "leggings", "skirt", "shorts",
    "dress", "jumpsuit",
    "jacket", "coat", "raincoat",
    "sneakers", "boots", "heels", "sandals",
    "hat", "scarf", "bag"
]

CATEGORY_MAP = {
    "t-shirt": "футболка", "top": "топ", "shirt": "рубашка", "blouse": "блузка",
    "hoodie": "худи", "sweater": "свитер", "cardigan": "кардиган",
    "jeans": "джинсы", "trousers": "брюки", "leggings": "леггинсы", "skirt": "юбка", "shorts": "шорты",
    "dress": "платье", "jumpsuit": "комбинезон",
    "jacket": "куртка", "coat": "пальто", "raincoat": "плащ",
    "sneakers": "кроссовки", "boots": "ботинки", "heels": "туфли/каблуки", "sandals": "сандалии",
    "hat": "шапка/шляпа", "scarf": "шарф", "bag": "сумка"
}

CATEGORY_GROUPS = {
    "all": {"label": "Все вещи", "items": None},
    "outer": {"label": "Верхняя одежда", "items": ["coat", "jacket", "raincoat"]},
    "tops": {"label": "Верх", "items": ["t-shirt", "top", "shirt", "blouse", "hoodie", "sweater", "cardigan"]},
    "bottoms": {"label": "Низ", "items": ["jeans", "trousers", "leggings", "shorts", "skirt"]},
    "dresses": {"label": "Платья/комбинезоны", "items": ["dress", "jumpsuit"]},
    "shoes": {"label": "Обувь", "items": ["sneakers", "boots", "heels", "sandals"]},
    "accessories": {"label": "Аксессуары", "items": ["hat", "scarf", "bag"]}
}

COLOR_LABELS = ["white","black","gray","red","orange","yellow","green","blue","purple","pink","brown","beige","maroon","olive"]
COLOR_MAP = {"white":"белый","black":"чёрный","gray":"серый","red":"красный","orange":"оранжевый","yellow":"жёлтый",
             "green":"зелёный","blue":"синий","purple":"фиолетовый","pink":"розовый","brown":"коричневый","beige":"бежевый",
             "maroon":"бордовый","olive":"оливковый"}

PAGE_SIZE = 10
# ---------------- Help text ----------------
HELP_TEXT = (
    "<b>О боте и обработке фото</b>\n\n"
    "Этот бот помогает управлять гардеробом и собирать «капсулы» — готовые подборки вещей, "
    "которые хорошо сочетаются между собой.\n\n"

    "<b>Что делает бот</b>\n"
    "• Сохраняет фото вещей в вашем гардеробе.\n"
    "• Автоматически предлагает категорию (рубашка, платье и т.д.) и цвет.\n"
    "• Хранит компактные векторные представления изображений для поиска и подбора.\n"
    "• Генерирует капсулы — наборы вещей с хорошей визуальной согласованностью.\n"
    "• Позволяет добавлять теги, описания, удалять и просматривать вещи и капсулы.\n\n"

    "<b>Как обрабатываются фото</b>\n"
    "1. Когда вы отправляете фото, бот предлагает добавить его в гардероб или просто проанализировать.\n"
    "2. Если вы добавляете — изображение скачивается и обрабатывается моделью CLIP, которая выдаёт:\n"
    "   • предположительную категорию (например «рубашка»),\n"
    "   • предполагаемый основной цвет и уровень доверия (в %),\n"
    "3. В базу данных сохраняются: ссылка на файл, метаданные (название, цвет, категория, описание).\n\n"

    "<b>Безопасность и приватность</b>\n"
    "• Бот не рассылает ваши фотографии третьим лицам автоматически.\n"
    "• Обработка выполняется там, где запущен бот: на вашем сервере или хостинге. Если бот у вас — данные остаются у вас.\n"

    "<b>Как пользоваться (коротко)</b>\n"
    "• /start — главное окно с кнопками.\n"
    "• /help — это сообщение.\n"
    "• «Создать капсулу» — сгенерировать подборку из ваших вещей.\n"
    "• «Мой гардероб» — просмотреть категории, добавить вещь, перейти в поиск.\n"
    "• В карточке вещи: добавить тег, добавить описание, удалить вещь, вернуться назад.\n\n"

    "<b>Советы</b>\n"
    "• Если модель предлагает не тот цвет/категорию — выберите «ввести вручную» и исправьте.\n"
)

# ---------------- In-memory states ----------------
pending_add: Dict[int, Dict[str, Any]] = {}
pending_action: Dict[int, Dict[str, Any]] = {}
pending_capsule: Dict[int, Dict[str, Any]] = {}
pending_photo_offer: Dict[int, Dict[str, Any]] = {}
last_menu_message: Dict[int, Dict[str, Any]] = {}  # хранит единственное текущее меню (chat_id, message_id, type)

# ---------------- DB pool ----------------
db_pool: asyncpg.pool.Pool = None

# ---------------- Keyboards ----------------
def main_menu_kb():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="👗 Мой гардероб", callback_data="menu_wardrobe")],
        [InlineKeyboardButton(text="🧩 Создать капсулу", callback_data="menu_generate_capsule")],
        [InlineKeyboardButton(text="📚 Мои капсулы", callback_data="menu_view_capsules")],
        [InlineKeyboardButton(text="❓ Помощь", callback_data="menu_help")]
    ])

def wardrobe_menu_kb_dynamic():
    rows = []
    for gid, info in CATEGORY_GROUPS.items():
        rows.append([InlineKeyboardButton(text=info["label"], callback_data=f"wardrobe_group:{gid}")])
    rows.append([InlineKeyboardButton(text="➕ Добавить вещь", callback_data="wardrobe_add_item"),
                 InlineKeyboardButton(text="🔎 Поиск", callback_data="wardrobe_search")])
    rows.append([InlineKeyboardButton(text="↩️ Назад в меню", callback_data="menu_back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)

def kb_name_choice():
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="Принять название ✅", callback_data="add_accept_name"),
                                               InlineKeyboardButton(text="Ввести название ✍️", callback_data="add_enter_name")]])

def kb_color_choice():
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="Принять цвет ✅", callback_data="add_accept_color"),
                                               InlineKeyboardButton(text="Ввести цвет ✍️", callback_data="add_enter_color")]])

def kb_final_choice():
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="Сохранить вещь 💾", callback_data="add_save"),
                                               InlineKeyboardButton(text="Отмена ❌", callback_data="add_cancel")]])

def feedback_kb():
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Да, верно ✅", callback_data="fb_yes")],
        [InlineKeyboardButton(text="Нет, подумай ещё 🔁", callback_data="fb_no_retry"),
         InlineKeyboardButton(text="Нет — я введу сам(а) ✍️", callback_data="fb_no_input")]
    ])

# ---------------- Utilities ----------------
def normalize_russian(s: Optional[str]) -> str:
    if not s:
        return ""
    x = s.lower().replace("ё", "е").strip()
    if x.endswith("ая") or x.endswith("яя"):
        return x[:-2] + "ый"
    if x.endswith("ое") or x.endswith("ее"):
        return x[:-2] + "ый"
    if x.endswith("ые") or x.endswith("ые"):
        return x[:-2] + "ый"
    return x

def format_dt(dt):
    try:
        if not dt:
            return "-"
        if isinstance(dt, datetime):
            return dt.astimezone().strftime("%d.%m.%Y %H:%M")
        return str(dt)
    except Exception:
        return str(dt)

def to_vector_from_bytes(b: Optional[bytes]) -> Optional[np.ndarray]:
    if b is None:
        return None
    return np.frombuffer(b, dtype=np.float32)

def cosine_sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return -1.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)

async def safe_delete_message(chat_id: int, message_id: int):
    try:
        if chat_id and message_id:
            await bot.delete_message(chat_id=chat_id, message_id=message_id)
    except Exception:
        pass

# NEW HELPER: удаляем сохранённое меню, если оно отличается от текущего callback.message
async def clear_last_menu_if_different(user_id: int, callback_message: Optional[types.Message] = None):
    prev = last_menu_message.get(user_id)
    if not prev:
        return
    try:
        # если prev это то же сообщение, что сейчас редактируем — ничего не трогаем
        if callback_message and prev.get("chat_id") == getattr(callback_message, "chat", {}).get("id", getattr(callback_message, "chat", None)) and prev.get("message_id") == getattr(callback_message, "message_id", None):
            return
    except Exception:
        # на всякий случай — продолжим и удалим prev
        pass
    try:
        await safe_delete_message(prev.get("chat_id"), prev.get("message_id"))
    except Exception:
        pass
    last_menu_message.pop(user_id, None)

async def show_menu(user_id: int, callback_msg: Optional[types.Message], text: str, kb, typ: str):
    """Единообразно показывает новое меню без дублей в чате."""

    # 1) Удаляем прошлое меню, если оно существует
    prev = last_menu_message.get(user_id)
    if prev:
        try:
            # Не удаляем если это то же сообщение, которое сейчас редактируем
            same = callback_msg and (
                prev["chat_id"] == callback_msg.chat.id and
                prev["message_id"] == callback_msg.message_id
            )
            if not same:
                await safe_delete_message(prev["chat_id"], prev["message_id"])
        except:
            pass
        last_menu_message.pop(user_id, None)

    # 2) Пробуем отредактировать сообщение, из которого пришёл callback
    if callback_msg:
        try:
            sent = await bot.edit_message_text(
                text, callback_msg.chat.id, callback_msg.message_id,
                reply_markup=kb, parse_mode="HTML"
            )
            last_menu_message[user_id] = {
                "chat_id": sent.chat.id,
                "message_id": sent.message_id,
                "type": typ
            }
            return sent
        except:
            pass

    # 3) Если редактирование нельзя — отправляем новое
    sent = await bot.send_message(user_id, text, reply_markup=kb, parse_mode="HTML")
    last_menu_message[user_id] = {
        "chat_id": sent.chat.id,
        "message_id": sent.message_id,
        "type": typ
    }
    return sent

async def replace_menu_message(user_id: int, callback_message: Optional[types.Message], text: str, reply_markup=None, parse_mode="HTML", typ: str = "menu"):
    """
    Единый контракт для показа/замены меню:
      - Если есть сохранённое last_menu_message — удаляем его (чтобы не дублировалось),
        **кроме** случая когда callback_message == last_menu_message (тогда проще редактировать).
      - Если callback_message доступно — пробуем редактировать его (preferred).
      - Иначе отправляем новое сообщение и сохраняем в last_menu_message.
    typ — произвольная строка для last_menu_message.type (например 'start','capsule_list' и т.д.)
    """
    prev = last_menu_message.get(user_id)
    # удаляем предыдущий, если он не тот же самый (чтобы не удалить callback.message, который мы хотим редактировать)
    try:
        if prev:
            same = False
            if callback_message and prev.get("chat_id") == callback_message.chat.id and prev.get("message_id") == callback_message.message_id:
                same = True
            if not same:
                await safe_delete_message(prev.get("chat_id"), prev.get("message_id"))
                last_menu_message.pop(user_id, None)
    except Exception:
        # не критично если не удалили
        pass

    # Попытаться отредактировать сообщение, из которого пришёл callback (если есть)
    if callback_message:
        try:
            sent = await bot.edit_message_text(text, chat_id=callback_message.chat.id, message_id=callback_message.message_id, reply_markup=reply_markup, parse_mode=parse_mode)
            last_menu_message[user_id] = {"chat_id": sent.chat.id, "message_id": sent.message_id, "type": typ}
            return sent
        except Exception:
            # fallthrough -> отправим новое
            pass

    # отправляем новое (одно)
    sent = await bot.send_message(user_id, text, reply_markup=reply_markup, parse_mode=parse_mode)
    last_menu_message[user_id] = {"chat_id": sent.chat.id, "message_id": sent.message_id, "type": typ}
    return sent

async def reply_or_edit(original_message: Optional[types.Message], chat_id: int, text: str, reply_markup=None, parse_mode="HTML"):
    try:
        if original_message and getattr(original_message, "chat", None) and getattr(original_message, "message_id", None):
            await bot.edit_message_text(text=text, chat_id=original_message.chat.id, message_id=original_message.message_id, parse_mode=parse_mode, reply_markup=reply_markup)
            return original_message
    except Exception:
        pass
    sent = await bot.send_message(chat_id, text, parse_mode=parse_mode, reply_markup=reply_markup)
    return sent

async def send_main_menu(user_id: int, text: Optional[str] = None, photo_path: Optional[str] = None):
    """
    Показать главное меню.
    Теперь объединяет картинку и текст в одно сообщение через caption.
    """
    # 1. Удаляем предыдущее меню (чтобы не дублировалось)
    prev = last_menu_message.get(user_id)
    if prev:
        try:
            await safe_delete_message(prev.get("chat_id"), prev.get("message_id"))
        except Exception:
            pass
        last_menu_message.pop(user_id, None)

    # Если текст пустой, ставим None
    if text is not None and text.strip() == "":
        text = None

    # Автопоиск картинки
    if not photo_path:
        for p in ("assets/welcome.png", "assets/welcome.jpg", "assets/start.png", "assets/start.jpg", "assets/logo.png"):
            if os.path.isfile(p):
                photo_path = p
                break

    kb = main_menu_kb()

    # 2. Пытаемся отправить фото СРАЗУ с текстом (caption)
    if photo_path and os.path.isfile(photo_path):
        try:
            img = FSInputFile(photo_path)
            # ВАЖНО: передаем text в caption
            sent = await bot.send_photo(user_id, photo=img, caption=text, parse_mode="HTML", reply_markup=kb)
            last_menu_message[user_id] = {"chat_id": sent.chat.id, "message_id": sent.message_id, "type": "start"}
            return
        except Exception as e:
            # Если текст слишком длинный (>1024) или другая ошибка фото,
            # пробуем отправить раздельно (фоллбек)
            print("send_main_menu: caption failed, sending separately:", e)
            try:
                img = FSInputFile(photo_path)
                sent = await bot.send_photo(user_id, photo=img, caption=None, reply_markup=kb)
                last_menu_message[user_id] = {"chat_id": sent.chat.id, "message_id": sent.message_id, "type": "start"}
                if text:
                    await bot.send_message(user_id, text, parse_mode="HTML")
                return
            except Exception:
                pass # Если и так не вышло, идем к отправке просто текста

    # 3. Если картинки нет или отправка упала — отправляем просто текст с кнопками
    try:
        prompt = text if text else "Выберите действие ниже."
        sent = await bot.send_message(user_id, prompt, parse_mode="HTML", reply_markup=kb)
        last_menu_message[user_id] = {"chat_id": sent.chat.id, "message_id": sent.message_id, "type": "start"}
    except Exception as e:
        print("send_main_menu fallback failed:", e)

# ---------------- Database helpers ----------------
async def create_pool_with_retries(dsn: str, attempts: int = 5, delay: float = 2.0):
    last_exc = None
    for i in range(attempts):
        try:
            pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
            return pool
        except Exception as e:
            last_exc = e
            print(f"[db] connect attempt {i+1}/{attempts} failed: {e}")
            await asyncio.sleep(delay)
    print(f"Не удалось подключиться к базе данных: {last_exc}")
    raise last_exc

async def init_db_and_migrate():
    async with db_pool.acquire() as conn:
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS wardrobe (
            id SERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL,
            file_id TEXT NOT NULL,
            emb BYTEA,
            name TEXT,
            color_en TEXT,
            color_ru TEXT,
            category_en TEXT,
            category_ru TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            description TEXT DEFAULT ''
        );
        """)
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS tags (
            id SERIAL PRIMARY KEY,
            item_id INTEGER NOT NULL REFERENCES wardrobe(id) ON DELETE CASCADE,
            user_id BIGINT NOT NULL,
            tag TEXT NOT NULL
        );
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_tags_item ON tags(item_id);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_tags_tag_lower ON tags(LOWER(tag));")
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS capsules (
            id SERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL,
            name TEXT NOT NULL,
            item_ids INTEGER[] NOT NULL,
            thumbnail_file_id TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            description TEXT DEFAULT ''
        );
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_capsules_user ON capsules(user_id);")

# ---------------- CLIP helpers ----------------
def clip_infer_logits(image_tensor):
    with torch.no_grad():
        text_cat = [f"a photo of a {c}" for c in CLOTHING_CATEGORIES]
        text_tokens = clip.tokenize(text_cat).to(device)
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = model.logit_scale.exp().to(device)
        cat_logits = (image_features @ text_features.t()).squeeze(0) * logit_scale
    return cat_logits.cpu()

def clip_color_logits(image_tensor):
    with torch.no_grad():
        text_colors = [f"the color is {c}" for c in COLOR_LABELS]
        color_tokens = clip.tokenize(text_colors).to(device)
        image_features = model.encode_image(image_tensor)
        color_features = model.encode_text(color_tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        color_features = color_features / color_features.norm(dim=-1, keepdim=True)
        logit_scale = model.logit_scale.exp().to(device)
        color_logits = (image_features @ color_features.t()).squeeze(0) * logit_scale
    return color_logits.cpu()

# ---------------- Capsule generation (улучшенный) ----------------
async def generate_capsule_items_for_user(user_id: int, candidates_per_group: int = 25) -> Tuple[List[Dict[str, Any]], float]:
    groups = {
        "tops": CATEGORY_GROUPS["tops"]["items"],
        "bottoms": CATEGORY_GROUPS["bottoms"]["items"],
        "dresses": CATEGORY_GROUPS["dresses"]["items"],
        "outer": CATEGORY_GROUPS["outer"]["items"],
        "shoes": CATEGORY_GROUPS["shoes"]["items"],
        "accessories": CATEGORY_GROUPS["accessories"]["items"]
    }

    async with db_pool.acquire() as conn:
        async def fetch_candidates(categories):
            if not categories:
                return []
            rows = await conn.fetch(
                "SELECT id, file_id, name, color_ru, category_en, emb FROM wardrobe WHERE user_id=$1 AND category_en = ANY($2::text[]) AND emb IS NOT NULL LIMIT $3",
                user_id, categories, candidates_per_group
            )
            items = []
            for r in rows:
                vec = to_vector_from_bytes(r['emb'])
                if vec is None:
                    continue
                items.append({
                    "id": r['id'],
                    "file_id": r['file_id'],
                    "name": r['name'] or "",
                    "color_ru": r['color_ru'] or "",
                    "category_en": r['category_en'] or "",
                    "emb_vec": vec
                })
            return items

        candidates = {k: await fetch_candidates(v) for k, v in groups.items()}

    selected: List[Dict[str, Any]] = []

    if candidates.get("dresses"):
        selected.append(candidates["dresses"][0])
    else:
        tops = candidates.get("tops", [])
        bottoms = candidates.get("bottoms", [])
        best_pair = (None, None, -999.0)
        for t in tops:
            for b in bottoms:
                s = cosine_sim(t['emb_vec'], b['emb_vec'])
                if s > best_pair[2]:
                    best_pair = (t, b, s)
        if best_pair[0] and best_pair[1]:
            selected.append(best_pair[0]); selected.append(best_pair[1])
        else:
            if tops:
                selected.append(tops[0])
            elif bottoms:
                selected.append(bottoms[0])

    def centroid(vectors: List[np.ndarray]) -> Optional[np.ndarray]:
        if not vectors:
            return None
        arr = np.vstack(vectors)
        c = np.mean(arr, axis=0)
        norm = np.linalg.norm(c) + 1e-8
        return c / norm

    SIM_THRESHOLD = 0.18

    for slot in ("outer", "shoes", "accessories"):
        pool = candidates.get(slot, []) or []
        if not pool:
            continue
        cent = centroid([s['emb_vec'] for s in selected]) if selected else None
        best_cand = None; best_score = -999.0
        for cand in pool:
            score = cosine_sim(cand['emb_vec'], cent) if cent is not None else 0.0
            if score > best_score:
                best_score = score; best_cand = cand
        if best_cand and best_score >= SIM_THRESHOLD:
            selected.append(best_cand)

    if len(selected) < 2:
        for k in ("tops","bottoms","dresses","outer","shoes","accessories"):
            if candidates.get(k):
                selected.append(candidates[k][0])
                if len(selected) >= 2: break

    avg_pair_sim = 0.0
    if len(selected) >= 2:
        sims = []
        for i in range(len(selected)):
            for j in range(i+1, len(selected)):
                sims.append(cosine_sim(selected[i]['emb_vec'], selected[j]['emb_vec']))
        avg_pair_sim = float(np.mean(sims)) if sims else 0.0

    return selected, avg_pair_sim

# ---------------- send capsule ----------------
async def send_capsule(user_id: int, force_regen: bool = False):
    # удаляем предыдущее last_menu_message если нужно (как у тебя)
    prev = last_menu_message.get(user_id)
    if prev:
        try:
            await safe_delete_message(prev.get("chat_id"), prev.get("message_id"))
        except Exception:
            pass
        last_menu_message.pop(user_id, None)

    prev_ids = []
    old = pending_capsule.get(user_id)
    if old and old.get("items"):
        prev_ids = [int(i["id"]) for i in old["items"]]

    selected = []
    avg_sim = 0.0

    # пробуем получить другой набор; если force_regen=False — достаточно одного вызова
    attempts = 3 if force_regen else 1
    for attempt in range(attempts):
        sel, sim = await generate_capsule_items_for_user(user_id, candidates_per_group=40)
        if not sel:
            continue
        # переставим случайно — чтобы уменьшить шанс идентичности
        import random
        random.shuffle(sel)

        sel_ids = [int(r['id']) for r in sel]
        # если набор отличается по составу — принимаем; иначе повторяем
        if set(sel_ids) != set(prev_ids) or attempt == attempts - 1:
            selected = sel
            avg_sim = sim
            break
        # иначе ждём и пробуем снова (маленькая пауза для детерминированных алгоритмов, необязательно)
        await asyncio.sleep(0.05)

    if not selected:
        await send_main_menu(user_id, "Недостаточно вещей для капсулы. Добавьте вещи.")
        return

    # формируем текст и клавиатуру (используй two_buttons_from_items если есть)
    lines = [f"🧩 <b>Капсула</b> — средняя схожесть <code>{avg_sim:.2f}</code>\n"]
    for r in selected:
        lines.append(f"• {CATEGORY_MAP.get(r.get('category_en',''), r.get('category_en',''))}: <b>{escape(r['name'] or '(без названия)')}</b>")
    text = "\n".join(lines)

    kb_rows = []
    try:
        kb_rows.extend(two_buttons_from_items(selected, lambda r: f"view_item_from_capsule:{r['id']}"))
    except Exception:
        for r in selected:
            kb_rows.append([InlineKeyboardButton(text=r.get('name') or "(без названия)", callback_data=f"view_item_from_capsule:{r['id']}")])

    kb_rows.append([InlineKeyboardButton(text="💾 Сохранить капсулу", callback_data="save_capsule"),
                    InlineKeyboardButton(text="🔁 Перегенерировать", callback_data="generate_capsule")])
    kb_rows.append([InlineKeyboardButton(text="❌ Закрыть", callback_data="close_capsule")])
    kb = InlineKeyboardMarkup(inline_keyboard=kb_rows)

    # отправка/редактирование сообщения
    preview_file = selected[0].get('file_id') if selected and selected[0].get('file_id') else None
    if preview_file:
        sent = await bot.send_photo(user_id, photo=preview_file, caption=text, parse_mode="HTML", reply_markup=kb)
    else:
        sent = await bot.send_message(user_id, text, parse_mode="HTML", reply_markup=kb)

    pending_capsule[user_id] = {
        "items": [{"id": r['id'], "name": r['name'], "file_id": r.get('file_id'), "category_en": r.get('category_en')} for r in selected],
        "avg_sim": avg_sim, "text": text, "chat_id": sent.chat.id, "message_id": sent.message_id, "created": datetime.now(timezone.utc)
    }
    last_menu_message[user_id] = {"chat_id": sent.chat.id, "message_id": sent.message_id, "type": "capsule"}


# ---------------- Handlers ----------------
@dp.message(Command(commands=["start"]))
async def cmd_start(message: types.Message):
    user_id = message.from_user.id
    caption = (
        "<b>Привет! Я — бот для управления гардеробом и сборки капсул.</b>\n\n"
        "• Собирать капсулы из твоих вещей\n"
        "• Сохранять подборки\n"
        "• Искать вещи по тегам/описанию\n\n"
        "Выбери действие ниже."
    )
    await send_main_menu(user_id, caption)

@dp.message(Command(commands=["help"]))
async def cmd_help(message: types.Message):
    await send_main_menu(message.from_user.id, HELP_TEXT)
@dp.callback_query(lambda c: c.data == "save_capsule")
async def save_capsule_callback(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    cap = pending_capsule.get(user_id)
    if not cap or not cap.get("items"):
        await callback.answer("Нет текущей капсулы для сохранения.", show_alert=True)
        return

    item_ids = [int(i["id"]) for i in cap["items"]]
    name = f"Capsule {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}"
    async with db_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO capsules(user_id, name, item_ids, created_at) VALUES($1,$2,$3,$4)",
            user_id, name, item_ids, datetime.now(timezone.utc)
        )

    await callback.answer("Капсула сохранена ✅")
    # обновим pending_capsule (опционально)
    pending_capsule[user_id].update({"saved": True, "saved_name": name})

@dp.message(lambda m: m.text is not None)
async def text_router(message: types.Message):
    text = (message.text or "").strip()
    lower = text.lower()
    user_id = message.from_user.id

    # 1) Pending actions (save_capsule_with_name, add_tag, add_desc)
    pa = pending_action.get(user_id)
    if pa:
        action = pa.get("action")
        if action == "add_tag":
            item_id = pa.get("item_id"); tag = text.strip()
            if not tag:
                await bot.send_message(user_id, "Тег не может быть пустым. Введите текст или /cancel.")
                return
            async with db_pool.acquire() as conn:
                exists = await conn.fetchval("SELECT 1 FROM tags WHERE item_id=$1 AND LOWER(tag)=LOWER($2) LIMIT 1", item_id, tag)
                if exists:
                    await bot.send_message(user_id, f"Тег «{escape(tag)}» уже есть.")
                else:
                    await conn.execute("INSERT INTO tags(item_id, user_id, tag) VALUES ($1, $2, $3)", item_id, user_id, tag)
                    await bot.send_message(user_id, f"Тег «{escape(tag)}» добавлен.")
            pending_action.pop(user_id, None); return

        if action == "add_desc":
            item_id = pa.get("item_id"); desc = text.strip()
            async with db_pool.acquire() as conn:
                found = await conn.fetchval("SELECT 1 FROM wardrobe WHERE id=$1 AND user_id=$2", item_id, user_id)
                if not found:
                    await bot.send_message(user_id, "Вещь не найдена или нет прав.")
                else:
                    await conn.execute("UPDATE wardrobe SET description=$1 WHERE id=$2 AND user_id=$3", desc, item_id, user_id)
                    await bot.send_message(user_id, "Описание сохранено.")
            pending_action.pop(user_id, None); return

        if action == "save_capsule_with_name":
            name = text.strip()
            if not name:
                await bot.send_message(user_id, "Имя не может быть пустым. Введите ещё раз или /cancel.")
                return
            items = pa.get("items", []); thumbnail = pa.get("thumbnail")
            async with db_pool.acquire() as conn:
                row = await conn.fetchrow("""
                    INSERT INTO capsules (user_id, name, item_ids, thumbnail_file_id, created_at)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING id
                """, user_id, name, items, thumbnail, datetime.now(timezone.utc))
            pending_action.pop(user_id, None)
            pending_capsule.pop(user_id, None)
            await send_main_menu(user_id, f"Капсула <b>{escape(name)}</b> сохранена ✅ (id {row['id']}).")
            return

    # 2) pending_add states (awaiting_name / awaiting_color / wait_search_text)
    state = pending_add.get(user_id)
    # безопасно определяем stage — если state == None, stage будет None
    stage = state.get("stage") if state else None

    if state and stage == "awaiting_name":
            # сохраняем имя и переводим в ready_to_confirm
            state["name"] = text
            state["stage"] = "ready_to_confirm"

            # попытка взять предложенный цвет из state (несколько ключей)
            suggested_color = (
                    state.get("suggested_color_ru")
                    or state.get("color_ru")
                    or state.get("suggested_color")
                    or state.get("color")
                    or ""
            )
            # если цвет английский код — переведём в русский через COLOR_MAP
            if suggested_color and suggested_color.lower() in COLOR_MAP:
                suggested_color = COLOR_MAP[suggested_color.lower()]

            # если нужного поля нет, оставим fallback — показать имя и попросить выбрать цвет
            if suggested_color:
                msg_text = f"Предлагаемый цвет: <b>{escape(suggested_color)}</b>"
            else:
                msg_text = f"Название сохранено: <b>{escape(text)}</b>\nВыберите цвет:"

            # пробуем отредактировать существующее suggestion-message, иначе отправляем новое
            prev_chat = state.get("suggestion_chat_id")
            prev_msg = state.get("suggestion_message_id")
            if prev_chat and prev_msg:
                try:
                    sent = await bot.edit_message_text(
                        msg_text,
                        chat_id=prev_chat,
                        message_id=prev_msg,
                        parse_mode="HTML",
                        reply_markup=kb_color_choice()
                    )
                    state["suggestion_chat_id"] = sent.chat.id
                    state["suggestion_message_id"] = sent.message_id
                    return
                except Exception:
                    # если редактирование упало — удалим старое и отправим новое дальше
                    try:
                        await safe_delete_message(prev_chat, prev_msg)
                    except Exception:
                        pass

            # отправляем новое сообщение с клавиатурой выбора цвета
            try:
                sent = await bot.send_message(user_id, msg_text, parse_mode="HTML", reply_markup=kb_color_choice())
                state["suggestion_chat_id"] = sent.chat.id
                state["suggestion_message_id"] = sent.message_id
            except Exception:
                # фоллбек — текстовое уведомление
                try:
                    await bot.send_message(user_id, "Название сохранено. Выберите цвет (введите текстом):")
                except Exception:
                    pass
            return
    if stage == "awaiting_color":
        # пользователь ввёл цвет вручную — сопоставляем с COLOR_MAP
        color_input = text.strip()
        if not color_input:
            await bot.send_message(user_id, "Цвет не может быть пустым. Введите цвет или /cancel.")
            return

        # пытаемся сопоставить русский/англ. варианты
        color_en = ""
        color_ru = color_input
        # сначала прямое совпадение по русской мапе (значения COLOR_MAP)
        for en, ru in COLOR_MAP.items():
            if ru.lower() == color_input.lower() or en.lower() == color_input.lower():
                color_en = en
                color_ru = ru
                break

        # если не найдена, попробуем нормализацию (на случай 'зелёная' -> 'зеленый' и т.д.)
        if not color_en:
            norm = normalize_russian(color_input)
            for en, ru in COLOR_MAP.items():
                if en.lower() == norm or ru.lower() == norm:
                    color_en = en
                    color_ru = ru
                    break

        # сохраняем в state (если хотите — сохраняем как введённый текст, даже если не распознали)
        state["color_ru"] = color_ru
        state["color_en"] = color_en or ""

        # переводим в готово и показываем финальную панель сохранения
        state["stage"] = "ready_to_confirm"
        try:
            await bot.edit_message_text(
                f"Название: <b>{escape(state.get('name', ''))}</b>\nЦвет: <b>{escape(color_ru)}</b>\n\nГотово к сохранению.",
                chat_id=state.get("suggestion_chat_id"),
                message_id=state.get("suggestion_message_id"),
                parse_mode="HTML",
                reply_markup=kb_final_choice()
            )
        except Exception:
            # если редактировать нельзя — отправим новое сообщение с тем же текстом
            try:
                sent = await bot.send_message(user_id,
                                              f"Название: <b>{escape(state.get('name', ''))}</b>\nЦвет: <b>{escape(color_ru)}</b>\n\nГотово к сохранению.",
                                              parse_mode="HTML",
                                              reply_markup=kb_final_choice())
                # обновляем ссылку на suggestion message
                state["suggestion_chat_id"] = sent.chat.id
                state["suggestion_message_id"] = sent.message_id
            except Exception:
                await bot.send_message(user_id, "Цвет сохранён. Готово к сохранению.", reply_markup=kb_final_choice())

        return

    # 3) Fallback textual commands
    if lower.startswith("/capsule") or "капсул" in lower:
        await send_capsule(user_id); return
    if "гардероб" in lower:
        # open wardrobe menu — try to reuse replace_menu_message for consistent behaviour
        lm = last_menu_message.get(user_id)
        if lm:
            try:
                # try to edit existing saved menu message (prefer), else replace
                await replace_menu_message(user_id, None, "Меню гардероба:", reply_markup=wardrobe_menu_kb_dynamic(), typ="wardrobe_menu")
                return
            except Exception:
                pass
        sent = await bot.send_message(user_id, "Меню гардероба:", reply_markup=wardrobe_menu_kb_dynamic())
        last_menu_message[user_id] = {"chat_id": sent.chat.id, "message_id": sent.message_id, "type": "wardrobe_menu"}
        return
    if lower.startswith("/help") or "помощ" in lower:
        await cmd_help(message); return
    if lower == "/cancel":
        if user_id in pending_add:
            pending_add.pop(user_id, None)
            await send_main_menu(user_id, "Операция отменена.")
            return
        if user_id in pending_action:
            pending_action.pop(user_id, None)
            await send_main_menu(user_id, "Операция отменена.")
            return

    await send_main_menu(user_id, "Не распознал команду. Используйте меню ниже.")

# ---------------- Photo handler ----------------
@dp.message(lambda m: m.photo is not None)
async def on_photo(message: types.Message):
    user_id = message.from_user.id
    state = pending_add.get(user_id)

    photo = message.photo[-1]
    file_id = photo.file_id

    if state and state.get("stage") == "wait_photo":
        try:
            file = await bot.get_file(file_id)
            bio = io.BytesIO(); await bot.download_file(file.file_path, bio); bio.seek(0)
            pil_image = Image.open(bio).convert("RGB")
        except Exception:
            await bot.send_message(user_id, "Не удалось открыть изображение. Пришлите другое фото.")
            return

        image_input = preprocess(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(image_input); emb = emb / emb.norm(dim=-1, keepdim=True)
        emb_bytes = emb.cpu().numpy().astype(np.float32).tobytes()

        cat_logits = clip_infer_logits(image_input); cat_probs = torch.softmax(cat_logits, dim=0)
        top_idx = int(torch.argmax(cat_probs).item()); top_cat_en = CLOTHING_CATEGORIES[top_idx]; top_cat_ru = CATEGORY_MAP.get(top_cat_en, top_cat_en); top_cat_conf = float(cat_probs[top_idx].item())
        color_logits = clip_color_logits(image_input); color_probs = torch.softmax(color_logits, dim=0)
        top_color_vals = torch.topk(color_probs, k=1); top_color_en = COLOR_LABELS[int(top_color_vals.indices[0])]
        top_color_ru = COLOR_MAP.get(top_color_en, top_color_en); top_color_conf = float(top_color_vals.values[0])

        state.update({
            "stage": "ready_to_confirm",
            "file_id": file_id,
            "emb_bytes": emb_bytes,
            "suggested_category_en": top_cat_en,
            "suggested_category_ru": top_cat_ru,
            "suggested_category_conf": top_cat_conf,
            "suggested_color_en": top_color_en,
            "suggested_color_ru": top_color_ru,
            "suggested_color_conf": top_color_conf,
            "name": f"{top_cat_ru}"
        })

        try:
            sent = await bot.send_message(
                user_id,
                f"Предлагаю категорию/название: <b>{escape(state['name'])}</b> (уверенность {top_cat_conf:.0%}).\n"
                f"Предлагаю цвет: <b>{escape(top_color_ru)}</b> (уверенность {top_color_conf:.0%}).\n\n"
                "Сначала выбери название: принять или ввести вручную.",
                parse_mode="HTML",
                reply_markup=kb_name_choice()
            )
            state["suggestion_message_id"] = sent.message_id; state["suggestion_chat_id"] = sent.chat.id
        except Exception:
            await send_main_menu(user_id, "Предложение готово. Используйте меню.")
        return

    # if not in add flow -> offer actions
    offer_msg = "Вы прислали фото. Хотите добавить его в гардероб или проанализировать?"
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="➕ Добавить в гардероб", callback_data=f"offer_add:{file_id}")],
        [InlineKeyboardButton(text="🔍 Проанализировать фото", callback_data=f"offer_analyze:{file_id}")],
        [InlineKeyboardButton(text="❌ Отменить", callback_data="offer_cancel")]
    ])
    sent = await bot.send_message(user_id, offer_msg, reply_markup=kb)
    pending_photo_offer[user_id] = {"file_id": file_id, "offer_message_id": sent.message_id, "chat_id": sent.chat.id}
    try:
        await safe_delete_message(message.chat.id, message.message_id)
    except Exception:
        pass

# ---------------- Offer callbacks ----------------
@dp.callback_query(lambda c: c.data and c.data.startswith("view_saved_cap_item:"))
async def view_saved_cap_item(callback: types.CallbackQuery):
    # Разбираем данные: view_saved_cap_item:ID_ВЕЩИ:ID_КАПСУЛЫ
    try:
        parts = callback.data.split(":")
        item_id = int(parts[1])
        cap_id = int(parts[2])
    except (IndexError, ValueError):
        await callback.answer("Ошибка данных кнопки", show_alert=True)
        return

    user_id = callback.from_user.id

    # 1. Удаляем текстовое сообщение со списком вещей (чтобы не засорять чат)
    prev = last_menu_message.get(user_id)
    if prev and prev.get("type") == "capsule_view":
        try:
            await safe_delete_message(prev.get("chat_id"), prev.get("message_id"))
        except Exception:
            pass
        last_menu_message.pop(user_id, None)

    # 2. Грузим вещь
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT file_id, name, color_ru, category_ru, description FROM wardrobe WHERE id=$1 AND user_id=$2",
            item_id, user_id
        )
        tags = await conn.fetch("SELECT tag FROM tags WHERE item_id=$1", item_id)

    if not row:
        await callback.answer("Вещь не найдена.", show_alert=True)
        # Если вещи нет, пробуем вернуть в капсулу
        back_cb = types.CallbackQuery(id=callback.id, from_user=callback.from_user, message=callback.message,
                                      data=f"view_capsule:{cap_id}")
        await general_callback_router(back_cb)
        return

    # 3. Формируем карточку
    caption = f"<b>{escape(row['name'] or '-')}</b>\n" \
              f"Цвет: {escape(row['color_ru'] or '-')}\n" \
              f"Категория: {escape(row['category_ru'] or '-')}"

    if row['description']:
        caption += f"\nОписание: {escape(row['description'])}"
    if tags:
        caption += f"\nТеги: {escape(', '.join(t['tag'] for t in tags))}"

    # 4. Кнопка НАЗАД ведет обратно в view_capsule:{cap_id}
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="↩️ Назад в капсулу", callback_data=f"view_capsule:{cap_id}")]
    ])

    # 5. Отправляем фото
    sent = await bot.send_photo(user_id, photo=row['file_id'], caption=caption, parse_mode="HTML", reply_markup=kb)

    # Запоминаем это сообщение, чтобы потом его можно было удалить при выходе
    last_menu_message[user_id] = {"chat_id": sent.chat.id, "message_id": sent.message_id, "type": "item_view_saved"}
    await callback.answer()


@dp.callback_query(lambda c: c.data and c.data.startswith("ask_del_cap:"))
async def ask_delete_capsule(callback: types.CallbackQuery):
    cap_id = int(callback.data.split(":", 1)[1])

    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🗑 Да, удалить", callback_data=f"delete_capsule_confirm:{cap_id}")],
        [InlineKeyboardButton(text="Нет, оставить", callback_data=f"view_capsule:{cap_id}")]
    ])

    # Редактируем текущее сообщение с капсулой на вопрос о подтверждении
    await replace_menu_message(
        callback.from_user.id,
        callback.message,
        "<b>Вы уверены, что хотите удалить эту капсулу?</b>",
        reply_markup=kb,
        typ="capsule_delete_confirm"
    )
    await callback.answer()
@dp.callback_query(lambda c: c.data and c.data.startswith("offer_"))
async def offer_callbacks(callback: types.CallbackQuery):
    data = callback.data; user_id = callback.from_user.id
    if data.startswith("offer_add:"):
        file_id = data.split(":",1)[1]
        pending_add[user_id] = {"stage":"wait_photo"}
        try:
            file = await bot.get_file(file_id)
            bio = io.BytesIO(); await bot.download_file(file.file_path, bio); bio.seek(0)
            pil_image = Image.open(bio).convert("RGB")
            image_input = preprocess(pil_image).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(image_input); emb = emb / emb.norm(dim=-1, keepdim=True)
            emb_bytes = emb.cpu().numpy().astype(np.float32).tobytes()
            cat_logits = clip_infer_logits(image_input); cat_probs = torch.softmax(cat_logits, dim=0)
            top_idx = int(torch.argmax(cat_probs).item()); top_cat_en = CLOTHING_CATEGORIES[top_idx]; top_cat_ru = CATEGORY_MAP.get(top_cat_en, top_cat_en); top_cat_conf = float(cat_probs[top_idx].item())
            color_logits = clip_color_logits(image_input); color_probs = torch.softmax(color_logits, dim=0)
            top_color_vals = torch.topk(color_probs, k=1); top_color_en = COLOR_LABELS[int(top_color_vals.indices[0])]
            top_color_ru = COLOR_MAP.get(top_color_en, top_color_en); top_color_conf = float(top_color_vals.values[0])

            entry = pending_add[user_id]
            entry.update({
                "stage":"ready_to_confirm",
                "file_id": file_id,
                "emb_bytes": emb_bytes,
                "suggested_category_en": top_cat_en,
                "suggested_category_ru": top_cat_ru,
                "suggested_category_conf": top_cat_conf,
                "suggested_color_en": top_color_en,
                "suggested_color_ru": top_color_ru,
                "suggested_color_conf": top_color_conf,
                "name": f"{top_cat_ru}"
            })
            offer = pending_photo_offer.pop(user_id, None)
            if offer:
                try: await safe_delete_message(offer["chat_id"], offer["offer_message_id"])
                except Exception: pass

            sent = await bot.send_message(user_id,
                f"Добавляем в гардероб. Предлагаю категорию/название: <b>{escape(entry['name'])}</b> (уверенность {top_cat_conf:.0%}).\n"
                f"Предлагаю цвет: <b>{escape(top_color_ru)}</b> (уверенность {top_color_conf:.0%}).\n\n"
                "Сначала выбери название: принять или ввести вручную.",
                parse_mode="HTML",
                reply_markup=kb_name_choice()
            )
            entry["suggestion_message_id"] = sent.message_id; entry["suggestion_chat_id"] = sent.chat.id
        except Exception:
            await bot.send_message(user_id, "Не удалось обработать фото для добавления.")
        await callback.answer(); return

    if data.startswith("offer_analyze:"):
        file_id = data.split(":",1)[1]
        try:
            file = await bot.get_file(file_id)
            bio = io.BytesIO(); await bot.download_file(file.file_path, bio); bio.seek(0)
            pil_image = Image.open(bio).convert("RGB")
            image_input = preprocess(pil_image).unsqueeze(0).to(device)
            cat_logits = clip_infer_logits(image_input); cat_probs = torch.softmax(cat_logits, dim=0)
            top_idx = int(torch.argmax(cat_probs).item()); top_cat_en = CLOTHING_CATEGORIES[top_idx]; top_cat_ru = CATEGORY_MAP.get(top_cat_en, top_cat_en); top_cat_conf = float(cat_probs[top_idx].item())
            color_logits = clip_color_logits(image_input); color_probs = torch.softmax(color_logits, dim=0)
            top_color_vals = torch.topk(color_probs, k=3); top_colors = [(COLOR_LABELS[int(i)], float(p)) for i, p in zip(top_color_vals.indices, top_color_vals.values)]
            colors_str = ", ".join([f"{COLOR_MAP.get(name, name)} ({p:.0%})" for name, p in top_colors])
            offer = pending_photo_offer.pop(user_id, None)
            if offer:
                try: await safe_delete_message(offer["chat_id"], offer["offer_message_id"])
                except Exception: pass
            await bot.send_message(user_id, f"Я думаю, это: <b>{escape(top_cat_ru)}</b> (уверенность {top_cat_conf:.0%}).\nЦвета: {escape(colors_str)}.", parse_mode="HTML", reply_markup=feedback_kb())
        except Exception:
            await bot.send_message(user_id, "Не удалось проанализировать фото.")
        await callback.answer(); return

    if data == "offer_cancel":
        offer = pending_photo_offer.pop(user_id, None)
        if offer:
            try: await safe_delete_message(offer["chat_id"], offer["offer_message_id"])
            except Exception: pass
        await callback.answer("Отменено"); return

    await callback.answer()

# ---------------- Wardrobe menu and viewing ----------------
@dp.callback_query(lambda c: c.data == "menu_wardrobe")
async def menu_wardrobe(callback: types.CallbackQuery):
    user_id = callback.from_user.id

    # CHANGED: гарантированно очищаем старое меню (если оно отличается), затем используем replace_menu_message
    await clear_last_menu_if_different(user_id, callback.message)
    try:
        await replace_menu_message(user_id, callback.message, "Меню гардероба:", reply_markup=wardrobe_menu_kb_dynamic(), typ="wardrobe_menu")
    except Exception:
        # fallback
        sent = await bot.send_message(user_id, "Меню гардероба:", reply_markup=wardrobe_menu_kb_dynamic())
        last_menu_message[user_id] = {"chat_id": sent.chat.id, "message_id": sent.message_id, "type": "wardrobe_menu"}
    await callback.answer()

@dp.callback_query(lambda c: c.data == "wardrobe_add_item")
async def wardrobe_add_item(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    pending_add[user_id] = {"stage": "wait_photo"}
    try:
        await bot.edit_message_reply_markup(chat_id=callback.message.chat.id, message_id=callback.message.message_id, reply_markup=None)
    except Exception:
        pass
    # CHANGED: удаляем предыдущее сохранённое меню, чтобы не оставалось дублей
    await clear_last_menu_if_different(user_id, callback.message)
    await bot.send_message(user_id, "Пришлите фото вещи, чтобы добавить.", reply_markup=None)
    await callback.answer()

@dp.callback_query(lambda c: c.data == "wardrobe_search")
async def wardrobe_search(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    pending_add[user_id] = {"stage":"wait_search_text"}
    try:
        await bot.edit_message_reply_markup(chat_id=callback.message.chat.id, message_id=callback.message.message_id, reply_markup=None)
    except Exception:
        pass
    # CHANGED: удаляем предыдущее сохранённое меню, чтобы не оставалось дублей
    await clear_last_menu_if_different(user_id, callback.message)
    await bot.send_message(user_id, "Введи текст для поиска (название, цвет, тег, описание).", reply_markup=None)
    await callback.answer()


@dp.callback_query(lambda c: c.data and c.data.startswith("wardrobe_group:"))
async def wardrobe_group_callback(callback: types.CallbackQuery):
    # Получаем название группы (tops, shoes и т.д.)
    group_id = callback.data.split(":", 1)[1]
    user_id = callback.from_user.id

    # Если нажали "Все вещи", сбрасываем фильтр
    target_group = group_id if group_id != "all" else None

    await callback.answer()
    # Вызываем функцию показа списка с фильтром по группе
    await show_wardrobe_list(callback.message, user_id, page=0, group=target_group)


async def show_wardrobe_list(origin_message: Optional[types.Message], user_id: int, page: int = 0,
                             page_size: int = PAGE_SIZE, group: Optional[str] = None):
    offset = page * page_size
    async with db_pool.acquire() as conn:
        # Если выбрана конкретная группа и в ней есть список категорий
        if group and group in CATEGORY_GROUPS and CATEGORY_GROUPS[group]["items"]:
            items = CATEGORY_GROUPS[group]["items"]
            rows = await conn.fetch(
                "SELECT id, name, color_ru, category_ru FROM wardrobe WHERE user_id=$1 AND category_en = ANY($2::text[]) ORDER BY created_at DESC LIMIT $3 OFFSET $4",
                user_id, items, page_size, offset
            )
            total = await conn.fetchval(
                "SELECT COUNT(*) FROM wardrobe WHERE user_id=$1 AND category_en = ANY($2::text[])", user_id, items)
            title = CATEGORY_GROUPS[group]["label"]
        else:
            # Иначе показываем всё
            rows = await conn.fetch(
                "SELECT id, name, color_ru, category_ru FROM wardrobe WHERE user_id=$1 ORDER BY created_at DESC LIMIT $2 OFFSET $3",
                user_id, page_size, offset
            )
            total = await conn.fetchval("SELECT COUNT(*) FROM wardrobe WHERE user_id=$1", user_id)
            title = "Все вещи"
            group = None  # Сбрасываем group если он был некорректным или "all"

    if not rows and page == 0:
        msg_text = f"В категории «{title}» пока пусто." if group else "Твой гардероб пока пуст — добавь вещи через «Добавить вещь»."
        try:
            # Используем replace_menu_message для консистентности (нужно убедиться что импорт есть или использовать логику ниже)
            await replace_menu_message(user_id, origin_message, msg_text, reply_markup=wardrobe_menu_kb_dynamic(),
                                       typ="wardrobe_empty")
        except Exception:
            await bot.send_message(user_id, msg_text, reply_markup=wardrobe_menu_kb_dynamic())
        return

    inline_rows = []
    for rec in rows:
        item_id = rec['id'];
        name = rec['name'] or '-';
        color = rec['color_ru'] or ''
        text = f"{name} — {color}"
        inline_rows.append([InlineKeyboardButton(text=text, callback_data=f"view_item:{item_id}")])

    inline_rows.append([InlineKeyboardButton(text="↩️ Назад в меню", callback_data="menu_wardrobe")])

    # --- ЛОГИКА ПАГИНАЦИИ (ИСПРАВЛЕННАЯ) ---
    nav_buttons = []
    # Добавляем group в callback_data, если он есть. Формат: wardrobe_page:PAGE:GROUP
    group_suffix = f":{group}" if group else ""

    if page > 0:
        nav_buttons.append(
            InlineKeyboardButton(text="◀️ Назад", callback_data=f"wardrobe_page:{page - 1}{group_suffix}"))
    if (page + 1) * page_size < (total or 0):
        nav_buttons.append(
            InlineKeyboardButton(text="▶️ Вперед", callback_data=f"wardrobe_page:{page + 1}{group_suffix}"))

    if nav_buttons:
        inline_rows.append(nav_buttons)

    kb = InlineKeyboardMarkup(inline_keyboard=inline_rows)

    await replace_menu_message(user_id, origin_message, f"{title} — страница {page + 1}:", reply_markup=kb,
                               typ="wardrobe_list")


@dp.callback_query(lambda c: c.data and c.data.startswith("wardrobe_page:"))
async def wardrobe_page_callback(callback: types.CallbackQuery):
    parts = callback.data.split(":")
    page = int(parts[1])
    group = parts[2] if len(parts) > 2 else None

    user_id = callback.from_user.id
    await callback.answer()
    await show_wardrobe_list(callback.message or callback.from_user, user_id, page=page, group=group)

# ---------------- View item handlers (unchanged, but last_menu_message tracking left intact) ----------------
@dp.callback_query(lambda c: c.data and c.data.startswith("view_item:"))
async def view_item_callback(callback: types.CallbackQuery):
    item_id = int(callback.data.split(":", 1)[1])
    user_id = callback.from_user.id

    # 1) Попытка удалить предыдущее меню / карточку, чтобы не засорять чат
    prev = last_menu_message.get(user_id)
    if prev:
        try:
            await safe_delete_message(prev.get("chat_id"), prev.get("message_id"))
        except Exception:
            # если удалить нельзя — попробуем убрать клавиатуру (fallback)
            try:
                await bot.edit_message_reply_markup(chat_id=prev.get("chat_id"), message_id=prev.get("message_id"), reply_markup=None)
            except Exception:
                pass
        # убираем запись, чтобы следующий экран не пытался удалить уже удалённое сообщение
        last_menu_message.pop(user_id, None)

    # 2) Получаем данные вещи из БД
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT file_id, name, color_ru, category_ru, created_at, description FROM wardrobe WHERE id=$1 AND user_id=$2",
            item_id, user_id
        )
        if not row:
            await callback.answer("Предмет не найден или у вас нет прав.", show_alert=True)
            return
        tags = await conn.fetch("SELECT id, tag FROM tags WHERE item_id=$1 ORDER BY id", item_id)

    file_id = row['file_id']; name = row['name'] or '-'; color_ru = row['color_ru'] or '-'; category_ru = row['category_ru'] or '-'
    created_at = row['created_at']; description = row['description'] or ''
    caption_lines = [
        f"<b>{escape(name)}</b>",
        f"Цвет: {escape(color_ru)}",
        f"Категория: {escape(category_ru)}",
        f"Добавлено: {escape(format_dt(created_at))}"
    ]
    if description:
        caption_lines.append(f"\nОписание: {escape(description)}")
    if tags:
        tag_texts = ", ".join(t['tag'] for t in tags)
        caption_lines.append(f"\nТеги: {escape(tag_texts)}")
    caption = "\n".join(caption_lines)

    kb_rows = [
        [InlineKeyboardButton(text="Добавить тег ➕", callback_data=f"add_tag:{item_id}"),
         InlineKeyboardButton(text="Добавить описание ✍️", callback_data=f"add_desc:{item_id}")],
        [InlineKeyboardButton(text="Удалить вещь ❌", callback_data=f"delete_item:{item_id}")],
        [InlineKeyboardButton(text="Назад к списку ↩️", callback_data="menu_wardrobe")]  # удобная кнопка назад
    ]
    for t in tags:
        kb_rows.append([InlineKeyboardButton(text=f"❌ {t['tag']}", callback_data=f"delete_tag:{t['id']}")])
    kb = InlineKeyboardMarkup(inline_keyboard=kb_rows)

    # 3) Отправляем карточку вещи и сохраняем её как last_menu_message
    try:
        sent = await bot.send_photo(user_id, photo=file_id, caption=caption, parse_mode="HTML", reply_markup=kb)
    except Exception:
        sent = await bot.send_message(user_id, caption, parse_mode="HTML", reply_markup=kb)

    last_menu_message[user_id] = {"chat_id": sent.chat.id, "message_id": sent.message_id, "type": "item_view"}
    await callback.answer()

@dp.callback_query(lambda c: c.data and c.data.startswith("view_item_from_capsule:"))
async def view_item_from_capsule(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    lm = last_menu_message.get(user_id)
    if lm and lm.get("type") in ("item_view", "item_from_cap"):
        try:
            await safe_delete_message(lm["chat_id"], lm["message_id"])
        except Exception:
            pass
        last_menu_message.pop(user_id, None)
    item_id = int(callback.data.split(":", 1)[1]); user_id = callback.from_user.id
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT file_id, name, color_ru, category_ru, created_at, description FROM wardrobe WHERE id=$1", item_id)
        if not row:
            await callback.answer("Предмет не найден.", show_alert=True); return
        tags = await conn.fetch("SELECT id, tag FROM tags WHERE item_id=$1 ORDER BY id", item_id)

    file_id = row['file_id']; name = row['name'] or '-'; color_ru = row['color_ru'] or '-'; category_ru = row['category_ru'] or '-'
    created_at = row['created_at']; description = row['description'] or ''
    caption_lines = [f"<b>{escape(name)}</b>", f"Цвет: {escape(color_ru)}", f"Категория: {escape(category_ru)}", f"Добавлено: {escape(format_dt(created_at))}"]
    if description: caption_lines.append(f"\nОписание: {escape(description)}")
    if tags:
        tag_texts = ", ".join(t['tag'] for t in tags); caption_lines.append(f"\nТеги: {escape(tag_texts)}")
    caption = "\n".join(caption_lines)

    kb_rows = [
        [InlineKeyboardButton(text="Добавить тег ➕", callback_data=f"add_tag:{item_id}"),
         InlineKeyboardButton(text="Добавить описание ✍️", callback_data=f"add_desc:{item_id}")],
        [InlineKeyboardButton(text="Удалить вещь ❌", callback_data=f"delete_item:{item_id}")],
        [InlineKeyboardButton(text="↩️ Вернуться в капсулу", callback_data="back_to_capsule")],
    ]
    for t in tags:
        kb_rows.append([InlineKeyboardButton(text=f"❌ {t['tag']}", callback_data=f"delete_tag:{t['id']}")])
    kb_rows.append([InlineKeyboardButton(text="Закрыть", callback_data="close_view")])
    kb = InlineKeyboardMarkup(inline_keyboard=kb_rows)

    try:
        sent = await bot.send_photo(user_id, photo=file_id, caption=caption, parse_mode="HTML", reply_markup=kb)
        last_menu_message[user_id] = {"chat_id": sent.chat.id, "message_id": sent.message_id, "type": "item_from_cap"}
    except Exception:
        sent = await bot.send_message(user_id, caption, parse_mode="HTML", reply_markup=kb)
        last_menu_message[user_id] = {"chat_id": sent.chat.id, "message_id": sent.message_id, "type": "item_from_cap"}
    await callback.answer()

@dp.callback_query(lambda c: c.data == "back_to_capsule")
async def back_to_capsule(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    cap = pending_capsule.get(user_id)
    await callback.answer()
    if not cap:
        await send_main_menu(user_id, "Капсула недоступна — сгенерирую новую.")
        await send_capsule(user_id); return

    kb_rows = []
    # по 2 вещи в ряд
    kb_rows.extend(two_buttons_from_items(cap.get("items", []), lambda it: f"view_item_from_capsule:{it.get('id')}"))

    kb_rows.append([InlineKeyboardButton(text="💾 Сохранить капсулу", callback_data="save_capsule"),
                    InlineKeyboardButton(text="🔁 Сгенерировать ещё", callback_data="generate_capsule")])
    kb_rows.append([InlineKeyboardButton(text="❌ Закрыть", callback_data="close_capsule")])
    kb = InlineKeyboardMarkup(inline_keyboard=kb_rows)

    try:
        await bot.edit_message_text(cap["text"], chat_id=cap["chat_id"], message_id=cap["message_id"], parse_mode="HTML", reply_markup=kb)
        last_menu_message[user_id] = {"chat_id": cap["chat_id"], "message_id": cap["message_id"], "type": "capsule"}
    except Exception:
        sent = await bot.send_message(user_id, cap["text"], parse_mode="HTML", reply_markup=kb)
        pending_capsule[user_id].update({"chat_id": sent.chat.id, "message_id": sent.message_id})
        last_menu_message[user_id] = {"chat_id": sent.chat.id, "message_id": sent.message_id, "type": "capsule"}

    # удаляем текущую карточку (если есть) чтобы не засорять
    lm = last_menu_message.get(user_id)
    if lm and lm.get("type") == "item_from_cap":
        try: await safe_delete_message(lm["chat_id"], lm["message_id"])
        except Exception: pass
    await callback.answer()

@dp.callback_query(lambda c: c.data == "close_capsule")
async def close_capsule(callback: types.CallbackQuery):
    user_id = callback.from_user.id

    # Забираем и удаляем капсулу из памяти
    cap = pending_capsule.pop(user_id, None)

    if cap:
        # сначала постараемся удалить сообщение-капсулу (чтобы не засорять чат)
        try:
            await safe_delete_message(cap.get("chat_id"), cap.get("message_id"))
        except Exception:
            # fallback: снять клавиатуру если удаление невозможно
            try:
                await bot.edit_message_reply_markup(chat_id=cap.get("chat_id"), message_id=cap.get("message_id"), reply_markup=None)
            except Exception:
                pass

    # Снимаем клавиатуру с текущего сообщения (если это карточка, откуда вызвали)
    try:
        if callback.message:
            await bot.edit_message_reply_markup(chat_id=callback.message.chat.id, message_id=callback.message.message_id, reply_markup=None)
    except Exception:
        pass

    # Попытка переиспользовать уже существующее меню (чтобы не создавать дубликат)
    lm = last_menu_message.get(user_id)
    used_existing = False
    if lm:
        # Не редактируем если lm ссылается на только что удалённую капсулу
        if not (cap and lm.get("chat_id") == cap.get("chat_id") and lm.get("message_id") == cap.get("message_id")):
            try:
                sent = await bot.edit_message_text("Главное меню:", chat_id=lm["chat_id"], message_id=lm["message_id"], reply_markup=main_menu_kb())
                last_menu_message[user_id] = {"chat_id": sent.chat.id, "message_id": sent.message_id, "type": "start"}
                used_existing = True
            except Exception:
                # если редактирование не получилось — будем отправлять новое ниже
                used_existing = False

    if not used_existing:
        # отправляем одно новое меню и сохраняем его
        try:
            sent = await bot.send_message(user_id, "Главное меню:", reply_markup=main_menu_kb())
            last_menu_message[user_id] = {"chat_id": sent.chat.id, "message_id": sent.message_id, "type": "start"}
        except Exception:
            # тихий fail-safe — ничего не делаем, но не дублируем
            pass

    await callback.answer("Капсула закрыта.")

# ---------------- Add flows callbacks (название/цвет/сохранить/отмена) ----------------
@dp.callback_query(lambda c: c.data is not None and c.data in {
    "add_accept_name", "add_enter_name", "add_accept_color", "add_enter_color", "add_save", "add_cancel"
})
async def add_flow_callbacks(callback: types.CallbackQuery):
    data = callback.data; user_id = callback.from_user.id
    state = pending_add.get(user_id)

    # Принять предложенное имя
    if data == "add_accept_name" and state and state.get("stage") in ("ready_to_confirm",):
        try:
            await bot.edit_message_text(
                f"Название установлено: <b>{escape(state.get('name',''))}</b>.\nВыберите цвет:",
                chat_id=state.get("suggestion_chat_id"),
                message_id=state.get("suggestion_message_id"),
                parse_mode="HTML",
                reply_markup=kb_color_choice()
            )
        except Exception:
            await bot.send_message(user_id, f"Название установлено: <b>{escape(state.get('name',''))}</b>.\nВыберите цвет:", parse_mode="HTML", reply_markup=kb_color_choice())
        await callback.answer("Название принято")
        return

    # Ввести имя вручную
    if data == "add_enter_name" and state and state.get("stage") in ("ready_to_confirm",):
        state["stage"] = "awaiting_name"
        try:
            await bot.edit_message_text("Хорошо — введите название вещи текстом.", chat_id=state.get("suggestion_chat_id"), message_id=state.get("suggestion_message_id"))
        except Exception:
            await bot.send_message(user_id, "Хорошо — введите название вещи текстом.")
        await callback.answer()
        return

    # Принять предложенный цвет
    if data == "add_accept_color" and state and state.get("stage") in ("ready_to_confirm",):
        state["color_en"] = state.get("suggested_color_en"); state["color_ru"] = state.get("suggested_color_ru"); state["stage"] = "ready_to_confirm"
        try:
            await bot.edit_message_text(
                f"Название: <b>{escape(state.get('name',''))}</b>\nЦвет: <b>{escape(state.get('color_ru',''))}</b>\n\nГотово к сохранению.",
                chat_id=state.get("suggestion_chat_id"),
                message_id=state.get("suggestion_message_id"),
                parse_mode="HTML",
                reply_markup=kb_final_choice()
            )
        except Exception:
            await bot.send_message(user_id, f"Название: <b>{escape(state.get('name',''))}</b>\nЦвет: <b>{escape(state.get('color_ru',''))}</b>\n\nГотово к сохранению.", parse_mode="HTML", reply_markup=kb_final_choice())
        await callback.answer("Цвет принят")
        return

    # Ввести цвет вручную
    if data == "add_enter_color" and state and state.get("stage") in ("ready_to_confirm",):
        state["stage"] = "awaiting_color"
        try:
            await bot.edit_message_text("Хорошо — введите цвет вещи текстом.", chat_id=state.get("suggestion_chat_id"), message_id=state.get("suggestion_message_id"))
        except Exception:
            await bot.send_message(user_id, "Хорошо — введите цвет вещи текстом.")
        await callback.answer()
        return

    # Сохранить вещь в БД
    if data == "add_save" and state and state.get("stage") in ("ready_to_confirm",):
        file_id = state.get("file_id"); emb_bytes = state.get("emb_bytes"); name = state.get("name","")
        color_en = state.get("color_en","") or ""; color_ru = state.get("color_ru","") or ""
        category_en = state.get("suggested_category_en","") or ""; category_ru = state.get("suggested_category_ru","") or ""
        created_at = datetime.now(timezone.utc)
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO wardrobe (user_id, file_id, emb, name, color_en, color_ru, category_en, category_ru, created_at, description)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, user_id, file_id, emb_bytes, name, color_en, color_ru, category_en, category_ru, created_at, "")
        try:
            await safe_delete_message(state.get("suggestion_chat_id"), state.get("suggestion_message_id"))
        except Exception:
            pass
        pending_add.pop(user_id, None)
        await send_main_menu(user_id, f"Вещь <b>{escape(name)}</b> добавлена в гардероб ✅")
        await callback.answer("Вещь сохранена ✅")
        return

    # Отмена
    if data == "add_cancel" and state:
        try:
            await safe_delete_message(state.get("suggestion_chat_id"), state.get("suggestion_message_id"))
        except Exception:
            pass
        pending_add.pop(user_id, None)
        await send_main_menu(user_id, "Операция добавления отменена.")
        await callback.answer("Добавление отменено")
        return

    # safety fallback
    await callback.answer()

# ---------------- Handlers for add_tag/add_desc/delete_tag/delete_item/close_view ----------------
@dp.callback_query(lambda c: c.data and c.data.startswith("add_tag:"))
async def add_tag_request(callback: types.CallbackQuery):
    item_id = int(callback.data.split(":",1)[1]); user_id = callback.from_user.id
    async with db_pool.acquire() as conn:
        has = await conn.fetchval("SELECT 1 FROM wardrobe WHERE id=$1 AND user_id=$2", item_id, user_id)
    if not has:
        await callback.answer("Нет прав или вещь не найдена.", show_alert=True); return
    try:
        if callback.message:
            await bot.edit_message_reply_markup(chat_id=callback.message.chat.id, message_id=callback.message.message_id, reply_markup=None)
    except Exception:
        pass
    pending_action[user_id] = {"action":"add_tag", "item_id": item_id}
    await bot.send_message(user_id, "Введите тег для этой вещи (одно слово или фраза). Для отмены /cancel")
    await callback.answer()

@dp.callback_query(lambda c: c.data and c.data.startswith("add_desc:"))
async def add_desc_request(callback: types.CallbackQuery):
    item_id = int(callback.data.split(":",1)[1]); user_id = callback.from_user.id
    async with db_pool.acquire() as conn:
        has = await conn.fetchval("SELECT 1 FROM wardrobe WHERE id=$1 AND user_id=$2", item_id, user_id)
    if not has:
        await callback.answer("Нет прав или вещь не найдена.", show_alert=True); return
    try:
        if callback.message:
            await bot.edit_message_reply_markup(chat_id=callback.message.chat.id, message_id=callback.message.message_id, reply_markup=None)
    except Exception:
        pass
    pending_action[user_id] = {"action":"add_desc", "item_id": item_id}
    await bot.send_message(user_id, "Введите описание для этой вещи. Для отмены /cancel")
    await callback.answer()

@dp.callback_query(lambda c: c.data and c.data.startswith("delete_tag:"))
async def delete_tag_callback(callback: types.CallbackQuery):
    tag_id = int(callback.data.split(":",1)[1]); user_id = callback.from_user.id
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT item_id, user_id, tag FROM tags WHERE id=$1", tag_id)
        if not row:
            await callback.answer("Тег не найден.", show_alert=True); return
        if row['user_id'] != user_id:
            await callback.answer("Нет прав удалять этот тег.", show_alert=True); return
        await conn.execute("DELETE FROM tags WHERE id=$1", tag_id)
        item_id = row['item_id']

    # обновляем карточку (если возможно)
    try:
        async with db_pool.acquire() as conn:
            r = await conn.fetchrow("SELECT file_id, name, color_ru, category_ru, created_at, description FROM wardrobe WHERE id=$1 AND user_id=$2", item_id, user_id)
            tags = await conn.fetch("SELECT id, tag FROM tags WHERE item_id=$1 ORDER BY id", item_id)
        if not r:
            await send_main_menu(user_id, "Вещь не найдена (после удаления тега)."); await callback.answer("Тег удалён."); return
        file_id = r['file_id']; name = r['name'] or '-'; color_ru = r['color_ru'] or '-'; category_ru = r['category_ru'] or '-'
        created_at = r['created_at']; description = r['description'] or ''
        caption_lines = [f"<b>{escape(name)}</b>", f"Цвет: {escape(color_ru)}", f"Категория: {escape(category_ru)}", f"Добавлено: {escape(format_dt(created_at))}"]
        if description: caption_lines.append(f"\nОписание: {escape(description)}")
        if tags: tag_texts = ", ".join(t['tag'] for t in tags); caption_lines.append(f"\nТеги: {escape(tag_texts)}")
        caption = "\n".join(caption_lines)

        kb_rows = [
            [InlineKeyboardButton(text="Добавить тег ➕", callback_data=f"add_tag:{item_id}"),
             InlineKeyboardButton(text="Добавить описание ✍️", callback_data=f"add_desc:{item_id}")],
            [InlineKeyboardButton(text="Удалить вещь ❌", callback_data=f"delete_item:{item_id}")]
        ]
        for t in tags:
            kb_rows.append([InlineKeyboardButton(text=f"❌ {t['tag']}", callback_data=f"delete_tag:{t['id']}")])
        kb_rows.append([InlineKeyboardButton(text="Закрыть", callback_data="close_view")])
        kb = InlineKeyboardMarkup(inline_keyboard=kb_rows)

        if callback.message:
            try:
                if callback.message.photo:
                    await bot.edit_message_caption(chat_id=callback.message.chat.id, message_id=callback.message.message_id, caption=caption, parse_mode="HTML", reply_markup=kb)
                else:
                    await bot.send_photo(callback.message.chat.id, photo=file_id, caption=caption, parse_mode="HTML", reply_markup=kb)
            except Exception:
                await send_main_menu(user_id, "Тег удалён.")
        else:
            await send_main_menu(user_id, "Тег удалён.")
        await callback.answer("Тег удалён.")
    except Exception:
        await callback.answer("Ошибка при удалении.", show_alert=True)
def two_buttons_from_items(items, cb_builder):
    """
    items - итерируемая коллекция объектов/dict/Record с полями 'id' и 'name'
    cb_builder - функция, принимающая элемент items и возвращающая callback_data (str)
    Возвращает list of rows (каждый ряд - list из 1 или 2 InlineKeyboardButton).
    """
    rows = []
    buf = []
    for it in items:
        try:
            label = it.get('name') if isinstance(it, dict) else (it['name'] if 'name' in it else None)
        except Exception:
            # объекты типа asyncpg.Record поддерживают индексацию по ключу
            label = getattr(it, 'name', None) or (it['name'] if hasattr(it, '__contains__') and 'name' in it else None)
        label = label or "(без названия)"
        btn = InlineKeyboardButton(text=label, callback_data=cb_builder(it))
        buf.append(btn)
        if len(buf) == 2:
            rows.append(buf)
            buf = []
    if buf:
        rows.append(buf)
    return rows

@dp.callback_query(lambda c: c.data and c.data.startswith("delete_item:"))
async def delete_item_request(callback: types.CallbackQuery):
    item_id = int(callback.data.split(":",1)[1]); user_id = callback.from_user.id
    async with db_pool.acquire() as conn:
        name = await conn.fetchval("SELECT name FROM wardrobe WHERE id=$1 AND user_id=$2", item_id, user_id)
    if not name:
        await callback.answer("Предмет не найден или у вас нет прав.", show_alert=True); return
    confirm_kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Подтвердить удаление ❗️", callback_data=f"delete_confirm:{item_id}")],
        [InlineKeyboardButton(text="Отмена ↩️", callback_data="delete_cancel")]
    ])
    try:
        if callback.message and callback.message.photo:
            await bot.edit_message_caption(chat_id=callback.message.chat.id, message_id=callback.message.message_id, caption=f"Удаление предмета: <b>{escape(name)}</b>\nВы уверены?", parse_mode="HTML", reply_markup=confirm_kb)
        else:
            await bot.edit_message_text(chat_id=callback.message.chat.id, message_id=callback.message.message_id, text=f"Удаление предмета: <b>{escape(name)}</b>\nВы уверены?", parse_mode="HTML", reply_markup=confirm_kb)
    except Exception:
        await bot.send_message(user_id, f"Удаление предмета: <b>{escape(name)}</b>\nВы уверены?", parse_mode="HTML", reply_markup=confirm_kb)
    await callback.answer()

@dp.callback_query(lambda c: c.data and c.data.startswith("delete_confirm:"))
async def delete_item_confirm(callback: types.CallbackQuery):
    item_id = int(callback.data.split(":",1)[1]); user_id = callback.from_user.id
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT file_id, name FROM wardrobe WHERE id=$1 AND user_id=$2", item_id, user_id)
        if not row:
            await callback.answer("Уже удалено или нет прав.", show_alert=True)
            try:
                await bot.edit_message_reply_markup(callback.message.chat.id, callback.message.message_id, reply_markup=None)
            except Exception:
                pass
            return
        name = row['name']
        await conn.execute("DELETE FROM wardrobe WHERE id=$1 AND user_id=$2", item_id, user_id)
    try:
        if callback.message and callback.message.photo:
            await bot.edit_message_caption(chat_id=callback.message.chat.id, message_id=callback.message.message_id, caption=f"🗑️ Предмет <b>{escape(name)}</b> удалён.", parse_mode="HTML", reply_markup=None)
        else:
            await bot.edit_message_text(chat_id=callback.message.chat.id, message_id=callback.message.message_id, text=f"🗑️ Предмет <b>{escape(name)}</b> удалён.", parse_mode="HTML", reply_markup=None)
    except Exception:
        await bot.send_message(user_id, f"🗑️ Предмет <b>{escape(name)}</b> удалён.", parse_mode="HTML")
    await callback.answer("Предмет удалён.", show_alert=False)

@dp.callback_query(lambda c: c.data == "delete_cancel")
async def delete_cancel(callback: types.CallbackQuery):
    try:
        await bot.edit_message_reply_markup(chat_id=callback.message.chat.id, message_id=callback.message.message_id, reply_markup=None)
    except Exception:
        pass
    await callback.answer("Удаление отменено", show_alert=False)

@dp.callback_query(lambda c: c.data == "close_view")
async def close_view(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    await callback.answer()

    lm = last_menu_message.get(user_id)
    cap = pending_capsule.get(user_id)

    # 1) Если открыта вещь из капсулы — вернём пользователя в капсулу
    if lm and lm.get("type") == "item_from_cap" and cap:
        try:
            kb_rows = []
            kb_rows.extend(two_buttons_from_items(cap.get("items", []), lambda it: f"view_item_from_capsule:{it.get('id')}"))

            kb_rows.append([InlineKeyboardButton(text="💾 Сохранить капсулу", callback_data="save_capsule"),
                            InlineKeyboardButton(text="🔁 Сгенерировать ещё", callback_data="generate_capsule")])
            kb_rows.append([InlineKeyboardButton(text="❌ Закрыть", callback_data="close_capsule")])
            kb = InlineKeyboardMarkup(inline_keyboard=kb_rows)

            if cap.get("chat_id") and cap.get("message_id"):
                try:
                    await bot.edit_message_text(cap.get("text", "Капсула"), chat_id=cap["chat_id"],
                                                message_id=cap["message_id"], parse_mode="HTML", reply_markup=kb)
                    last_menu_message[user_id] = {"chat_id": cap["chat_id"], "message_id": cap["message_id"], "type": "capsule"}
                except Exception:
                    sent = await bot.send_message(user_id, cap.get("text", "Капсула"), parse_mode="HTML", reply_markup=kb)
                    pending_capsule[user_id].update({"chat_id": sent.chat.id, "message_id": sent.message_id})
                    last_menu_message[user_id] = {"chat_id": sent.chat.id, "message_id": sent.message_id, "type": "capsule"}
            else:
                sent = await bot.send_message(user_id, cap.get("text", "Капсула"), parse_mode="HTML", reply_markup=kb)
                pending_capsule[user_id].update({"chat_id": sent.chat.id, "message_id": sent.message_id})
                last_menu_message[user_id] = {"chat_id": sent.chat.id, "message_id": sent.message_id, "type": "capsule"}

            # удалить временную карточку вещи, если она была создана
            try:
                temp = cap.get("temp_item_msg")
                if temp:
                    await safe_delete_message(temp.get("chat_id"), temp.get("message_id"))
                    cap.pop("temp_item_msg", None)
            except Exception:
                pass

            return
        except Exception:
            # упадёт в общий fallback ниже
            pass

    # 2) Иначе — попробуем открыть гардероб через show_wardrobe_list (если есть)
    try:
        if 'show_wardrobe_list' in globals():
            # попытаемся восстановить страницу из last_menu_message (если она там сохранена)
            page = 0
            if lm and isinstance(lm.get("meta"), dict):
                page = int(lm["meta"].get("page", 0))
            # передаём origin_message, чтобы replace_menu_message внутри show_wardrobe_list работал корректно
            origin_msg = callback.message if getattr(callback, "message", None) else None
            await show_wardrobe_list(origin_msg, user_id, page=page)
            return
    except Exception:
        # если show_wardrobe_list упала — продолжаем в fallback
        try:
            logger = logging.getLogger("close_view")
            logger.exception("show_wardrobe_list failed in close_view")
        except Exception:
            pass

    # 3) Последний fallback — показать главное меню или отправить простое сообщение
    try:
        if 'send_main_menu' in globals():
            await send_main_menu(user_id, "Закрыто.")
        else:
            await bot.send_message(user_id, "Закрыто.")
    except Exception:
        # ничего не делаем — молча игнорируем ошибку закрытия
        pass

# ---------------- General callbacks: menu navigation, capsule save, feedback ----------------
@dp.callback_query(lambda c: c.data == "generate_capsule")
async def generate_capsule_cb(callback: types.CallbackQuery):
    await callback.answer("Генерирую новую капсулу…")
    await send_capsule(callback.from_user.id, force_regen=True)


@dp.callback_query()
async def general_callback_router(callback: types.CallbackQuery):
    data = callback.data or ""; user_id = callback.from_user.id

    # menu navigation
    if data == "menu_generate_capsule":
        await callback.answer(); await send_capsule(user_id); return
    if data == "menu_help":
        await callback.answer()
        try:
            await send_main_menu(user_id, HELP_TEXT)
        except Exception:
            try:
                sent = await bot.send_message(user_id, HELP_TEXT, parse_mode="HTML")
                last_menu_message[user_id] = {"chat_id": sent.chat.id, "message_id": sent.message_id, "type": "start"}
            except Exception:
                pass
        return
    if data == "menu_back":
        await callback.answer()

        # 1) Попытка удалить текущее сообщение (чтобы не оставлять "карточку" откуда нажали назад)
        try:
            if callback.message and callback.message.chat and callback.message.message_id:
                await safe_delete_message(callback.message.chat.id, callback.message.message_id)
        except Exception:
            # молча игнорируем ошибки удаления
            pass

        # 2) Показать главное меню, стараясь не дублировать
        # send_main_menu уже реализована так, чтобы удалять/редактировать last_menu_message — используем её.
        try:
            await send_main_menu(user_id)
        except Exception:
            # Фоллбек: просто отправляем и обновляем last_menu_message вручную
            try:
                sent = await bot.send_message(user_id, "Главное меню:", reply_markup=main_menu_kb())
                last_menu_message[user_id] = {"chat_id": sent.chat.id, "message_id": sent.message_id, "type": "start"}
            except Exception:
                pass

        return
    # view saved capsules
    # Внутри general_callback_router: замените обработку data == "menu_view_capsules" на этот блок
    if data == "menu_view_capsules":
        await callback.answer()
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, name, created_at FROM capsules WHERE user_id=$1 ORDER BY created_at DESC", user_id)

        if not rows:
            # используем replace_menu_message чтобы аккуратно показать ответ и удалить старое меню
            await replace_menu_message(user_id, callback.message, "У тебя ещё нет сохранённых капсул.",
                                       reply_markup=main_menu_kb(), typ="start")
            return

        kb_rows = [[InlineKeyboardButton(text=f"{r['name']} — {format_dt(r['created_at'])}",
                                         callback_data=f"view_capsule:{r['id']}")] for r in rows]
        kb_rows.append([InlineKeyboardButton(text="↩️ Назад", callback_data="menu_back")])
        kb = InlineKeyboardMarkup(inline_keyboard=kb_rows)

        try:
            await replace_menu_message(user_id, callback.message, "Твои капсулы:", reply_markup=kb, typ="capsule_list")
        except Exception:
            sent = await bot.send_message(user_id, "Твои капсулы:", reply_markup=kb)
            last_menu_message[user_id] = {"chat_id": sent.chat.id, "message_id": sent.message_id,
                                          "type": "capsule_list"}
        return

    if data.startswith("view_capsule:"):
        cap_id = int(data.split(":", 1)[1])

        # 1. Если мы вернулись из просмотра вещи (сообщение с фото),
        # его нельзя отредактировать в текст. Поэтому удаляем его принудительно.
        if callback.message.photo:
            try:
                await safe_delete_message(callback.message.chat.id, callback.message.message_id)
                # Обнуляем message, чтобы replace_menu_message отправил новое, а не пытался редактировать удаленное
                callback.message = None
            except Exception:
                pass
        else:
            await callback.answer()

        # Чистим запись о последнем меню, если это было что-то другое
        await clear_last_menu_if_different(user_id, callback.message)

        async with db_pool.acquire() as conn:
            cap = await conn.fetchrow(
                "SELECT id, name, item_ids, created_at FROM capsules WHERE id=$1 AND user_id=$2", cap_id, user_id)

        if not cap:
            # Если капсула удалена, кидаем в главное меню
            await replace_menu_message(user_id, callback.message, "Капсула не найдена.",
                                       reply_markup=main_menu_kb(), typ="start")
            return

        item_ids = cap['item_ids'] or []
        lines = [f"💾 <b>{escape(cap['name'])}</b> — {format_dt(cap['created_at'])}", "", "Список вещей:"]
        kb_rows = []

        if item_ids:
            async with db_pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT id, name, file_id, category_ru FROM wardrobe WHERE id = ANY($1::int[])", item_ids)

            # добавляем кнопки по 2 в ряд, без слова "Открыть:"
            kb_rows.extend(two_buttons_from_items(rows, lambda r: f"view_saved_cap_item:{r['id']}:{cap_id}"))

            lines.append("\nВыберите вещь для просмотра.")
        else:
            lines.append("В капсуле нет сохранённых вещей.")

        text = "\n".join(lines)

        kb_rows.append([InlineKeyboardButton(text="❌ Удалить капсулу", callback_data=f"ask_del_cap:{cap['id']}"),
                        InlineKeyboardButton(text="↩️ Назад к списку капсул", callback_data="menu_view_capsules")])
        kb = InlineKeyboardMarkup(inline_keyboard=kb_rows)

        await replace_menu_message(user_id, callback.message, text, reply_markup=kb, typ="capsule_view")
        return
    # Внутри general_callback_router: замените обработку delete_capsule_confirm на этот блок
    if data.startswith("delete_capsule_confirm:"):
        cap_id = int(data.split(":", 1)[1])
        async with db_pool.acquire() as conn:
            await conn.execute("DELETE FROM capsules WHERE id=$1 AND user_id=$2", cap_id, user_id)

        # Попробуем обновить текущий список капсул в том же сообщении (если вызвано из списка)
        try:
            async with db_pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT id, name, created_at FROM capsules WHERE user_id=$1 ORDER BY created_at DESC", user_id)
            if rows:
                kb_rows = [[InlineKeyboardButton(text=f"{r['name']} — {format_dt(r['created_at'])}",
                                                 callback_data=f"view_capsule:{r['id']}")] for r in rows]
                kb_rows.append([InlineKeyboardButton(text="↩️ Назад", callback_data="menu_back")])
                kb = InlineKeyboardMarkup(inline_keyboard=kb_rows)
                # если есть callback.message — редактируем её, иначе отправим новое
                if callback.message:
                    await bot.edit_message_text("Капсула удалена. Обновлённый список:",
                                                chat_id=callback.message.chat.id,
                                                message_id=callback.message.message_id, parse_mode="HTML",
                                                reply_markup=kb)
                    last_menu_message[user_id] = {"chat_id": callback.message.chat.id,
                                                  "message_id": callback.message.message_id, "type": "capsule_list"}
                else:
                    sent = await bot.send_message(user_id, "Капсула удалена. Обновлённый список:", parse_mode="HTML",
                                                  reply_markup=kb)
                    last_menu_message[user_id] = {"chat_id": sent.chat.id, "message_id": sent.message_id,
                                                  "type": "capsule_list"}
            else:
                # больше нет капсул — очищаем запись и показываем главное меню
                last_menu_message.pop(user_id, None)
                await send_main_menu(user_id, "Капсула удалена.")
        except Exception as e:
            print("delete_capsule_confirm: update failed:", e)
            try:
                last_menu_message.pop(user_id, None)
            except Exception:
                pass
            await send_main_menu(user_id, "Капсула удалена.")
        await callback.answer("Капсула удалена")
        return

    # feedback (simple)
    if data.startswith("fb_"):
        if data == "fb_yes":
            await send_main_menu(user_id, "Спасибо за подтверждение.")
            await callback.answer("Спасибо!"); return
        if data == "fb_no_retry":
            await send_main_menu(user_id, "Повторный анализ — пришлите фото заново.")
            await callback.answer(); return
        if data == "fb_no_input":
            await bot.send_message(user_id, "Введите правильную метку текстом (или /cancel)."); await callback.answer(); return

    await callback.answer()

# ---------------- Search helper ----------------
async def do_search(message: types.Message, user_id: int, query: str):
    query = (query or "").strip()
    print(f"[DEBUG] do_search called for user={user_id!r} query={query!r}")
    if not query:
        await bot.send_message(user_id, "Пустой запрос. Введите текст или /cancel чтобы выйти.", reply_markup=None)
        return

    q_orig = query
    q_norm = normalize_russian(query)
    q_short = q_norm
    if q_norm.endswith("ый"):
        q_short = q_norm[:-2]

    like_patterns = [f"%{q_orig}%", f"%{q_norm}%"]
    if q_short and q_short != q_norm:
        like_patterns.append(f"%{q_short}%")

    async with db_pool.acquire() as conn:
        params = [user_id]
        idx = 2
        where_parts = []
        for p in like_patterns:
            params.append(p)
            where_parts.append(f"w.name ILIKE ${idx} OR w.color_ru ILIKE ${idx} OR w.description ILIKE ${idx} OR t.tag ILIKE ${idx}")
            idx += 1
        where_sql = " OR ".join(where_parts)
        sql = f"""
            SELECT DISTINCT w.id, w.name, w.color_ru, w.created_at
            FROM wardrobe w
            LEFT JOIN tags t ON t.item_id = w.id
            WHERE w.user_id = $1 AND ({where_sql})
            ORDER BY w.created_at DESC
            LIMIT 200
        """
        rows = await conn.fetch(sql, *params)
        print(f"[DEBUG] do_search found {len(rows)} rows for user={user_id!r} query={query!r}")

    if not rows:
        await bot.send_message(user_id, "Ничего не найдено. Попробуйте другой запрос или /cancel чтобы выйти.", reply_markup=None)
        return

    kb_rows = []
    for rec in rows:
        name = rec['name'] or "(без названия)"; color = rec['color_ru'] or ""
        kb_rows.append([InlineKeyboardButton(text=f"{name} — {color}".strip(), callback_data=f"view_item:{rec['id']}")])
    kb = InlineKeyboardMarkup(inline_keyboard=kb_rows)

    bottom_kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔎 Новый поиск", callback_data="search_continue")],
        [InlineKeyboardButton(text="⛔ Завершить поиск", callback_data="search_end")]
    ])

    await bot.send_message(user_id, f"Найдено {len(rows)} предметов:", reply_markup=kb)
    await bot.send_message(user_id, "Чтобы сделать ещё поиск — введите новый запрос. Чтобы выйти — нажмите «Завершить поиск» или /cancel.", reply_markup=bottom_kb)

# ---------------- Startup ----------------
async def on_startup():
    global db_pool
    db_pool = await create_pool_with_retries(DATABASE_URL, attempts=5, delay=2.0)
    await init_db_and_migrate()
    try:
        await bot.set_my_commands([
            types.BotCommand("start", "Запустить бота"),
            types.BotCommand("help", "Помощь"),
            types.BotCommand("capsule", "Сгенерировать капсулу")
        ])
    except Exception:
        pass

async def main():
    await on_startup()
    print("Bot starting...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        traceback.print_exc()
