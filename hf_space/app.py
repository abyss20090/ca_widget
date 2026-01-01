import os
import re
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import gradio as gr
import httpx
from openai import OpenAI

print(">>> GRADIO VERSION:", gr.__version__)

# ============================================================
# Paths
# ============================================================
BASE_DIR = Path(__file__).parent
SOURCES_DIR = BASE_DIR / "sources"

# ============================================================
# Utils
# ============================================================
def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def _normalize_content(x: Any) -> str:
    """
    Make sure content is ALWAYS a plain string (Gradio messages format requirement).
    This prevents: [{'text': '...', 'type': 'text'}] leaking into chat history.
    """
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        # Common gradio multimodal piece
        if "text" in x and isinstance(x["text"], str):
            return x["text"]
        # If it looks like {"role":..., "content":...}
        if "content" in x:
            return _normalize_content(x["content"])
        return str(x)
    if isinstance(x, list):
        # join all parts
        parts = [_normalize_content(p) for p in x]
        parts = [p for p in parts if p.strip()]
        return "\n".join(parts).strip()
    return str(x)

# ============================================================
# Load sources (RAG) - store lowercase for fast search
# ============================================================
def _load_sources() -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    if not SOURCES_DIR.exists():
        print("WARNING: sources/ folder not found")
        return docs

    for path in SOURCES_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".md", ".html", ".htm"}:
            continue

        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print("Read error:", path, e)
            continue

        if "<html" in raw.lower() or "<body" in raw.lower() or "<p" in raw.lower():
            plain = _strip_html(raw)
        else:
            plain = re.sub(r"\s+", " ", raw).strip()

        if not plain:
            continue

        snippet = plain[:4000]
        docs.append(
            {
                "path": str(path.relative_to(BASE_DIR)),
                "text": snippet,
                "lower": snippet.lower(),
            }
        )

    print(f"Loaded {len(docs)} source documents.")
    return docs

DOCS = _load_sources()

def _retrieve(message: str, top_k: int = 4) -> List[Dict[str, str]]:
    if not DOCS:
        return []

    q = (message or "").lower().strip()
    if not q:
        return []

    # tokenization
    tokens = [w for w in re.split(r"[^a-z0-9]+", q) if len(w) > 2]

    # add IB expansions
    extra_keywords = []
    if "ib" in q:
        extra_keywords += ["international baccalaureate", "ib programme", "ib program"]
    keywords = tokens + extra_keywords

    scores: List[tuple[int, Dict[str, str]]] = []
    for doc in DOCS:
        text = doc["lower"]
        score = 0
        for kw in keywords:
            if kw and kw in text:
                score += 1
        if score > 0:
            scores.append((score, doc))

    if not scores:
        return []

    scores.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scores[:top_k]]

# ============================================================
# Guardrails (FAST, NO OpenAI calls)
# ============================================================
CA_ENTITY_KEYWORDS = [
    "cheshire", "cheshire academy", "cheshireacademy", "cheshireacademy.org",
    "切舍尔", "切舍尔中学", "柴郡", "柴郡学院", "柴俊学院",
    "我们学校", "我学校", "本校",
]

# broad school topics (helps when user doesn't type "Cheshire")
CA_TOPIC_KEYWORDS = [
    # admissions/academics/campus
    "admission", "apply", "application", "deadline", "tuition", "fees", "scholarship",
    "boarding", "dorm", "campus", "calendar", "forms",
    "ib", "international baccalaureate", "class", "course", "academics",
    "financial aid", "visit", "open house",
    # athletics/news
    "athletics", "sports", "basketball", "soccer", "baseball", "football", "hockey",
    "tennis", "golf", "track", "match", "game", "score", "tournament", "championship",
    "news", "update", "result",
    # Chinese
    "招生", "申请", "报名", "截止", "截止日期", "学费", "费用", "奖学金", "助学金",
    "寄宿", "宿舍", "校园", "日历", "表格", "课程", "学术", "访校", "开放日",
    "体育", "运动", "篮球", "足球", "棒球", "网球", "田径", "冠军", "比赛", "比分", "战报", "赛事", "新闻",
]

SENSITIVE_SCHOOL_CASE_KEYWORDS = [
    "lawsuit", "litigation", "sued", "court", "settlement", "complaint",
    "case", "allegation", "scandal",
    "诉讼", "起诉", "官司", "法院", "和解", "案件", "丑闻", "指控",
]

ADULT_CONTENT_KEYWORDS = [
    "porn", "xxx", "nude", "sex", "hookup", "onlyfans",
    "色情", "黄片", "裸照", "成人视频", "约炮", "做爰",
]

SELF_HARM_KEYWORDS = [
    "suicide", "kill myself", "self-harm", "cut myself", "end my life",
    "自杀", "想死", "自残", "割腕", "结束生命",
]

def _contains_any(text: str, keywords: List[str]) -> bool:
    t = (text or "").lower()
    return any(k.lower() in t for k in keywords if k)

def _guess_lang_code_from_message(message: str) -> str:
    s = message or ""
    if re.search(r"[\u4e00-\u9fff]", s):
        return "zh"
    return "en"

REFUSAL_TEMPLATES = {
    "non_ca": {
        "en": (
            "I can only answer questions about **Cheshire Academy** (academics, admissions, campus life, athletics, forms, calendars, etc.).\n\n"
            "Please re-ask your question **about Cheshire Academy**."
        ),
        "zh": (
            "我只能回答 **Cheshire Academy（切舍尔中学）** 相关的问题（学术、招生、校园生活、体育、表格、日历等）。\n\n"
            "请你把问题改成 **与切舍尔中学有关** 的问题再问一遍。"
        ),
    },
    "sensitive_school": {
        "en": (
            "I’m here to help with **Cheshire Academy admissions and campus information**. "
            "This topic isn’t appropriate to discuss here (e.g., legal cases/allegations).\n\n"
            "Please ask a different question about **programs, admissions, tuition, deadlines, boarding, campus life,** etc."
        ),
        "zh": (
            "我可以帮助回答 **切舍尔中学的招生与校园信息**。但这个问题涉及不适合在这里讨论的内容（例如法律纠纷/指控/具体案件）。\n\n"
            "请你换一个关于 **课程项目、招生申请、学费、截止日期、寄宿与校园生活** 等方面的问题再问一遍。"
        ),
    },
    "adult": {
        "en": "I can’t help with that. Please ask an appropriate question about **Cheshire Academy**.",
        "zh": "对不起，我不能回答这个问题。请换一个与 **切舍尔中学** 相关且合适的问题。",
    },
    "self_harm": {
        "en": (
            "I’m really sorry you’re feeling this way. I can’t help with self-harm requests.\n\n"
            "If you’re in immediate danger, call your local emergency number right now. "
            "If you’re in the U.S., you can call or text **988** for the Suicide & Crisis Lifeline."
        ),
        "zh": (
            "听到你这样说我很难过。我不能帮助自残/自杀相关的请求。\n\n"
            "如果你有立即的危险，请立刻联系当地紧急电话。若你在美国，可拨打或短信 **988**（危机干预热线）。"
        ),
    },
}

def _render_refusal(kind: str, forced_lang_base: Optional[str], message: str) -> str:
    base = (forced_lang_base or "").split("-")[0].strip().lower()
    if not base:
        base = _guess_lang_code_from_message(message)
    if base not in ("en", "zh"):
        base = "en"
    return REFUSAL_TEMPLATES.get(kind, {}).get(base) or REFUSAL_TEMPLATES["non_ca"]["en"]

def _route_message(message: str) -> str:
    m = (message or "").strip()
    if not m:
        return "allow"

    # Hard blocks first (instant)
    if _contains_any(m, SELF_HARM_KEYWORDS):
        return "self_harm"
    if _contains_any(m, ADULT_CONTENT_KEYWORDS):
        return "adult"
    if _contains_any(m, SENSITIVE_SCHOOL_CASE_KEYWORDS):
        return "sensitive_school"

    # Determine CA-related:
    # 1) explicit keywords
    is_ca_kw = _contains_any(m, CA_ENTITY_KEYWORDS) or _contains_any(m, CA_TOPIC_KEYWORDS)

    # 2) retrieval signal (more robust than keywords)
    # If we can retrieve anything from sources, it's likely school-related.
    retrieved = _retrieve(m, top_k=2)
    is_ca_retrieval = len(retrieved) > 0

    if not (is_ca_kw or is_ca_retrieval):
        return "non_ca"

    return "allow"

# ============================================================
# OpenAI (with strict timeout, no hanging)
# ============================================================
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
FALLBACK_EN = (
    "I'm not certain from my current references. "
    "Please contact admission@cheshireacademy.org | +1-203-439-7250."
)
FALLBACK_ZH = (
    "我目前从现有资料中无法确认。建议直接联系学校：admission@cheshireacademy.org | +1-203-439-7250。"
)

_OAI_CLIENT: Optional[OpenAI] = None

def _get_oai_client() -> Optional[OpenAI]:
    global _OAI_CLIENT
    if _OAI_CLIENT is not None:
        return _OAI_CLIENT

    api_key = os.environ.get("OPENAI_API_KEY", "") or ""
    if not api_key.strip():
        print("WARNING: OPENAI_API_KEY is not set.")
        _OAI_CLIENT = None
        return None

    # IMPORTANT: timeout prevents infinite "processing" in Spaces
    timeout_s = float(os.getenv("OPENAI_TIMEOUT", "20"))
    try:
        _OAI_CLIENT = OpenAI(
            api_key=api_key,
            timeout=httpx.Timeout(timeout_s),
            max_retries=0,
        )
        return _OAI_CLIENT
    except Exception as e:
        print("OpenAI client init error:", e)
        _OAI_CLIENT = None
        return None

def _ask(message: str, forced_lang_base: Optional[str] = None) -> str:
    """
    OpenAI call with RAG. Always returns quickly due to timeout + exception handling.
    """
    system = (
        "You are a multilingual Cheshire Academy FAQ assistant.\n"
        "You are given reference text from the official Cheshire Academy website.\n"
        "Answer questions about Cheshire Academy based ONLY on the provided references.\n"
        "If the references do not clearly answer, say you are not sure and suggest contacting the school. Do NOT guess.\n"
        "Do NOT discuss allegations, lawsuits, or specific legal cases; redirect to admissions/campus questions.\n"
        "Be concise and student-friendly.\n"
    )

    if forced_lang_base and forced_lang_base != "auto":
        system += f"Always answer in the language whose base code is '{forced_lang_base}'.\n"

    retrieved = _retrieve(message, top_k=4)
    context = ""
    if retrieved:
        chunks = []
        for doc in retrieved:
            chunks.append(f"Source file: {doc['path']}\n{doc['text']}")
        context = (
            "Reference excerpts from the official Cheshire Academy website:\n\n"
            + "\n\n---\n\n".join(chunks)
        )

    client = _get_oai_client()
    if client is None:
        # quick fallback based on language
        return FALLBACK_ZH if forced_lang_base == "zh" else FALLBACK_EN

    messages = [{"role": "system", "content": system}]
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": message})

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=600,
        )
        out = (resp.choices[0].message.content or "").strip()
        if out:
            return out
        return FALLBACK_ZH if forced_lang_base == "zh" else FALLBACK_EN
    except Exception as e:
        print("OpenAI error:", repr(e))
        return FALLBACK_ZH if forced_lang_base == "zh" else FALLBACK_EN

# ============================================================
# Languages & greetings (ALL translated, NO dynamic OpenAI translate)
# ============================================================
LANG_ITEMS = [
    ("auto", "Auto / 自动 (auto)"),
    ("en", "English"),
    ("zh", "中文（简体）"),
    ("zh-Hant", "中文（繁體）"),
    ("yue", "粵語 / Cantonese"),
    ("es", "Español"),
    ("fr", "Français"),
    ("de", "Deutsch"),
    ("ja", "日本語"),
    ("ko", "한국어"),
    ("pt", "Português"),
    ("ru", "Русский"),
    ("uk", "Українська"),
    ("ar", "العربية"),
    ("hi", "हिन्दी"),
    ("bn", "বাংলা"),
    ("id", "Bahasa Indonesia"),
    ("vi", "Tiếng Việt"),
    ("th", "ภาษาไทย"),
    ("it", "Italiano"),
    ("nl", "Nederlands"),
    ("sv", "Svenska"),
    ("da", "Dansk"),
    ("fi", "Suomi"),
    ("pl", "Polski"),
    ("tr", "Türkçe"),
    ("he", "עברית"),
    ("el", "Ελληνικά"),
    ("cs", "Čeština"),
    ("ro", "Română"),
    ("hu", "Magyar"),
    ("bg", "Български"),
    ("sr", "Српски"),
    ("hr", "Hrvatski"),
    ("sk", "Slovenčina"),
    ("sl", "Slovenščina"),
    ("et", "Eesti"),
    ("lv", "Latviešu"),
    ("lt", "Lietuvių"),
    ("ms", "Bahasa Melayu"),
    ("fil", "Filipino"),
    ("sw", "Kiswahili"),
    ("ta", "தமிழ்"),
    ("te", "తెలుగు"),
    ("ml", "മലയാളം"),
    ("mr", "मराठी"),
    ("gu", "ગુજરાતી"),
    ("pa", "ਪੰਜਾਬੀ"),
    ("fa", "فارسی"),
    ("ur", "اردو"),
]

LANG_LABEL_BY_CODE = {code: label for code, label in LANG_ITEMS}
LANG_CODE_BY_LABEL = {label: code for code, label in LANG_ITEMS}

def base_lang(code: str) -> str:
    return (code or "en").split("-")[0]

GREETINGS = {
    "en": (
        "👋 Hi! I'm the Cheshire Academy Chatbot.\n\n"
        "I can answer questions about academics, admissions, campus life, athletics, forms, calendars, and more.\n"
        "• Ask in English or change the language via the globe icon.\n"
        "• One clear question at a time works best.\n\n"
        "How can I help today?"
    ),
    "zh": (
        "👋 你好，我是切舍尔中学的智能助手。\n\n"
        "我可以解答：课程与学术、招生申请、校园生活、体育项目、表格与日历等问题。\n"
        "• 可直接用中文提问，或点击右上角地球图标切换语言。\n"
        "• 建议一次只问一个清晰的问题，效果更好。\n\n"
        "请问你想了解什么？"
    ),
    "zh-Hant": (
        "👋 你好，我是切舍爾中學的智能助手。\n\n"
        "我可以解答：課程與學術、招生申請、校園生活、體育項目、表格與日曆等問題。\n"
        "• 可直接用中文提問，或點擊右上角地球圖標切換語言。\n"
        "• 建議一次只問一個清晰的問題，效果更好。\n\n"
        "請問你想了解什麼？"
    ),
    "yue": (
        "👋 你好！我係切舍爾中學嘅智能助手。\n\n"
        "我可以解答：課程與學術、招生申請、校園生活、體育項目、表格同日曆等問題。\n"
        "• 你可以用粵語/中文提問，或者撳右上角地球圖示轉語言。\n"
        "• 建議一次問一條清晰問題，效果最好。\n\n"
        "想了解咩？"
    ),
    "es": (
        "👋 ¡Hola! Soy el chatbot de Cheshire Academy.\n\n"
        "Puedo responder preguntas sobre estudios, admisiones, vida en el campus, deportes, formularios, calendarios y más.\n"
        "• Pregunta en español o cambia de idioma con el icono del globo.\n"
        "• Una pregunta clara a la vez funciona mejor.\n\n"
        "¿En qué puedo ayudarte hoy?"
    ),
    "fr": (
        "👋 Bonjour ! Je suis le chatbot de Cheshire Academy.\n\n"
        "Je peux répondre aux questions sur les études, l’admission, la vie sur le campus, le sport, les formulaires, le calendrier, etc.\n"
        "• Pose ta question en français ou change de langue avec l’icône du globe.\n"
        "• Une question claire à la fois, c’est l’idéal.\n\n"
        "Comment puis-je t’aider ?"
    ),
    "de": (
        "👋 Hallo! Ich bin der Chatbot der Cheshire Academy.\n\n"
        "Ich beantworte Fragen zu Unterricht, Aufnahme, Campusleben, Sport, Formularen, Kalendern und mehr.\n"
        "• Stelle deine Frage auf Deutsch oder wechsle die Sprache über das Globus-Symbol.\n"
        "• Eine klare Frage nach der anderen funktioniert am besten.\n\n"
        "Wie kann ich heute helfen?"
    ),
    "ja": (
        "👋 こんにちは！Cheshire Academy のチャットボットです。\n\n"
        "学業・入学（出願）・キャンパスライフ・スポーツ・書類・カレンダーなどについて回答できます。\n"
        "• 日本語で質問するか、地球アイコンで言語を変更できます。\n"
        "• 1回に1つの明確な質問だとより良いです。\n\n"
        "今日は何をお手伝いしましょうか？"
    ),
    "ko": (
        "👋 안녕하세요! Cheshire Academy 챗봇입니다.\n\n"
        "학업, 입학/지원, 캠퍼스 생활, 스포츠, 서류, 일정 등 학교 관련 질문에 답할 수 있어요.\n"
        "• 한국어로 질문하거나 지구본 아이콘으로 언어를 바꿀 수 있어요.\n"
        "• 한 번에 하나의 명확한 질문이 가장 좋아요.\n\n"
        "무엇을 도와드릴까요?"
    ),
    "pt": (
        "👋 Olá! Sou o chatbot da Cheshire Academy.\n\n"
        "Posso responder perguntas sobre estudos, admissões, vida no campus, esportes, formulários, calendários e mais.\n"
        "• Pergunte em português ou mude o idioma pelo ícone do globo.\n"
        "• Uma pergunta clara por vez funciona melhor.\n\n"
        "Como posso ajudar hoje?"
    ),
    "ru": (
        "👋 Привет! Я чат-бот Cheshire Academy.\n\n"
        "Я могу отвечать на вопросы об учебе, поступлении, жизни кампуса, спорте, формах, календарях и другом.\n"
        "• Задавайте вопрос по-русски или переключайте язык через значок глобуса.\n"
        "• Лучше задавать по одному четкому вопросу.\n\n"
        "Чем помочь сегодня?"
    ),
    "uk": (
        "👋 Вітаю! Я чатбот Cheshire Academy.\n\n"
        "Можу відповідати на питання про навчання, вступ, життя кампусу, спорт, форми, календарі тощо.\n"
        "• Питайте українською або змінюйте мову через значок глобуса.\n"
        "• Найкраще — одне чітке питання за раз.\n\n"
        "Чим можу допомогти сьогодні?"
    ),
    "ar": (
        "👋 مرحبًا! أنا روبوت الدردشة الخاص بـ Cheshire Academy.\n\n"
        "يمكنني الإجابة عن أسئلة حول الدراسة، القبول، الحياة في الحرم، الرياضة، النماذج، التقويم والمزيد.\n"
        "• اسأل بالعربية أو غيّر اللغة عبر أيقونة الكرة الأرضية.\n"
        "• سؤال واضح واحد في كل مرة يعطي نتيجة أفضل.\n\n"
        "كيف يمكنني مساعدتك اليوم؟"
    ),
    "hi": (
        "👋 नमस्ते! मैं Cheshire Academy का चैटबॉट हूँ।\n\n"
        "मैं पढ़ाई, प्रवेश, कैंपस लाइफ़, खेल, फ़ॉर्म, कैलेंडर आदि से जुड़े सवालों के जवाब दे सकता/सकती हूँ।\n"
        "• हिंदी में पूछें या ग्लोब आइकन से भाषा बदलें।\n"
        "• एक बार में एक स्पष्ट प्रश्न सबसे अच्छा रहता है।\n\n"
        "आज मैं कैसे मदद करूँ?"
    ),
    "bn": (
        "👋 হ্যালো! আমি Cheshire Academy-এর চ্যাটবট।\n\n"
        "আমি একাডেমিকস, ভর্তি, ক্যাম্পাস জীবন, খেলাধুলা, ফর্ম, ক্যালেন্ডার ইত্যাদি সম্পর্কে প্রশ্নের উত্তর দিতে পারি।\n"
        "• বাংলায় জিজ্ঞেস করুন বা গ্লোব আইকন দিয়ে ভাষা বদলান।\n"
        "• একবারে একটি স্পষ্ট প্রশ্ন করলে ভালো কাজ করে।\n\n"
        "আজ কীভাবে সাহায্য করতে পারি?"
    ),
    "id": (
        "👋 Hai! Saya chatbot Cheshire Academy.\n\n"
        "Saya bisa menjawab pertanyaan tentang akademik, penerimaan, kehidupan kampus, olahraga, formulir, kalender, dan lainnya.\n"
        "• Tanya dalam Bahasa Indonesia atau ganti bahasa lewat ikon bumi.\n"
        "• Satu pertanyaan yang jelas setiap kali akan bekerja paling baik.\n\n"
        "Apa yang bisa saya bantu hari ini?"
    ),
    "vi": (
        "👋 Xin chào! Tôi là chatbot của Cheshire Academy.\n\n"
        "Tôi có thể trả lời về học thuật, tuyển sinh, đời sống nội trú/campus, thể thao, biểu mẫu, lịch và nhiều nội dung khác.\n"
        "• Hỏi bằng tiếng Việt hoặc đổi ngôn ngữ bằng biểu tượng quả địa cầu.\n"
        "• Mỗi lần một câu hỏi rõ ràng sẽ hiệu quả hơn.\n\n"
        "Tôi có thể giúp gì hôm nay?"
    ),
    "th": (
        "👋 สวัสดี! ฉันคือแชตบอทของ Cheshire Academy\n\n"
        "ฉันตอบคำถามเกี่ยวกับการเรียน การรับสมัคร ชีวิตในแคมปัส กีฬา แบบฟอร์ม ปฏิทิน และอื่น ๆ ได้\n"
        "• ถามเป็นภาษาไทยหรือเปลี่ยนภาษาด้วยไอคอนลูกโลก\n"
        "• ถามทีละคำถามที่ชัดเจนจะดีที่สุด\n\n"
        "วันนี้ให้ช่วยอะไรดี?"
    ),
    "it": (
        "👋 Ciao! Sono il chatbot di Cheshire Academy.\n\n"
        "Posso rispondere su studi, ammissioni, vita nel campus, sport, moduli, calendari e altro.\n"
        "• Fai domande in italiano o cambia lingua con l’icona del globo.\n"
        "• Una domanda chiara alla volta funziona meglio.\n\n"
        "Come posso aiutarti oggi?"
    ),
    "nl": (
        "👋 Hoi! Ik ben de chatbot van Cheshire Academy.\n\n"
        "Ik kan vragen beantwoorden over onderwijs, toelating, campusleven, sport, formulieren, kalenders en meer.\n"
        "• Stel je vraag in het Nederlands of verander de taal via het wereldbol-icoon.\n"
        "• Eén duidelijke vraag per keer werkt het best.\n\n"
        "Waarmee kan ik vandaag helpen?"
    ),
    "sv": (
        "👋 Hej! Jag är Cheshire Academys chatbot.\n\n"
        "Jag kan svara på frågor om studier, antagning, campusliv, idrott, formulär, kalender och mer.\n"
        "• Fråga på svenska eller byt språk via jordglobsikonen.\n"
        "• En tydlig fråga i taget fungerar bäst.\n\n"
        "Hur kan jag hjälpa idag?"
    ),
    "da": (
        "👋 Hej! Jeg er Cheshire Academys chatbot.\n\n"
        "Jeg kan svare på spørgsmål om akademik, optagelse, campusliv, sport, formularer, kalendere og mere.\n"
        "• Spørg på dansk eller skift sprog via globus-ikonet.\n"
        "• Ét klart spørgsmål ad gangen virker bedst.\n\n"
        "Hvordan kan jeg hjælpe i dag?"
    ),
    "fi": (
        "👋 Hei! Olen Cheshire Academyn chatbot.\n\n"
        "Voin vastata kysymyksiin opinnoista, hakemisesta, kampuselämästä, urheilusta, lomakkeista, kalentereista ja muusta.\n"
        "• Kysy suomeksi tai vaihda kieltä maapallo-kuvakkeesta.\n"
        "• Yksi selkeä kysymys kerrallaan toimii parhaiten.\n\n"
        "Miten voin auttaa tänään?"
    ),
    "pl": (
        "👋 Cześć! Jestem chatbotem Cheshire Academy.\n\n"
        "Mogę odpowiadać na pytania o naukę, rekrutację, życie na kampusie, sport, formularze, kalendarze i inne.\n"
        "• Pytaj po polsku lub zmień język ikoną globusa.\n"
        "• Najlepiej zadawać jedno jasne pytanie naraz.\n\n"
        "W czym mogę pomóc?"
    ),
    "tr": (
        "👋 Merhaba! Ben Cheshire Academy sohbet botuyum.\n\n"
        "Akademikler, başvuru/kabul, kampüs yaşamı, spor, formlar, takvimler ve daha fazlası hakkında soruları yanıtlayabilirim.\n"
        "• Türkçe sorabilir veya dünya simgesinden dili değiştirebilirsiniz.\n"
        "• Her seferinde tek net soru en iyisidir.\n\n"
        "Bugün nasıl yardımcı olabilirim?"
    ),
    "he": (
        "👋 שלום! אני הצ׳אטבוט של Cheshire Academy.\n\n"
        "אני יכול/ה לענות על שאלות על לימודים, קבלה, חיי קמפוס, ספורט, טפסים, לוחות שנה ועוד.\n"
        "• שאל/י בעברית או החלף/י שפה באמצעות סמל הגלובוס.\n"
        "• שאלה ברורה אחת בכל פעם עובדת הכי טוב.\n\n"
        "איך אפשר לעזור היום?"
    ),
    "el": (
        "👋 Γεια! Είμαι το chatbot του Cheshire Academy.\n\n"
        "Μπορώ να απαντήσω για σπουδές, εισαγωγή, ζωή στο campus, αθλητισμό, φόρμες, ημερολόγια και άλλα.\n"
        "• Ρώτησε στα ελληνικά ή άλλαξε γλώσσα από το εικονίδιο της υδρογείου.\n"
        "• Μία καθαρή ερώτηση τη φορά είναι καλύτερα.\n\n"
        "Πώς μπορώ να βοηθήσω σήμερα;"
    ),
    "cs": (
        "👋 Ahoj! Jsem chatbot Cheshire Academy.\n\n"
        "Mohu odpovídat na otázky o studiu, přijímání, životě na kampusu, sportu, formulářích, kalendářích a dalších.\n"
        "• Ptej se česky nebo změň jazyk přes ikonu glóbu.\n"
        "• Nejlépe funguje jedna jasná otázka najednou.\n\n"
        "S čím mohu pomoci?"
    ),
    "ro": (
        "👋 Salut! Sunt chatbotul Cheshire Academy.\n\n"
        "Pot răspunde la întrebări despre studii, admitere, viața în campus, sport, formulare, calendare și altele.\n"
        "• Întreabă în română sau schimbă limba din pictograma globului.\n"
        "• O întrebare clară pe rând funcționează cel mai bine.\n\n"
        "Cu ce te pot ajuta azi?"
    ),
    "hu": (
        "👋 Szia! A Cheshire Academy chatbotja vagyok.\n\n"
        "Tudok válaszolni tanulmányokkal, felvételivel, campusélettel, sporttal, űrlapokkal, naptárakkal kapcsolatos kérdésekre.\n"
        "• Kérdezz magyarul vagy válts nyelvet a földgömb ikonon.\n"
        "• Egyszerre egy világos kérdés működik a legjobban.\n\n"
        "Miben segíthetek ma?"
    ),
    "bg": (
        "👋 Здравей! Аз съм чатботът на Cheshire Academy.\n\n"
        "Мога да отговарям на въпроси за обучение, прием, живот в кампуса, спорт, формуляри, календари и др.\n"
        "• Питай на български или смени езика чрез иконата с глобуса.\n"
        "• Най-добре е по един ясен въпрос наведнъж.\n\n"
        "С какво да помогна днес?"
    ),
    "sr": (
        "👋 Здраво! Ја сам чатбот Cheshire Academy.\n\n"
        "Могу да одговарам на питања о настави, упису, животу на кампусу, спорту, формуларима, календарима и још.\n"
        "• Питај на српском или промени језик преко иконе глобуса.\n"
        "• Једно јасно питање одједном је најбоље.\n\n"
        "Како могу да помогнем данас?"
    ),
    "hr": (
        "👋 Bok! Ja sam chatbot Cheshire Academy.\n\n"
        "Mogu odgovarati na pitanja o nastavi, upisima, životu na kampusu, sportu, obrascima, kalendarima i više.\n"
        "• Pitaj na hrvatskom ili promijeni jezik preko ikone globusa.\n"
        "• Najbolje radi jedno jasno pitanje odjednom.\n\n"
        "Kako mogu pomoći danas?"
    ),
    "sk": (
        "👋 Ahoj! Som chatbot Cheshire Academy.\n\n"
        "Môžem odpovedať na otázky o štúdiu, prijímaní, živote na kampuse, športe, formulároch, kalendároch a ďalších.\n"
        "• Pýtaj sa po slovensky alebo zmeň jazyk cez ikonu glóbusu.\n"
        "• Najlepšie funguje jedna jasná otázka naraz.\n\n"
        "Ako môžem pomôcť?"
    ),
    "sl": (
        "👋 Živjo! Sem chatbot Cheshire Academy.\n\n"
        "Lahko odgovorim na vprašanja o študiju, vpisu, življenju v kampusu, športu, obrazcih, koledarjih in več.\n"
        "• Vprašaj v slovenščini ali zamenjaj jezik z ikono globusa.\n"
        "• Najbolje deluje eno jasno vprašanje naenkrat.\n\n"
        "Kako lahko pomagam danes?"
    ),
    "et": (
        "👋 Tere! Olen Cheshire Academy vestlusrobot.\n\n"
        "Saan vastata küsimustele õppe, vastuvõtu, campus’e elu, spordi, vormide, kalendrite ja muu kohta.\n"
        "• Küsi eesti keeles või vaheta keelt gloobuse ikoonilt.\n"
        "• Üks selge küsimus korraga töötab kõige paremini.\n\n"
        "Kuidas saan aidata?"
    ),
    "lv": (
        "👋 Sveiki! Es esmu Cheshire Academy čatbots.\n\n"
        "Varu atbildēt uz jautājumiem par mācībām, uzņemšanu, dzīvi campusā, sportu, veidlapām, kalendāriem u.c.\n"
        "• Jautā latviski vai maini valodu ar globusa ikonu.\n"
        "• Vislabāk — viens skaidrs jautājums vienā reizē.\n\n"
        "Kā varu palīdzēt šodien?"
    ),
    "lt": (
        "👋 Sveiki! Aš esu Cheshire Academy pokalbių robotas.\n\n"
        "Galiu atsakyti į klausimus apie mokslą, priėmimą, gyvenimą miestelyje, sportą, formas, kalendorius ir kt.\n"
        "• Klausk lietuviškai arba keisk kalbą per gaublio ikoną.\n"
        "• Geriausia — po vieną aiškų klausimą.\n\n"
        "Kuo galiu padėti šiandien?"
    ),
    "ms": (
        "👋 Hai! Saya chatbot Cheshire Academy.\n\n"
        "Saya boleh jawab soalan tentang akademik, kemasukan, kehidupan kampus, sukan, borang, kalendar dan banyak lagi.\n"
        "• Tanya dalam Bahasa Melayu atau tukar bahasa melalui ikon glob.\n"
        "• Satu soalan yang jelas setiap kali paling berkesan.\n\n"
        "Apa yang boleh saya bantu hari ini?"
    ),
    "fil": (
        "👋 Kumusta! Ako ang chatbot ng Cheshire Academy.\n\n"
        "Maaari akong sumagot tungkol sa akademiko, admissions, buhay sa campus, sports, forms, calendars, at iba pa.\n"
        "• Magtanong sa Filipino o palitan ang wika gamit ang globe icon.\n"
        "• Isang malinaw na tanong sa bawat pagkakataon ang mas epektibo.\n\n"
        "Paano ako makakatulong ngayon?"
    ),
    "sw": (
        "👋 Hujambo! Mimi ni chatbot wa Cheshire Academy.\n\n"
        "Naweza kujibu maswali kuhusu masomo, udahili, maisha ya kampasi, michezo, fomu, kalenda na mengine.\n"
        "• Uliza kwa Kiswahili au badilisha lugha kupitia ikoni ya dunia.\n"
        "• Swali moja lililo wazi kwa wakati mmoja hufanya kazi vizuri zaidi.\n\n"
        "Naweza kukusaidia vipi leo?"
    ),
    "ta": (
        "👋 வணக்கம்! நான் Cheshire Academy-யின் சாட்பாட்.\n\n"
        "படிப்பு, சேர்க்கை, வளாக வாழ்க்கை, விளையாட்டு, படிவங்கள், நாட்காட்டி போன்ற கேள்விகளுக்கு பதில் அளிக்க முடியும்.\n"
        "• தமிழில் கேளுங்கள் அல்லது உலகக் குறியீட்டில் மொழியை மாற்றலாம்.\n"
        "• ஒரே நேரத்தில் ஒரு தெளிவான கேள்வி சிறந்தது.\n\n"
        "இன்று நான் எப்படி உதவலாம்?"
    ),
    "te": (
        "👋 నమస్తే! నేను Cheshire Academy చాట్‌బాట్‌ను.\n\n"
        "అకాడెమిక్స్, అడ్మిషన్స్, క్యాంపస్ లైఫ్, క్రీడలు, ఫార్ములు, క్యాలెండర్లు తదితరాలపై ప్రశ్నలకు సమాధానం ఇవ్వగలను.\n"
        "• తెలుగు లో అడగండి లేదా గ్లోబ్ ఐకాన్ ద్వారా భాష మార్చండి.\n"
        "• ఒక్కసారి ఒక స్పష్టమైన ప్రశ్న అడిగితే మెరుగ్గా పనిచేస్తుంది.\n\n"
        "ఈ రోజు ఎలా సహాయం చేయగలను?"
    ),
    "ml": (
        "👋 ഹലോ! ഞാൻ Cheshire Academy-യുടെ ചാറ്റ്ബോട്ടാണ്.\n\n"
        "പഠനം, അഡ്മിഷൻ, ക്യാമ്പസ് ജീവിതം, സ്പോർട്സ്, ഫോമുകൾ, കലണ്ടർ എന്നിവയെക്കുറിച്ചുള്ള ചോദ്യങ്ങൾക്ക് മറുപടി നൽകാം.\n"
        "• മലയാളത്തിൽ ചോദിക്കൂ അല്ലെങ്കിൽ ഗ്ലോബ് ഐക്കൺ വഴി ഭാഷ മാറ്റൂ.\n"
        "• ഒരേസമയം ഒരു വ്യക്തമായ ചോദ്യം ഏറ്റവും നല്ലത്.\n\n"
        "ഇന്ന് എങ്ങനെ സഹായിക്കാം?"
    ),
    "mr": (
        "👋 नमस्कार! मी Cheshire Academy चा चॅटबॉट आहे.\n\n"
        "अकॅडमिक्स, प्रवेश, कॅम्पस जीवन, क्रीडा, फॉर्म्स, कॅलेंडर इत्यादींबद्दल प्रश्नांची उत्तरे देऊ शकतो/शकते.\n"
        "• मराठीत विचारा किंवा ग्लोब आयकॉनने भाषा बदला.\n"
        "• एकावेळी एक स्पष्ट प्रश्न सर्वात चांगला.\n\n"
        "आज मी कशी/कसा मदत करू?"
    ),
    "gu": (
        "👋 નમસ્તે! હું Cheshire Academy નો ચેટબોટ છું.\n\n"
        "અકાદમિક્સ, પ્રવેશ, કેમ્પસ જીવન, રમતો, ફોર્મ્સ, કેલેન્ડર વગેરે અંગે પ્રશ્નોના જવાબ આપી શકું છું.\n"
        "• ગુજરાતી માં પૂછો અથવા ગ્લોબ આઇકનથી ભાષા બદલો.\n"
        "• એક સમયે એક સ્પષ્ટ પ્રશ્ન વધુ સારું કામ કરે છે.\n\n"
        "આજે હું કેવી રીતે મદદ કરી શકું?"
    ),
    "pa": (
        "👋 ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ Cheshire Academy ਦਾ ਚੈਟਬੋਟ ਹਾਂ।\n\n"
        "ਮੈਂ ਅਕੈਡਮਿਕਸ, ਐਡਮਿਸ਼ਨ, ਕੈਂਪਸ ਲਾਈਫ਼, ਖੇਡਾਂ, ਫਾਰਮ, ਕੈਲੰਡਰ ਆਦਿ ਬਾਰੇ ਸਵਾਲਾਂ ਦੇ ਜਵਾਬ ਦੇ ਸਕਦਾ/ਸਕਦੀ ਹਾਂ।\n"
        "• ਪੰਜਾਬੀ ਵਿੱਚ ਪੁੱਛੋ ਜਾਂ ਗਲੋਬ ਆਇਕਨ ਨਾਲ ਭਾਸ਼ਾ ਬਦਲੋ।\n"
        "• ਇੱਕ ਵਾਰੀ ਇੱਕ ਸਾਫ਼ ਸਵਾਲ ਸਭ ਤੋਂ ਵਧੀਆ।\n\n"
        "ਅੱਜ ਮੈਂ ਕਿਵੇਂ ਮਦਦ ਕਰਾਂ?"
    ),
    "fa": (
        "👋 سلام! من چت‌بات Cheshire Academy هستم.\n\n"
        "می‌توانم به پرسش‌هایی دربارهٔ تحصیل، پذیرش، زندگی در کمپ، ورزش، فرم‌ها، تقویم و موارد دیگر پاسخ بدهم.\n"
        "• به فارسی بپرسید یا از طریق آیکون کرهٔ زمین زبان را تغییر دهید.\n"
        "• هر بار یک سؤال واضح بهترین نتیجه را می‌دهد.\n\n"
        "امروز چطور می‌توانم کمک کنم؟"
    ),
    "ur": (
        "👋 سلام! میں Cheshire Academy کا چیٹ بوٹ ہوں۔\n\n"
        "میں تعلیمی امور، داخلہ، کیمپس لائف، کھیل، فارمز، کیلنڈر وغیرہ سے متعلق سوالات کے جواب دے سکتا/سکتی ہوں۔\n"
        "• اردو میں پوچھیں یا گلوب آئیکن سے زبان تبدیل کریں۔\n"
        "• ایک وقت میں ایک واضح سوال بہترین رہتا ہے۔\n\n"
        "آج میں کیسے مدد کر سکتا/سکتی ہوں؟"
    ),
}

def greeting_for(code: str) -> str:
    c = code or "auto"
    if c == "auto":
        return GREETINGS["en"]
    # prefer exact, then base, then fallback en
    if c in GREETINGS:
        return GREETINGS[c]
    b = base_lang(c)
    return GREETINGS.get(b, GREETINGS["en"])

SEARCH_PLACEHOLDERS = {
    "en": "Search languages",
    "zh": "搜索语言",
    "zh-Hant": "搜尋語言",
    "yue": "搜尋語言",
    "fr": "Rechercher une langue",
    "es": "Buscar idioma",
    "de": "Sprache suchen",
    "ja": "言語を検索",
    "ko": "언어 검색",
    "ru": "Найти язык",
    "uk": "Пошук мови",
    "ar": "ابحث عن لغة",
    "hi": "भाषा खोजें",
    "bn": "ভাষা অনুসন্ধান করুন",
    "id": "Cari bahasa",
    "vi": "Tìm ngôn ngữ",
    "th": "ค้นหาภาษา",
    "it": "Cerca lingua",
    "pt": "Pesquisar idioma",
}

def search_placeholder_for(code: str) -> str:
    b = base_lang(code)
    return SEARCH_PLACEHOLDERS.get(b, SEARCH_PLACEHOLDERS["en"])

# ============================================================
# History normalization (Gradio messages format)
# ============================================================
def _ensure_messages_history(history: Any) -> List[Dict[str, str]]:
    """
    Ensure we ALWAYS return:
      [{"role":"assistant","content":"..."}, {"role":"user","content":"..."}, ...]
    """
    if not history:
        return []

    out: List[Dict[str, str]] = []

    if isinstance(history, list) and history:
        first = history[0]

        # already messages format
        if isinstance(first, dict) and "role" in first and "content" in first:
            for m in history:
                if isinstance(m, dict) and "role" in m and "content" in m:
                    out.append({
                        "role": str(m["role"]),
                        "content": _normalize_content(m["content"]),
                    })
            return out

        # old tuple/list format (user, assistant)
        if isinstance(first, (tuple, list)) and len(first) == 2:
            for pair in history:
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    continue
                u, a = pair
                u = _normalize_content(u)
                a = _normalize_content(a)
                if u.strip():
                    out.append({"role": "user", "content": u})
                if a.strip():
                    out.append({"role": "assistant", "content": a})
            return out

    # fallback: try to stringify the whole thing safely
    try:
        s = _normalize_content(history)
        if s.strip():
            out.append({"role": "assistant", "content": s})
    except Exception:
        pass
    return out

# ============================================================
# Main respond / greet
# ============================================================
def respond(message: str, history, lang_code: str):
    history_msgs = _ensure_messages_history(history)

    message = (message or "").strip()
    if not message:
        return "", history_msgs

    forced = None if lang_code == "auto" else base_lang(lang_code)

    # FAST routing (no OpenAI calls for refusals)
    route = _route_message(message)
    if route != "allow":
        answer = _render_refusal(route, forced, message)
    else:
        answer = _ask(message, forced_lang_base=forced)

    history_msgs.append({"role": "user", "content": message})
    history_msgs.append({"role": "assistant", "content": _normalize_content(answer)})

    return "", history_msgs

def greet(lang_code: str):
    msg = greeting_for(lang_code)
    return [{"role": "assistant", "content": _normalize_content(msg)}]

# ============================================================
# Language menu UI logic (queue=False for instant response)
# ============================================================
def toggle_lang_menu(is_open: bool, lang_code: str):
    new_open = not bool(is_open)
    if new_open:
        current_label = LANG_LABEL_BY_CODE.get(lang_code, LANG_LABEL_BY_CODE["auto"])
        return (
            True,
            gr.update(visible=True),
            gr.update(value="", placeholder=search_placeholder_for(lang_code)),
            gr.update(value=current_label),
        )
    else:
        return (
            False,
            gr.update(visible=False),
            gr.update(),
            gr.update(),
        )

def filter_languages(search: str, lang_code: str):
    s = (search or "").strip().lower()
    if s:
        filtered = [(code, label) for code, label in LANG_ITEMS if s in label.lower()]
        if not filtered:
            filtered = LANG_ITEMS
    else:
        filtered = LANG_ITEMS

    choices = [label for _, label in filtered]
    current_label = LANG_LABEL_BY_CODE.get(lang_code, LANG_LABEL_BY_CODE["auto"])
    if current_label not in choices and choices:
        current_label = choices[0]
        lang_code = LANG_CODE_BY_LABEL.get(current_label, "auto")

    return gr.update(choices=choices, value=current_label), lang_code

def select_language(label: str):
    code = LANG_CODE_BY_LABEL.get(label, "auto")
    return (
        code,
        greet(code),
        gr.update(value="", placeholder=search_placeholder_for(code)),
        gr.update(visible=False),
        False,
    )

# ============================================================
# CSS (fix white bar + fixed overlay panel)
# ============================================================
CSS = """
#ca-wrapper {
  font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Top blue bar */
#ca-header {
  position: relative;
  background:#002f5f;
  color:#fff;
  padding:10px 16px;
  display:flex;
  align-items:center;
  justify-content:space-between;
}

#ca-title {
  font-size:18px;
  font-weight:600;
  color:#ffffff;
  flex: 1;
  padding-right:12px;
  white-space:normal;
  overflow:visible;
  text-overflow:clip;
}

/* Globe button */
#ca-lang-btn button {
  background:#e5e7eb;
  border-radius:999px;
  color:#1f2933;
  border:none;
  width:40px;
  height:40px;
  font-size:20px;
  display:flex;
  align-items:center;
  justify-content:center;
}
#ca-lang-btn button:hover { filter:brightness(1.03); }

/* Fixed overlay panel (prevents layout whitespace) */
#ca-lang-panel {
  position: fixed !important;
  top: 86px !important;
  right: 16px !important;
  width:280px;
  max-height:360px;
  background:#fff;
  border-radius:12px;
  box-shadow:0 12px 32px rgba(0,0,0,.35);
  padding:10px 10px 8px 10px;
  z-index:9999;
}

/* Search box */
#ca-lang-search textarea {
  resize:none !important;
  min-height:34px !important;
  max-height:34px !important;
  padding:6px 10px !important;
  font-size:13px !important;
  border-radius:999px !important;
  border:1px solid #d1d5db !important;
}

/* Language list, scrollable */
#ca-lang-radio {
  max-height:270px;
  overflow-y:auto;
  margin-top:8px;
}
"""

# ============================================================
# UI
# ============================================================
with gr.Blocks(fill_height=True, elem_id="ca-wrapper") as demo:
    lang_state = gr.State("auto")
    menu_open = gr.State(False)

    # Header
    with gr.Row(elem_id="ca-header"):
        gr.HTML('<div id="ca-title">Cheshire Academy Chatbot</div>')
        lang_btn = gr.Button("🌐", elem_id="ca-lang-btn", scale=0)

    # Language panel (fixed overlay)
    with gr.Column(visible=False, elem_id="ca-lang-panel") as lang_panel:
        lang_search = gr.Textbox(
            value="",
            placeholder=search_placeholder_for("en"),
            label="",
            show_label=False,
            lines=1,
            elem_id="ca-lang-search",
        )
        lang_radio = gr.Radio(
            choices=[label for _, label in LANG_ITEMS],
            value=LANG_LABEL_BY_CODE["auto"],
            label="",
            show_label=False,
            elem_id="ca-lang-radio",
        )

    chatbot = gr.Chatbot(height=420)
    msg_box = gr.Textbox(
        label="Ask a question",
        placeholder="Type your question here…",
        lines=2,
    )
    send_btn = gr.Button("Send")
    clear_btn = gr.ClearButton([chatbot, msg_box])

    # Chat logic (queue=True ok)
    send_btn.click(
        respond,
        inputs=[msg_box, chatbot, lang_state],
        outputs=[msg_box, chatbot],
        queue=True,
    )

    # Initial greeting (fast)
    demo.load(greet, inputs=[lang_state], outputs=[chatbot], queue=False)

    # Globe: toggle language panel (must be instant)
    lang_btn.click(
        toggle_lang_menu,
        inputs=[menu_open, lang_state],
        outputs=[menu_open, lang_panel, lang_search, lang_radio],
        queue=False,
    )

    # Search filter (instant)
    lang_search.change(
        filter_languages,
        inputs=[lang_search, lang_state],
        outputs=[lang_radio, lang_state],
        queue=False,
    )

    # Select language (instant UI + greeting update)
    lang_radio.change(
        select_language,
        inputs=[lang_radio],
        outputs=[lang_state, chatbot, lang_search, lang_panel, menu_open],
        queue=False,
    )

    # Click outside closes panel
    gr.HTML(
        """
        <script>
        (function() {
          function setup() {
            const panel = document.getElementById('ca-lang-panel');
            const btnWrapper = document.getElementById('ca-lang-btn');
            if (!panel || !btnWrapper) { setTimeout(setup, 600); return; }
            const btn = btnWrapper.querySelector('button');
            if (!btn) { setTimeout(setup, 600); return; }

            document.addEventListener('click', function(ev) {
              const style = window.getComputedStyle(panel);
              if (style.display === 'none' || panel.hidden) return;
              if (panel.contains(ev.target) || btnWrapper.contains(ev.target)) return;
              btn.click();
            });
          }
          window.setTimeout(setup, 600);
        })();
        </script>
        """,
        visible=False,
    )

if __name__ == "__main__":
    demo.launch(css=CSS)
