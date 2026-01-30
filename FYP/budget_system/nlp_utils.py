import re
from typing import List, Dict, Any
import spacy
from rapidfuzz import fuzz, process

nlp = spacy.load("en_core_web_sm")

# ---------------- Keyword Lists ----------------
EXPENSE_KEYWORDS = [
    "tapau", "makan", "minum", "belanja",


    # --- Core spending verbs ---
    "spend", "spent", "pay", "paid", "paying",
    "buy", "bought", "purchase", "purchased",
    "order", "ordered", "checkout",

    # --- Usage verbs ---
    "use", "used", "took", "take", "taken",

    # --- Fee-related ---
    "fee", "fees", "charges", "charge", "charged",
    "surcharge", "service fee", "processing fee",
    "admin fee", "atm fee", "maintenance fee",

    # --- Losing money ---
    "lose", "lost", "missing", "stolen",

    # --- Giving money away ---
    "give", "gave", "donate", "donated",
    "tip", "tipped", "charity", "sponsor", "sponsored",

    # --- Deduction-based ---
    "deduct", "deducted", "withdraw", "withdrawn", "cash out",

    # --- Transfers going OUT ---
    "transfer out", "transfer_out", "tf out",
    "banked out", "send money", "sent money",
    "sent rm", "paid rm", "settle debt", "settled",

    # --- Bills & utilities ---
    "pay bill", "paybills", "settle bill", "renew", "renewed",
    "subscription renewal", "auto debit", "autodebit",

    # --- Transport triggers ---
    "toll", "fare", "petrol", "fuel", "diesel",
    "parking", "grab", "uber", "ride", "bus fare",
    "train fare", "touch n go", "tng", "smarttag",

    # --- Delivery & platforms ---
    "grabfood", "foodpanda", "delivery fee",

    # --- Marketplace (expenses) ---
    "shopee", "lazada", "zalora", "shein",
    "temu", "aliexpress", "amazon",

    # --- Malaysian slang (expenses) ---
    "belanja", "tapau", "tapao", "tapaued",
    "beli", "bayar", "topap", "topap-ed",
    "makan", "minum", "bawak makan",

    # --- Outgoing refunds ---
    "refund_out", "return payment", "hutang bayar",
]


INCOME_KEYWORDS = [

    # --- Receiving verbs ---
    "receive", "received", "get", "got",
    "earn", "earned", "incoming", "credited",
    "credit", "credited rm",

    # --- Salary & wages ---
    "salary", "gaji", "wage", "payout", "payday",

    # --- Bonuses & allowances ---
    "bonus", "allowance", "incentive",
    "stipend", "commission", "overtime pay",
    "ot pay", "increment",

    # --- Gifts ---
    "gift", "given", "angpau", "angpao",
    "duit raya", "duit hadiah",

    # --- Business / side income ---
    "freelance", "side gig", "side job",
    "side income", "business income",
    "sales", "sell", "sold", "resell", "resold",
    "profit from sale",

    # --- Transfers IN ---
    "deposit", "banked in", "bank in",
    "transfer in", "transfer_in", "tf in",
    "duit masuk", "masuk duit",

    # --- Wallet top-ups (crediting user wallet) ---
    "topup bonus", "reload bonus",

    # --- Refunds & cashback ---
    "refund", "rebate", "cashback", "reward",
    "payback", "compensation", "reimbursement",

    # --- Investment income ---
    "dividend", "interest", "profit", "return",
    "roi", "capital gain",

    # --- Prize money ---
    "win", "won", "prize", "jackpot",
    "rewarded", "contest winning",

    # --- Scholarships ---
    "scholarship", "bursary", "grant",
]


CATEGORIES = {

    # =====================================================
    #                    FOOD & DRINK
    # =====================================================
    "food": [
        "tapau", "tapao", "tapaued",
        "makan", "minum", "air", "nasi", "lunch", "dinner", "breakfast","water",


        # General
        "food", "meal", "eat", "snack", "snacks", "drinks",
        "drink", "beverage", "cafe", "restaurant",

        # Meals
        "breakfast", "lunch", "dinner", "brunch", "supper",

        # Malaysian foods
        "nasi lemak", "nasi goreng", "nasi ayam", "ayam penyet",
        "roti canai", "roti telur", "thosai", "chapati",
        "mee goreng", "mee kari", "laksa", "asam laksa",
        "char kuey teow", "char kway teow", "maggie goreng",
        "ayam goreng", "ikan goreng", "ramen", "udon", "sushi",

        # Drinks
        "teh tarik", "teh ais", "kopi", "coffee", "milo",
        "bandung", "lime juice", "sirap ais",

        # Snacks & desserts
        "kuih", "ice cream", "dessert", "cake", "pastry",
        "donut", "bread", "bun", "croissant",

        # Chill/Popular chains
        "mcdonald", "mcd", "kfc", "burger king",
        "pizza hut", "dominos", "marrybrown",
        "subway", "starbucks", "dunkin",
        "secret recipe", "tealive", "chatime",
        "gigi coffee", "old town", "papparich",

        # Groceries
        "groceries", "grocery", "supermarket",
        "jaya grocer", "mydin", "aeon", "lotus",
        "giant", "tesco", "fresh market",

        # Delivery
        "grabfood", "foodpanda"
    ],

    # =====================================================
    #                      TRANSPORT
    # =====================================================
    "transport": [
        "transport", "grab", "uber", "e-hailing",
        "taxi", "bus", "train", "lrt", "mrt",
        "monorail", "erls", "rapidkl",

        "toll", "petrol", "fuel", "diesel",
        "parking", "fare", "ticket",

        "touch n go", "tng", "smarttag",
        "car", "motor", "bike", "motorcycle",
    ],

    # =====================================================
    #                     SHOPPING / RETAIL
    # =====================================================
    "shopping": [
        "shopping", "mall", "retail",

        # Clothing
        "clothes", "clothing", "shoes", "fashion",
        "apparel", "jacket", "shirt", "pants",

        # Brands
        "uniqlo", "h&m", "zara", "cotton on",
        "puma", "nike", "adidas",

        # Online shopping
        "shopee", "lazada", "zalora", "shein",
        "temu", "aliexpress", "amazon",

        # Electronics
        "electronics", "gadget", "phone case",
        "earpods", "charger", "powerbank",
        "laptop", "keyboard", "mouse",
    ],

    # =====================================================
    #                        BILLS
    # =====================================================
    "bills": [
        "bill", "electric", "water", "wifi", "internet",
        "broadband", "tnb", "tm", "unifi",
        "celcom", "maxis", "digi", "umobile",

        "insurance", "insurance premium",
        "loan", "mortgage", "rent", "utilities",
        "ptptn", "tax", "quit rent",

        # Streaming
        "spotify", "netflix", "disney", "youtube premium",
        "apple music", "prime video",
    ],

    # =====================================================
    #                 ENTERTAINMENT & LEISURE
    # =====================================================
    "entertainment": [
        "movie", "cinema", "concert", "festival", "music",
        "game", "gaming", "steam", "ps5", "playstation",
        "xbox", "nintendo", "mobile legends", "pubg",

        "bowling", "karaoke", "zoo", "museum",
        "theme park", "funfair",

    ],

    # =====================================================
    #                      HEALTHCARE
    # =====================================================
    "healthcare": [
        "clinic", "hospital", "doctor", "pharmacy",
        "medicine", "prescription", "vitamin",
        "watsons", "guardian", "first aid",
        "covid test", "medical checkup",
    ],

    # =====================================================
    #                      EDUCATION
    # =====================================================
    "education": [
        "school", "tuition", "university",
        "course", "fees", "exam fees",
        "stationery", "book", "notebook",
        "textbook", "pen", "pencil",
    ],

    # =====================================================
    #                       BANKING
    # =====================================================
    "banking": [
        "bank fee", "service charge", "processing fee",
        "atm fee", "withdrawal fee", "transfer fee",
        "exchange fee", "foreign exchange fee",
    ],

    # =====================================================
    #                    PERSONAL CARE
    # =====================================================
    "personal_care": [
        "salon", "haircut", "barber", "spa",
        "massage", "skincare", "soap",
        "shampoo", "perfume", "makeup",
    ],

    # =====================================================
    #                         PETS
    # =====================================================
    "pets": [
        "pet food", "cat food", "dog food",
        "vet", "pet shop", "grooming",
        "cat litter", "pet accessories",
    ],

    # =====================================================
    #                      HOME SUPPLIES
    # =====================================================
    "home": [
        "ikea", "furniture", "home decor",
        "cleaning supplies", "detergent",
        "dish soap", "broom", "mop",
    ],
    "income": [
    "salary", "gaji", "gift", "bonus", "allowance", "commission",
    "rebate", "refund", "cashback", "bank in", "duit masuk",
    "deposit", "earned", "interest", "dividend", "profit",
    "sell", "sold", "payment received", 
    "parent", "parents"
    ],


    # =====================================================
    #                     OTHER / UNKNOWN
    # =====================================================
    "other": [
        "other", "misc", "miscellaneous",
        "uncategorized"
    ]
}


CATEGORY_KEYWORDS_FLAT = {cat: kws for cat, kws in CATEGORIES.items()}



def lemmatize(text: str) -> List[str]:
    """Return lemmas for tokens in the text (lowercased)."""
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if not token.is_punct and not token.is_space]


def _normalize_thousands_commas(text: str) -> str:
    """
    Remove commas that are thousands separators inside numbers while keeping
    commas used as separators for sentences.
    E.g. '1,234.56' -> '1234.56'
    """
    return re.sub(r'(?<=\d),(?=\d)', '', text)



def extract_amount(text: str) -> List[float]:

    if not text or not text.strip():
        return []

    t = text
    t = t.replace('₨', ' RM ').replace('MYR', ' RM ').replace('myr', ' RM ').replace('$', ' RM ')
    t = t.lower()
    t = _normalize_thousands_commas(t)

    matches = []

    for m in re.finditer(r'rm\s*([+-]?\d+(?:\.\d{1,2})?)', t, flags=re.IGNORECASE):
        try:
            matches.append((m.start(), float(m.group(1))))
        except ValueError:
            pass

    for m in re.finditer(r'(?<![\w.])([+-]?\d+(?:\.\d{1,2})?)', t):
        start = m.start()

        if any(abs(start - p) < 5 for p, _ in matches):
            continue

        try:
            matches.append((start, float(m.group(1))))
        except ValueError:
            pass

    matches.sort(key=lambda x: x[0])

    seen = set()
    result = []
    for _, val in matches:
        if val not in seen:
            result.append(val)
            seen.add(val)

    return result


def detect_transaction_type(text: str) -> str:
    """Detect whether the text indicates 'income', 'expense', or 'unknown'."""
    lemmas = lemmatize(text)
    txt = text.lower()

    # FIX: If sentence says "give/gave me" → INCOME
    if re.search(r"\b(giv(?:e|en|ing)|gave)\b.*\bme\b", txt):
        return "income"

    # FIX: "from friend/parents" → income
    if "from" in txt and any(word in txt for word in ["friend", "parents", "parent", "dad", "mum", "mom", "bro", "sis"]):
        return "income"

    # If I am the giver → EXPENSE
    if re.search(r"\bi\b.*\b(giv(?:e|en|ing)|gave)\b", txt):
        return "expense"

    # If "give/gave" + target person → EXPENSE
    # Someone else gave (no "I") → INCOME
    if re.search(r"\b(dad|mom|mum|parent|parents|friend|bro|sis|brother|sister)\b.*\b(giv(?:e|en|ing)|gave)\b", txt) \
    and not re.search(r"\bi\b", txt):
        return "income"

    
    if any(word in txt for word in ["spent", "spend"]):
        return "expense"


    for tok in re.findall(r"[A-Za-z]+", txt):
        fuzzy_give = process.extractOne(tok, ["give", "gave", "given", "giving"], scorer=fuzz.ratio)
        if fuzzy_give and fuzzy_give[1] >= 80:

            # Fuzzy "give me" → INCOME
            if "me" in txt:
                return "income"

            # Fuzzy "I give" → EXPENSE
            if re.search(r"\bi\b", txt):
                return "expense"

            # Fuzzy "give" + target → EXPENSE
            if any(person in txt for person in ["parent", "parents", "mum", "mom", "dad", "friend", "bro", "sis", "brother", "sister"]):
                return "expense"


    verb_tokens = re.findall(r"[A-Za-z]+", txt)

    # Fuzzy income verbs
    for tok in verb_tokens:
        match = process.extractOne(tok, INCOME_KEYWORDS, scorer=fuzz.ratio)
        if match and match[1] >= 80:
            return "income"

    # Fuzzy expense verbs
    for tok in verb_tokens:
        match = process.extractOne(tok, EXPENSE_KEYWORDS, scorer=fuzz.ratio)
        if match and match[1] >= 80:
            return "expense"


    has_expense = any(w in lemmas or re.search(rf"\b{re.escape(w)}\b", txt) for w in EXPENSE_KEYWORDS)
    has_income = any(w in lemmas or re.search(rf"\b{re.escape(w)}\b", txt) for w in INCOME_KEYWORDS)

    if has_expense and has_income:
        # Prefer expense when explicit expense verbs present
        if any(re.search(rf"\b{re.escape(w)}\b", txt) for w in ["pay", "paid", "spend", "spent", "buy", "bought", "gave"]):
            return "expense"
        return "income"

    if has_expense:
        return "expense"
    if has_income:
        return "income"

    return "unknown"


def match_category(token: str) -> str:
    """
    Fuzzy-match a token against known category keywords.
    Returns category name if confident, else 'other'.
    """
    token = token.lower()
    best_cat = "other"
    best_score = 0
    for cat, keywords in CATEGORY_KEYWORDS_FLAT.items():
        match = process.extractOne(token, keywords, scorer=fuzz.ratio)
        if match:
            _, score, _ = match
            if score > best_score and score >= 78:
                best_score = score
                best_cat = cat
    return best_cat


def detect_category(text: str, transaction_type: str) -> str:
    """
    Determine a category by trying:
    - direct lemma/key matching
    - regex word boundaries
    - fuzzy token matching fallback
    """
    text_lemmas = lemmatize(text)
    txt = text.lower()

    # Prefer income categories for income messages
    if transaction_type == "income":
        possible = {"income": CATEGORIES["income"]}
    else:
        possible = {k: v for k, v in CATEGORIES.items() if k != "income"}

    # Exact or lemma-based match
    for cat, keys in possible.items():
        for k in keys:
            if k in text_lemmas or re.search(rf"\b{re.escape(k)}\b", txt):
                return cat

    # Fuzzy fallback: check each token
    tokens = re.findall(r"[A-Za-z0-9]+" , txt)
    for tok in tokens:
        cat = match_category(tok)
        if cat != "other":
            return cat

    # Default
    return "income" if transaction_type == "income" else "other"


# ---------------- Public API ----------------

def analyze_single_message(text: str) -> List[Dict[str, Any]]:
    """
    Analyze a single short message and return a list of one-or-more
    transaction dicts: {type, amount, category}
    """
    ttype = detect_transaction_type(text)
    amounts = extract_amount(text)
    category = detect_category(text, ttype)

    if ttype == "unknown" and amounts:
        if category == "income":
            ttype = "income"
        else:
            ttype = "expense"

    if not amounts:
        return [{"type": ttype, "amount": None, "category": category}]

    return [{"type": ttype, "amount": float(a), "category": category} for a in amounts]


def analyze_message(text: str) -> List[Dict[str, Any]]:
    """
    Split possibly multi-transaction text into parts, analyze each, and
    return a concatenated list of transaction dicts.
    Splitting logic tries to split on ';', ' and ', ' & ', ' also ', newlines and commas after numbers.
    """
    if not text or not text.strip():
        return []

    separators = r'(?:\s*;\s*|\s+\band\b\s+|\s+\&\s+|\s+\balso\b\s+|\n|(?<=\d),\s+| \/\s+)'
    parts = [p.strip() for p in re.split(separators, text, flags=re.IGNORECASE) if p.strip()]

    results = []
    for p in parts:
        results.extend(analyze_single_message(p))
    return results



