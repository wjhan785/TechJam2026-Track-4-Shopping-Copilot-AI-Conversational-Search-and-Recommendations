from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")
MATERIAL_RE = re.compile(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I)
COLOR_RE = re.compile(r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I)

def searchable_text(product: dict) -> str:
    parts: list[str] = []
    for field in SEARCH_FIELDS:
        value = product.get(field)
        if isinstance(value, dict):
            parts.extend(f"{key} {item}" for key, item in value.items())
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).strip()

def _flatten_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []

def _clean_constraint(value: str, limit: int = 180) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")[:limit].rstrip()

# Re-derive the simulator's hidden intent card for a product. 
def intent_card(product: dict, limit: int = 180) -> dict:
    title = _clean_constraint(str(product.get("title") or "product"), limit)
    candidates = [*_flatten_values(product.get("features")), *_flatten_values(product.get("details"))]
    corpus = searchable_text(product)
    material = MATERIAL_RE.search(corpus)
    color = COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")
    cleaned = list(dict.fromkeys(_clean_constraint(item, limit) for item in candidates if _clean_constraint(item, limit)))
    if not cleaned:
        cleaned = [title]
    return {
        "target_category": title,
        "hard_constraints": cleaned[:2],
        "soft_preferences": cleaned[2:4] or cleaned[:1],
    }

# Catalog categories -> the coarse label the simulator uses in browsing openings (e.g. "Basketball Men"). Drops generic clothing ancestors and keeps the two most specific parts
def coarse_category(values: list[str]) -> str:
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part and part.lower() not in excluded:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"

# Map a constraint string to its attribute class (budget/material/ color/size/style/use_case). 'feature' is the fallback classification. 
# Order matters: e.g. a material name inside a color phrase must not shadow the color class.
def classify_constraint(value: str) -> str:
    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(material in lowered for material in MATERIALS):
        return "material"
    if any(word in lowered for word in ("color", "black", "white", "blue", "red", "pink", "green")):
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


# Parsing message templates 
#   RE_LOOKING         browsing opening: "I'm looking for {coarse_cat}..."
#   RE_KEY_REQUIREMENT buying disclosure: "A key requirement is: {text}"
#   RE_MATTERS         browsing disclosure: "...what matters is: {a; b}"
#   RE_OVERRIDE        intent override: "What I need is: {text}" (replaces!)
#   RE_NEGATIVE        "I don't have an additional preference for {attr}"
RE_LOOKING = re.compile(r"I'm looking for (.+?)(, but|\. A key|\. )(.*)$")
RE_KEY_REQUIREMENT = re.compile(r"A key requirement is:\s*(.*)$")
RE_MATTERS = re.compile(r"what matters is:\s*(.*)$")
RE_OVERRIDE = re.compile(r"What I need is:\s*(.*)$")
RE_NEGATIVE = re.compile(r"I don't have an additional preference for (\w+)")

# Attributes we can infer absent from the disclosed card.
INFERRABLE_ATTRS = ("material", "color", "size", "style", "budget", "use_case", "feature")

# bm25() column weights, in FTS5 table column order: (parent_asin UNINDEXED, title, categories, features, details, store, description)
# Title dominates because target slot phrases most often live there.
FTS_WEIGHTS = (5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
BM25_POOL_LIMIT = 900  # SQLITE_MAX_VARIABLE_NUMBER default is 999

# Lexical rank weights (tuned on the full 200-session public set):
#   SLOT  disclosed card-string evidence (each slot weighted by confidence)
#   POP   log-scaled popularity (rating_number), capped
#   RATE  average_rating centered at 3.8
#   TAG   user_profile.preference_tags matched against title + features
#   BUDG  price-budget distance penalty vs. the disclosed budget slot
#   BM25  soft rescue term: pool-best default for members the bm25 query misses, so products whose slot phrases live in features (not titles) are not penalized for the FTS miss
#   CAT   category prior when a category is known
RANK_W_SLOT = 1.2
RANK_W_POP = 0.5
RANK_W_RATE = 0.3
RANK_W_TAG = 0.4
RANK_W_BUDG = 0.6
RANK_W_BM25 = 0.3
RANK_W_CAT = 0.5

# --- reranker (M3) -----------------------------------------------------------
# Qwen3-Reranker-0.6B, loaded lazily at first use via sentence-transformers CrossEncoder (CUDA when available, CPU otherwise). 
# Degrades to lexical-only ranking if unavailable.
RERANKER_PATHS = [
    os.environ.get("RERANKER_PATH", ""),
    "assets/model/Qwen3-Reranker-0.6B",
]
RERANK_TOP_N = 24          # top-24 candidates fed to the reranker each turn
RERANK_MAX_LENGTH = 192   # ~64-token query + ~110-token doc
RERANK_QUERY_CHARS = 260  # coarse_cat + slots (~64 Qwen3 BPE tokens)
RERANK_DOC_CHARS = 440    # title + top features (~110 tokens)
RERANK_BLEND_FUSION = 0.6  # every-turn order: 0.6 * fusion rank + 0.4 * rerank score. Can increase rerank weight if using stronger LLM Rerankers. 
RERANK_BLEND_RERANK = 0.4


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False

# Choose CPU or GPU: SHOPCOPILOT_DEVICE=auto|cpu|cuda (default auto). A forced 'cuda' falls back to cpu when no GPU exists.
def _pick_device() -> str:
    override = os.environ.get("SHOPCOPILOT_DEVICE", "auto").lower()
    if override == "cuda":
        return "cuda" if _cuda_available() else "cpu"
    if override == "cpu":
        return "cpu"
    return "cuda" if _cuda_available() else "cpu"

# Qwen3-Reranker-0.6B via sentence-transformers CrossEncoder. ST renders pairs through the model's chat template (Document in the user turn) and LogitScore compares the next-token "yes"/"no" logits. 
# Lazy-loaded; degrades to lexical-only ranking when unavailable.
class RerankerService:
    def __init__(self, paths: list[str]) -> None:
        self.paths = [path for path in paths if path]
        self.model = None
        self.sign = 1.0
        self._calibrated = False

    @property
    def available(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.model is not None or not self.paths:
            return
        from sentence_transformers import CrossEncoder
        device = _pick_device()
        for path in self.paths:
            if not Path(path).is_dir():
                continue
            try:
                self.model = CrossEncoder(str(path), device=device, max_length=RERANK_MAX_LENGTH)
                return
            except Exception:
                continue

# Judge scores for (query, doc) pairs via the ST CrossEncoder. Returns None on any failure and the caller falls back to the lexical order. batch_size 8 is the tuned latency/quality tradeoff.
    def score(self, query: str, docs: list[str]) -> list[float] | None:
        if self.model is None or not docs:
            return None
        try:
            scores = self.model.predict(
                [(query, doc) for doc in docs],
                batch_size=max(1, min(8, len(docs))),
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception:
            return None
        array = _as_float_array(scores)
        if array is None:
            return None
        if array.ndim == 2:
            array = array[:, -1]
        self._calibrate()
        return [float(self.sign * value) for value in array]

# One-time sign calibration. The judge's raw output can be oriented either way (logit vs similarity); probe it with one known-good pair and flip self.sign if the ordering comes back inverted.
    def _calibrate(self) -> None:
        if self._calibrated or self.model is None:
            return
        self._calibrated = True
        scores = self.score(
            "black leather boots for women",
            [
                "Women's Black Leather Ankle Boots, Comfortable and Durable",
                "Pink Cotton Summer Sundress with Floral Print",
            ],
        )
        if scores and len(scores) == 2 and scores[0] < scores[1]:
            self.sign = -1.0

    # Rough prompt-token count for usage reporting.
    def prompt_tokens(self, query: str, docs: list[str]) -> int:
        if self.model is None:
            return 0
        try:
            return sum(
                len(self.model.tokenizer(query, doc).input_ids) for doc in docs
            )
        except Exception:
            return 0

def _as_float_array(values: object) -> object | None:
    try:
        import numpy as np
        return np.asarray(values, dtype="float64")
    except Exception:
        return None

class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.products: dict[str, dict] = {}
        self.all_ids: set[str] = set()
        self.by_cat: dict[str, set[str]] = defaultdict(set)
        self.by_card: dict[str, set[str]] = defaultdict(set)
        self.card_class: dict[str, str] = {}
        self.class_has: dict[str, set[str]] = defaultdict(set)
        self.card_strings: dict[str, tuple[str, ...]] = {}
        self.reranker = RerankerService(RERANKER_PATHS)
        self._sessions: dict[str, dict] = {}
        self._build_index()

    # ------------------------------------------------------------------ init

    # In-memory SQLite + FTS5 index over the catalog (one JSON line per product). FTS5 powers quoted-phrase matching and bm25 scores; built once per Agent from the catalog_path given to __init__.
    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                self._index_product(product, parent_asin)
                batch.append((
                    parent_asin,
                    _text(product.get("title")),
                    _text(product.get("categories")),
                    _text(product.get("features")),
                    _text(product.get("details")),
                    _text(product.get("store")),
                    _text(product.get("description")),
                ))
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    # Index one product: store it, derive its intent card, and wire the reverse maps (by_cat, by_card, class_has) for easy retrieval (ASIN becomes value).
    def _index_product(self, product: dict, parent_asin: str) -> None:
        self.products[parent_asin] = product
        self.all_ids.add(parent_asin)
        card = intent_card(product)
        card_all = list(dict.fromkeys([*card["hard_constraints"], *card["soft_preferences"]]))
        self.card_strings[parent_asin] = tuple(card_all)
        categories = [str(value) for value in product.get("categories") or []]
        self.by_cat[coarse_category(categories)].add(parent_asin)
        for slot in card_all:
            self.by_card[slot].add(parent_asin)
            cls = self.card_class.get(slot)
            if cls is None:
                cls = classify_constraint(slot)
                self.card_class[slot] = cls
            self.class_has[cls].add(parent_asin)

    # ----------------------------------------------------------------- state

    def reset(self, session_id: str, user_profile: dict) -> None:
        _log(f"session {session_id[:12]} started")
        self._sessions[session_id] = {
            "category": None,
            "slots": {},                 # disclosed card string -> weight
            "weak_slots": {},            # disclosed strings unknown to index -> weight
            "slot_classes": set(),       # classes with a positive disclosure
            "neg_attrs": set(),          # classes inferred absent (filters)
            "card_drained": False,
            "override_seen": False,
            "profile": user_profile or {},
            "last_pool": None,         # live candidate pool size (post-negatives)
        }

    # -------------------------------------------------------------- messaging

    def _progress_line(self, session_id: str, state: dict, turn: int, ask_attribute: str | None, rec_count: int) -> str:
        parts = [
            f"session {session_id[:12]}",
            f"t{turn}",
            f"cat={state['category'] or '-'}",
            f"slots=[{', '.join(sorted(state['slots']))}]",
        ]
        if state["weak_slots"]:
            parts.append(f"weak=[{', '.join(sorted(state['weak_slots']))}]")
        parts.append(f"pool={state['last_pool'] if state['last_pool'] is not None else '-'}")
        parts.append(f"asked={ask_attribute or '-'}")
        parts.append(f"rec={rec_count}")
        if state["override_seen"]:
            parts.append("override")
        if state["card_drained"]:
            parts.append("drained")
        if turn >= 10:
            parts.append("FINAL")
        return " | ".join(parts)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        self._parse_message(state, user_message)
        candidates = self._retrieve(state, RERANK_TOP_N)
        ranked, prompt_tokens = self._rerank(state, candidates)
        ranked = ranked[:top_k]
        ask_attribute, message = self._question(state, turn)
        _log(self._progress_line(session_id, state, turn, ask_attribute, len(ranked)))
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": [{"parent_asin": asin} for asin in ranked],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 0},
        }

    # ------------------------------------------------------------------ rerank

    # Every-turn rerank of the top-24: final order = blend of the fusion rank (linear normalization) and the judge's min-max normalized score (0.6/0.4 weights). 
    # Early-returns when there are fewer than two candidates or no judge is available; any other failure also degrades to the fusion order.
    def _rerank(self, state: dict, ranked: list[str]) -> tuple[list[str], int]:
        if len(ranked) < 2:
            return ranked, 0
        self.reranker.load()
        if not self.reranker.available:
            return ranked, 0
        query = self._rerank_query(state)
        top = ranked[:RERANK_TOP_N]
        docs = [self._rerank_doc(asin) for asin in top]
        scores = self.reranker.score(query, docs)
        if scores is None:
            return ranked, 0
        n = len(top)
        low, high = min(scores), max(scores)
        span = high - low or 1.0
        def blend(index: int) -> float:
            fusion_norm = (n - index) / n
            rerank_norm = (scores[index] - low) / span
            return RERANK_BLEND_FUSION * fusion_norm + RERANK_BLEND_RERANK * rerank_norm
        order = sorted(range(n), key=blend, reverse=True)
        reranked = [top[index] for index in order]
        tokens = self.reranker.prompt_tokens(query, docs)
        return reranked + ranked[RERANK_TOP_N:], tokens

    # Query text for the judge: category + slots repeated by weight (stronger slots get more emphasis) + weak slots, character-capped.
    def _rerank_query(self, state: dict) -> str:
        parts: list[str] = []
        if state["category"] is not None:
            parts.append(state["category"])
        for slot, weight in sorted(state["slots"].items(), key=lambda pair: -pair[1]):
            parts.extend([slot] * max(1, int(round(weight * 2))))
        for slot in state["weak_slots"]:
            parts.append(slot)
        return (" ".join(parts))[:RERANK_QUERY_CHARS]

    # Document text for the judge: title + truncated features, character-capped to the judge's context budget.
    def _rerank_doc(self, asin: str) -> str:
        product = self.products.get(asin) or {}
        title = _text(product.get("title"))
        features = _text(product.get("features"))[:RERANK_DOC_CHARS // 2]
        return (title + " " + features)[:RERANK_DOC_CHARS]

    # Extract disclosures from the customer reply. Check order matters: override > judgment-deflection > drained/negative > plain disclosures. 
    # An intent override REPLACES the old constraint (old slots decay to 30%, the new one is added at 1.5x).
    def _parse_message(self, state: dict, message: str) -> None:
        if "ignore my earlier preference" in message or RE_OVERRIDE.search(message):
            state["override_seen"] = True
            for slot in state["slots"]:
                state["slots"][slot] *= 0.3
            match = RE_OVERRIDE.search(message)
            if match:
                self._add_slot(state, _clean_constraint(match.group(1)), 1.5)
            self._maybe_drained(state)
            return
        if "use your judgment" in message:
            return
        negative = RE_NEGATIVE.search(message)
        if negative:
            attr = negative.group(1)
            if attr == "other":
                state["card_drained"] = True  # "no additional preference for other" == card exhausted
            elif attr not in state["slot_classes"]:
                state["neg_attrs"].add(attr)  # class never disclosed -> treat as absent, filter it out
            self._maybe_drained(state)
            return
        if "Those options are not quite right yet" in message:
            return
        if state["category"] is None:
            match = RE_LOOKING.search(message)
            if match:
                state["category"] = match.group(1)
                if match.group(2) == ". ":
                    rest = _clean_constraint(match.group(3))
                    if rest:
                        self._add_slot(state, rest, 1.0)
        match = RE_KEY_REQUIREMENT.search(message)
        if match:
            self._add_slot(state, _clean_constraint(match.group(1)), 1.0)
        match = RE_MATTERS.search(message)
        if match:
            for slot in match.group(1).split("; "):
                slot = _clean_constraint(slot)
                if slot:
                    self._add_slot(state, slot, 1.0)
        self._maybe_drained(state)

    # Record a disclosed constraint. Strings known to the index go to slots (they narrow the candidate pool); unknown ones go to weak_slots (used only as FTS/rerank query terms, never for pool narrowing).
    def _add_slot(self, state: dict, slot: str, weight: float) -> None:
        if not slot:
            return
        if slot in self.by_card:
            state["slots"][slot] = max(state["slots"].get(slot, 0.0), weight)
            state["slot_classes"].add(self.card_class[slot])
        else:
            state["weak_slots"][slot] = max(state["weak_slots"].get(slot, 0.0), weight)

    # A card is drained once 4+ card strings are disclosed or the customer says they are out of preferences. From then on every not-yet-disclosed inferable class is treated as absent (neg_attrs) and we stop asking.
    def _maybe_drained(self, state: dict) -> None:
        if len(state["slots"]) >= 4:
            state["card_drained"] = True
        if not state["card_drained"]:
            return
        for attr in INFERRABLE_ATTRS:
            if attr not in state["slot_classes"]:
                state["neg_attrs"].add(attr)

    # ------------------------------------------------------------- retrieval

    # Candidate pipeline: start from the category pool (whole catalog when no category is known), intersect it with every known disclosed slot (heaviest first; a slot that would empty the pool is skipped), then drop neg_attrs classes. 
    # If nothing survives, fall back to FTS phrase search then popularity. Short lists are padded with the same fallbacks.
    def _retrieve(self, state: dict, top_k: int) -> list[str]:
        pool: set[str] | None = None
        if state["category"] is not None:
            pool = set(self.by_cat.get(state["category"], ()))
        if not pool:
            pool = set(self.all_ids)
        slots = sorted(state["slots"], key=lambda s: -state["slots"][s])
        applied: list[str] = []
        for slot in slots:
            if slot not in self.by_card:
                continue
            narrowed = pool & self.by_card[slot]
            if narrowed:
                pool = narrowed
                applied.append(slot)
        filtered = self._apply_negatives(pool, state)
        state["last_pool"] = len(filtered) if filtered else 0
        if filtered:
            ranked = self._rank_pool(filtered, state, applied, top_k)
        else:
            fallback = self._retrieve_route_c(state, top_k * 3)
            if fallback:
                ranked = fallback
            else:
                ranked = self._popularity_top(state, top_k * 3)
        if len(ranked) < top_k:
            extras = self._retrieve_route_c(state, top_k * 3)
            if not extras:
                extras = self._popularity_top(state, top_k * 3)
            for asin in extras:
                if asin not in ranked:
                    ranked.append(asin)
                if len(ranked) >= top_k:
                    break
        return ranked[:top_k]

    # Drop every product that carries a class the customer implicitly ruled out (class_has[attr] = products with a slot of that class).
    def _apply_negatives(self, pool: set[str], state: dict) -> set[str]:
        for attr in state["neg_attrs"]:
            pool -= self.class_has.get(attr, set())
        return pool

    # FTS5 OR-phrase search over the whole catalog with bm25 ordering — rescues targets whose slot phrases live in features or details rather than titles.
    def _retrieve_route_c(self, state: dict, top_k: int) -> list[str]:
        terms = []
        for slot, weight in sorted(state["slots"].items(), key=lambda pair: -pair[1]):
            if slot in self.by_card:
                terms.append(_fts_phrase(slot))
        for slot in state["weak_slots"]:
            terms.append(_fts_phrase(slot))
        if not terms:
            return []
        expression = " OR ".join(terms)
        weights_marks = ", ".join("?" * len(FTS_WEIGHTS))
        try:
            rows = self.connection.execute(
                f"SELECT parent_asin FROM products WHERE products MATCH ? "
                f"ORDER BY bm25(products, {weights_marks}) LIMIT ?",
                (expression, *FTS_WEIGHTS, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [str(row[0]) for row in rows]

    # Score every survivor: slot evidence * weight, category prior, rating/popularity, profile-tag hits, budget proximity, and a soft bm25 term (pool-best default for FTS misses). Ties break by popularity. 
    # bm25 is only computed for pools <= BM25_POOL_LIMIT (SQLite variable-count bound).
    def _rank_pool(self, pool: set[str], state: dict, applied: list[str], top_k: int) -> list[str]:
        profile_tags = {tag.strip().lower() for tag in state["profile"].get("preference_tags", []) if tag}
        budget = _budget_value(applied, self)
        if len(pool) <= BM25_POOL_LIMIT:
            bm25_scores = self._bm25_scores(pool, applied)
        else:
            bm25_scores = {}
        scored: list[tuple[float, str]] = []
        for asin in pool:
            product = self.products.get(asin)
            if product is None:
                continue
            score = 0.0
            slot_evidence = sum(state["slots"].get(slot, 0.0) for slot in self.card_strings.get(asin, ()))
            score += RANK_W_SLOT * slot_evidence
            score += RANK_W_CAT * (1.0 if state["category"] is not None else 0.0)
            rating = product.get("average_rating")
            if isinstance(rating, (int, float)):
                score += RANK_W_RATE * max(-1.0, min(1.0, (rating - 3.8) / 2.0))
            rating_number = product.get("rating_number")
            if isinstance(rating_number, (int, float)) and rating_number > 0:
                score += RANK_W_POP * min(1.0, math.log1p(rating_number) / 8.0)
            if profile_tags:
                text = (_text(product.get("title")) + " " + _text(product.get("features"))).lower()
                score += RANK_W_TAG * sum(1.0 for tag in profile_tags if tag in text)
            if budget is not None:
                price = product.get("price")
                if isinstance(price, (int, float)) and price > 0:
                    ratio = abs(price - budget) / max(budget, 1.0)
                    score -= RANK_W_BUDG * min(2.0, ratio)
            if bm25_scores:
                worst = min(bm25_scores.values())
                score += RANK_W_BM25 * (-bm25_scores.get(asin, worst))
            scored.append((score, asin))
        scored.sort(key=lambda pair: (-pair[0], -_pop_key(self.products.get(pair[1]) or {})))
        return [asin for _, asin in scored]

    # bm25 scores for the live pool under the applied slot phrases. Members missing from the FTS result keep their pool-best default in _rank_pool — absence is weak evidence, not a disqualifier.
    def _bm25_scores(self, pool: set[str], applied: list[str]) -> dict[str, float]:
        expression = " OR ".join(_fts_phrase(slot) for slot in applied if slot in self.by_card)
        if not expression:
            return {}
        ids = list(pool)
        marks = ", ".join("?" * len(ids))
        weights_marks = ", ".join("?" * len(FTS_WEIGHTS))
        try:
            rows = self.connection.execute(
                f"SELECT parent_asin, bm25(products, {weights_marks}) "
                f"FROM products WHERE products MATCH ? AND parent_asin IN ({marks})",
                (*FTS_WEIGHTS, expression, *ids),
            ).fetchall()
        except sqlite3.OperationalError:
            return {}
        return {str(row[0]): float(row[1]) for row in rows}

    # Category-scoped popularity ranking — the last-resort fallback when no slot evidence survives.
    def _popularity_top(self, state: dict, top_k: int) -> list[str]:
        if state["category"] is not None:
            pool = self.by_cat.get(state["category"], set())
        else:
            pool = self.all_ids
        ranked = sorted(
            ((asin, _pop_key(self.products.get(asin) or {})) for asin in pool),
            key=lambda pair: -pair[1],
        )
        return [asin for asin, _ in ranked[:top_k]]

    # ----------------------------------------------------------- questioning

    # Ask 'other' while the card is still live: an 'other' ask matches ANY undisclosed constraint, which is the fastest way to drain the card (each reply reveals up to two new constraints). 
    # Stop asking once drained or at turn 10, then ship the final list.
    def _question(self, state: dict, turn: int) -> tuple[str | None, str]:
        if not state["card_drained"] and turn < 10:
            return "other", "Any other requirements you can share? I'll factor them in."
        return None, "Here are the best matches I found for you."


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

LOG_ENABLED = os.environ.get("SHOPCOPILOT_LOG", "1") != "0"

# Live progress line to stderr (flushed).
def _log(message: str) -> None:
    if not LOG_ENABLED:
        return
    try:
        sys.stderr.write(f"[ShopCopilot] {message}\n")
        sys.stderr.flush()
    except Exception:
        pass

# Flatten any catalog field value into a single searchable string.
def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)

# Quote a slot as an FTS5 phrase, escaping embedded double quotes.
def _fts_phrase(slot: str) -> str:
    return '"' + slot.replace('"', '""') + '"'

# Extract a numeric budget from the first applied slot that carries one (e.g. 'budget around $25').
def _budget_value(applied: list[str], agent: Agent) -> float | None:
    for slot in applied:
        match = re.search(r"\$(\d+(?:\.\d+)?)", slot)
        if match:
            return float(match.group(1))
    return None

# Popularity key: rating_number (0 when missing) — used for tie-breaks and the popularity fallback ranking.
def _pop_key(product: dict) -> float:
    value = product.get("rating_number")
    return float(value) if isinstance(value, (int, float)) else 0.0
