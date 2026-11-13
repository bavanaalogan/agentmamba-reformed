"""Question parsing: identifiers, dates, and a learned intent classifier.

* Identifiers are extracted with explicit, configurable patterns (plant IDs have fixed formats).
* Intent is predicted by a TF-IDF + logistic-regression classifier trained on TRAINING-split
  question texts and their ``question_type`` labels only (``train_intent_classifier``).
* Each intent maps to an ``AnswerSpec`` describing which record field answers it, which event
  types may carry that field, which identifier kind anchors it, and which temporal / version
  constraint applies. This is a declared, inspectable operationalization of the question
  grammar of the benchmark; it does not generalize to arbitrary free-text questions.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path

ID_PATTERNS: dict[str, str] = {
    "heat": r"\b[A-Z0-9]+-H\d{4}\b",
    "ladle": r"\b[A-Z0-9]+-LD\d{2}\b",
    "coil": r"\b[A-Z0-9]+-CL\d{4}\b",
    "slab": r"\b[A-Z0-9]+-SL\d{4}\b",
    "cast_seq": r"\b[A-Z0-9]+-CS\d{2}\b",
    "work_order": r"\b[A-Z0-9]+-WO\d{3}\b",
    "document": r"\b[A-Z0-9]+-SOP-[A-Z]+\b",
    "asset": r"\b[A-Z0-9]+-(?:(?:CC|HSM|BOF|BF|UT)\d|PLANT)\b",
}
DATE_RE = re.compile(r"(?<![\d-])(\d{4}-\d{2}-\d{2})(?![\d-])")   # also inside "2026-03-14T09:30"
# clock time stated in a question ("at 09:30", "2026-03-14 09:30", "2026-03-14T09:30")
TIME_RE = re.compile(r"(?<![\d:])([01]\d|2[0-3]):([0-5]\d)(?![\d])")
LISTING_RE = re.compile(r"\b(timeline|sequence of|list (?:all|the)|reconstruct)\b", re.I)
MEASUREMENT_LEXICON = {
    "tap temperature": "tap_temperature", "carbon": "carbon", "oxygen": "oxygen",
    "hydrogen": "hydrogen", "nitrogen": "nitrogen",
}
CAUSAL_RE = re.compile(r"\b(caus\w*|root[- ]cause|why|due to|responsible)\b", re.I)


@dataclass(frozen=True)
class AnswerSpec:
    answer_field: str                   # Event attribute or observation name
    event_types: tuple[str, ...]
    anchor_kinds: tuple[str, ...]
    constraint: str | None = None       # before_date | on_date | latest_version | latest_before_link
    single_valued: bool = True          # distinct values from distinct records => conflict
    causal: bool = False
    hypothesis_field: str | None = None
    hypothesis_types: tuple[str, ...] = ()
    confirm_field: tuple[str, str] | None = None   # (observation, required value) for causal claims


SPECS: dict[str, AnswerSpec] = {
    "measurement": AnswerSpec("<measurement>", ("bof_heat_record",), ("heat",)),
    "ladle_of_heat": AnswerSpec("ladle_id", ("ladle_assignment",), ("heat",)),
    "cast_seq_of_coil": AnswerSpec("cast_seq_id", ("cast_record",), ("coil",)),
    "ladle_warning": AnswerSpec("finding", ("ladle_inspection",), ("ladle",), "before_date", single_valued=False),
    "wo_outcome": AnswerSpec("outcome", ("maintenance_followup",), ("work_order",)),
    # ladles are maintained assets too, and their IDs parse as kind "ladle"
    "wo_opened_on": AnswerSpec("work_order_id", ("work_order",), ("asset", "ladle"), "on_date"),
    "ladle_thickness_at": AnswerSpec("lining_thickness", ("ladle_inspection",), ("ladle",), "on_date"),
    "procedure_current_value": AnswerSpec("limit_value", ("procedure_doc",), ("document",), "latest_version"),
    "coil_defect_cause": AnswerSpec("cause", ("rca_report",), ("coil",), causal=True, hypothesis_field="abnormality",
                                    hypothesis_types=("cast_record",), confirm_field=("rca_status", "confirmed")),
    "upstream_ladle_condition": AnswerSpec("finding", ("ladle_inspection",), ("coil",), "latest_before_link"),
    "caster_abnormality": AnswerSpec("abnormality", ("cast_record",), ("heat",)),
    # Category A (real UCI #851 daily summaries)
    "energy_peak_on_date": AnswerSpec("peak_usage", ("energy_daily_summary",), ("asset",), "on_date"),
    "energy_total_on_date": AnswerSpec("total_usage", ("energy_daily_summary",), ("asset",), "on_date"),
    "energy_load_type_on_date": AnswerSpec("dominant_load_type", ("energy_daily_summary",), ("asset",), "on_date"),
}


@dataclass
class ParsedQuery:
    text: str
    intent: str
    intent_confidence: float
    ids: dict[str, list[str]] = field(default_factory=dict)   # kind -> ids in order of appearance
    dates: list[str] = field(default_factory=list)
    measurement: str | None = None
    causal_language: bool = False

    resolved_from_context: dict[str, str] = field(default_factory=dict)   # id -> kind
    times: list[str] = field(default_factory=list)                        # "HH:MM"
    listing: bool = False             # asks for a timeline / list of records, not one value

    @property
    def intent_trusted(self) -> bool:
        return self.intent_confidence >= INTENT_MIN_CONFIDENCE

    @property
    def spec(self) -> AnswerSpec | None:
        """Typed answer spec, or None when the intent is ambiguous/unknown or not trusted
        (then only document-grounded evidence can answer)."""
        if not self.intent_trusted:
            return None
        base = SPECS.get(self.intent)
        if base and not any(self.ids.get(k) for k in base.anchor_kinds) and \
                any(ids for k, ids in self.ids.items() if k not in base.anchor_kinds):
            # The question names identifiers, but none of the kind this typed intent is about
            # (e.g. a pump ID under a work-order intent): answer it as an untyped question.
            return None
        if base and base.answer_field == "<measurement>":
            return AnswerSpec(self.measurement or "<unknown>", base.event_types, base.anchor_kinds)
        return base

    def anchor_map(self) -> dict[str, str]:
        """{identifier: kind} the answer must be about. Typed intents use identifiers of the
        spec's anchor kinds; if none were named, explicitly supplied identifiers of unknown kind
        still count (never ask for an identifier the engineer already gave). Without a typed
        spec, every named identifier is an anchor (document-grounded lookup)."""
        spec = self.spec
        if spec is None:
            return {i: k for k, ids in self.ids.items() for i in ids}
        out = {i: k for k in spec.anchor_kinds for i in self.ids.get(k, [])}
        if not out:
            out = {i: "identifier" for i in self.ids.get("identifier", [])}
        return out

    def anchors(self) -> list[str]:
        return list(self.anchor_map())

    def all_ids(self) -> list[str]:
        return [i for ids in self.ids.values() for i in ids]

    def focus_terms(self) -> list[str]:
        """Content words of the question (identifiers, dates, question words and generic
        record vocabulary removed), stemmed. A document passage must contain all of them."""
        text = TIME_RE.sub(" ", DATE_RE.sub(" ", self.text))
        for i in self.all_ids():
            text = text.replace(i, " ")
        words = re.findall(r"[a-zA-Z]+", text.lower())
        return list(dict.fromkeys(stem(w) for w in words if w not in FOCUS_STOPWORDS and len(w) > 2))

    def to_dict(self) -> dict:
        return {"intent": self.intent, "intent_confidence": round(self.intent_confidence, 4),
                "intent_trusted": self.intent_trusted, "ids": self.ids, "dates": self.dates,
                "measurement": self.measurement, "causal_language": self.causal_language,
                "resolved_from_context": self.resolved_from_context, "times": self.times,
                "listing": self.listing}


# Below this classifier confidence the typed AnswerSpec is not used. Every train/val/test question of
# both benchmarks parses at >= 0.783 (validation minimum 0.783), so 0.5 leaves them all unchanged.
INTENT_MIN_CONFIDENCE = 0.5

# Generic plant identifiers that the benchmark-specific patterns above do not cover, e.g.
# TEST-LADLE-007 or PLANT2-HEAT-1042: upper-case, hyphenated, at least one digit.
GENERIC_ID_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+\b")
ID_KEYWORDS = {"LADLE": "ladle", "LD": "ladle", "HEAT": "heat", "COIL": "coil", "CL": "coil", "SLAB": "slab",
               "WO": "work_order", "WORKORDER": "work_order", "SOP": "document", "DOC": "document",
               "CASTER": "asset", "MILL": "asset", "BOF": "asset", "FURNACE": "asset",
               "ALM": "alarm", "ALARM": "alarm", "TRIP": "trip", "FAULT": "fault", "INCIDENT": "incident"}
PRECEDING_KIND = {"ladle": "ladle", "heat": "heat", "coil": "coil", "slab": "slab", "order": "work_order",
                  "procedure": "document", "document": "document", "asset": "asset", "caster": "asset",
                  "mill": "asset", "converter": "asset", "furnace": "asset", "equipment": "asset",
                  "pump": "asset", "motor": "asset", "fan": "asset", "valve": "asset",
                  "compressor": "asset", "turbine": "asset", "boiler": "asset",
                  "alarm": "alarm", "trip": "trip", "fault": "fault", "incident": "incident"}
FOCUS_STOPWORDS = set("""what which who whom whose when where why how was were is are be been being did does do
the a an of for on in at to by with from and or this that these those it its any some there their was record
recorded records documented noted note reported logged entry ladle heat coil slab asset plant
according value pump motor fan valve compressor turbine boiler caster mill converter furnace machine device
happened happen occurred occur show shown reading readings measured measurement level equipment around after
before during time timestamp then also trigger triggered triggering""".split())
# The last line is generic question vocabulary for log questions ("what was the vibration reading
# around 09:30", "what happened ..."): words that name the *kind of question*, not its content.


def stem(word: str) -> str:
    w = word.lower()
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def extract_ids(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    spans = []
    for kind, pat in ID_PATTERNS.items():
        for m in re.finditer(pat, text):
            spans.append(m.span())
            out.setdefault(kind, [])
            if m.group(0) not in out[kind]:
                out[kind].append(m.group(0))
    for m in GENERIC_ID_RE.finditer(text):
        tok = m.group(0)
        if not any(ch.isdigit() for ch in tok) or any(a < m.end() and m.start() < b for a, b in spans):
            continue
        kind = next((ID_KEYWORDS[p] for p in tok.split("-") if p in ID_KEYWORDS), None)
        if kind is None:
            prev = re.findall(r"[a-zA-Z]+", text[: m.start()].lower())
            previous = prev[-1] if prev else ""
            if previous == "id" and len(prev) > 1:
                previous = prev[-2]
            kind = PRECEDING_KIND.get(previous, "identifier")
        out.setdefault(kind, [])
        if tok not in out[kind]:
            out[kind].append(tok)
    return out


def clarification_reason(parsed: ParsedQuery) -> str | None:
    """Do not silently choose one of several named entities or dates.

    Multi-entity comparison and date-range semantics are not implemented by the extractive
    verifier, so those requests need a narrower question rather than a guessed answer.
    """
    by_kind: dict[str, list[str]] = {}
    for ident, kind in parsed.anchor_map().items():
        by_kind.setdefault(kind, []).append(ident)
    for kind, ids in by_kind.items():
        if len(ids) > 1:
            return f"multiple {kind} identifiers are named ({', '.join(ids)}); specify one"
    if len(set(parsed.dates)) > 1:
        return f"multiple dates are named ({', '.join(parsed.dates)}); specify one date"
    return None


class IntentClassifier:
    """TF-IDF (word 1-2 grams) + multinomial logistic regression. IDs/dates are masked so the
    classifier learns question phrasing, not identifiers."""

    VERSION = "intent-v1"

    def __init__(self):
        self.pipe = None

    @staticmethod
    def _mask(text: str) -> str:
        for kind, pat in ID_PATTERNS.items():
            text = re.sub(pat, f" <{kind}> ", text)
        text = GENERIC_ID_RE.sub(lambda m: " <id> " if any(c.isdigit() for c in m.group(0)) else m.group(0), text)
        return DATE_RE.sub(" <date> ", text).lower()

    def fit(self, texts: list[str], labels: list[str]) -> "IntentClassifier":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        self.pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1, token_pattern=r"<?\w+>?"),
                                  LogisticRegression(max_iter=2000, C=10.0))
        self.pipe.fit([self._mask(t) for t in texts], labels)
        return self

    def predict(self, text: str) -> tuple[str, float]:
        if self.pipe is None:
            raise RuntimeError("intent classifier not trained/loaded")
        proba = self.pipe.predict_proba([self._mask(text)])[0]
        i = int(proba.argmax())
        return str(self.pipe.classes_[i]), float(proba[i])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"version": self.VERSION, "pipe": self.pipe}, fh)

    @classmethod
    def load(cls, path: Path) -> "IntentClassifier":
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        if blob.get("version") != cls.VERSION:
            raise ValueError(f"intent classifier version mismatch: {blob.get('version')}")
        obj = cls()
        obj.pipe = blob["pipe"]
        return obj


class QueryProcessor:
    def __init__(self, classifier: IntentClassifier):
        self.classifier = classifier

    def parse(self, text: str) -> ParsedQuery:
        intent, conf = self.classifier.predict(text)
        low = text.lower()
        measurement = next((v for k, v in MEASUREMENT_LEXICON.items() if k in low), None)
        return ParsedQuery(text=text, intent=intent, intent_confidence=conf, ids=extract_ids(text),
                           dates=DATE_RE.findall(text), measurement=measurement,
                           causal_language=bool(CAUSAL_RE.search(text)),
                           times=[f"{h}:{m}" for h, m in TIME_RE.findall(text)], listing=bool(LISTING_RE.search(text)))
