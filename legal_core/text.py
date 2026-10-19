"""Chuẩn hoá chữ, tách âm tiết + bigram, khoá Điều — chép nguyên từ legal_eval.py của benchmark."""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

_WORD = re.compile(r"\w+")

def norm(text: str) -> str:
    """NFC, sửa 'Ð' (Eth) thành 'Đ', gộp khoảng trắng, chữ thường."""
    t = unicodedata.normalize("NFC", text or "")
    t = t.replace("Ð", "Đ").replace("ð", "đ")
    return re.sub(r"\s+", " ", t).strip().lower()


def syllables(text: str) -> list[str]:
    return _WORD.findall(norm(text))


def syllable_bigrams(text: str) -> list[str]:
    s = syllables(text)
    return s + [f"{a}_{b}" for a, b in zip(s, s[1:])]


def dieu_number(text: Optional[str]) -> Optional[str]:
    m = re.search(r"điều\s+(\d+[a-zđ]?)\b", norm(text or ""))
    return m.group(1) if m else None


def article_key(so_hieu: Optional[str], dieu: Optional[str]) -> Optional[str]:
    if not so_hieu or not dieu:
        return None
    return f"{so_hieu.lower()}|{dieu}"
