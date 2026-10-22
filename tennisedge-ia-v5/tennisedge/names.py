"""Normalización de nombres para cruzar fuentes (Sackmann, RapidAPI, The Odds API)."""
from __future__ import annotations

import re
import unicodedata

_SPECIAL = str.maketrans({'ł': 'l', 'ø': 'o', 'đ': 'd', 'ß': 'ss', 'æ': 'ae', 'œ': 'oe', 'ı': 'i'})


def normalize(name) -> str:
    value = unicodedata.normalize('NFKD', str(name or '').casefold().translate(_SPECIAL))
    value = ''.join(c for c in value if not unicodedata.combining(c))
    value = re.sub(r'[^a-z0-9]+', ' ', value)
    return ' '.join(value.split())


def key(name) -> str:
    """Clave independiente del orden ("Alcaraz Carlos" == "Carlos Alcaraz")."""
    tokens = normalize(name).split()
    return ' '.join(sorted(tokens))


class NameIndex:
    """Resuelve nombres externos a la clave interna sin coincidencias ambiguas."""

    def __init__(self, keys):
        self.keys = set(keys)
        self.by_surname: dict[str, list[str]] = {}
        for k in self.keys:
            for tok in k.split():
                if len(tok) > 2:
                    self.by_surname.setdefault(tok, []).append(k)

    def resolve(self, name) -> str | None:
        k = key(name)
        if not k:
            return None
        if k in self.keys:
            return k
        toks = normalize(name).split()
        if not toks:
            return None
        # "C. Alcaraz", "Alcaraz C." -> inicial + apellido único
        initials = [t for t in toks if len(t) == 1]
        longs = [t for t in toks if len(t) > 1]
        if longs:
            cands = None
            for t in longs:
                c = set(self.by_surname.get(t, []))
                cands = c if cands is None else cands & c
            cands = cands or set()
            if initials:
                cands = {c for c in cands if any(tok[0] == initials[0] for tok in c.split() if tok not in longs)}
            if len(cands) == 1:
                return next(iter(cands))
        return None
