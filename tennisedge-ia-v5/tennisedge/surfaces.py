"""Superficie y nivel a partir del nombre del torneo (para feeds sin ese dato, como The Odds API)."""
from __future__ import annotations

from .names import normalize

# Identificación conservadora por nombre de torneo; si no hay coincidencia única, no se adivina.
_SURFACE_RULES = {
    'Hard': ('australian open','us open','indian wells','miami open','miami masters',
             'cincinnati','canadian open','toronto','montreal','shanghai masters',
             'paris masters','paris indoor','vienna','basel','beijing','tokyo',
             'stockholm','rotterdam','doha','dubai','acapulco','washington','almaty',
             'brussels','european open','ningbo open','grand prix auvergne rhone alpes',
             'brisbane','auckland','adelaide','seoul','osaka','wuhan'),
    'Clay': ('french open','roland garros','monte carlo','madrid open','mutua madrid',
             'italian open','internazionali bnl','rome masters','rome open',
             'barcelona open','buenos aires','rio open','rio de janeiro',
             'hamburg open','estoril','bastad','umag','geneva','gstaad','bucharest'),
    'Grass': ('wimbledon','queens club','queen s club','cinch championships',
              'halle','s hertogenbosch','rosmalen','eastbourne','mallorca championships',
              'nottingham','bad homburg','berlin grass'),
}




_LEVEL_RULES = {
    'G': ('australian open', 'french open', 'roland garros', 'wimbledon', 'us open'),
    'M': ('indian wells', 'miami', 'monte carlo', 'madrid', 'italian open', 'rome', 'canadian open', 'toronto',
          'montreal', 'cincinnati', 'shanghai', 'paris masters', 'beijing', 'wuhan', 'doha', 'dubai'),
}


def resolve_surface(title: str, key: str = '') -> str | None:
    text = normalize(f'{title} {key}')
    found = {s for s, phrases in _SURFACE_RULES.items() if any(p in text for p in phrases)}
    return found.pop() if len(found) == 1 else None


def resolve_level(title: str, key: str = '') -> str:
    text = normalize(f'{title} {key}')
    if 'challenger' in text:
        return 'C'
    if 'itf' in text:
        return 'ITF'
    for lvl, phrases in _LEVEL_RULES.items():
        if any(p in text for p in phrases):
            return lvl
    return 'A'


def resolve_tour(key: str, title: str = '') -> str | None:
    k = f'{key} {title}'.lower()
    if any(x in k for x in ('doubles', 'mixed')):
        return None
    if 'wta' in k:
        return 'WTA'
    if 'atp' in k:
        return 'ATP'
    return None
