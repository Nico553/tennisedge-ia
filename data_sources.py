"""Obtención voluntaria de históricos para investigación privada no comercial.

Jeff Sackmann/Tennis Abstract CC BY-NC-SA 4.0. Redistribuir solo respetando licencia.
Este módulo NUNCA consulta ni automatiza Betano.
"""
from __future__ import annotations
from io import BytesIO
from datetime import datetime
import requests
import pandas as pd

BASE = 'https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main'
ATTRIBUTION = 'Jeff Sackmann / Tennis Abstract — CC BY-NC-SA 4.0, archivo de Aneeshers (copia histórica de junio de 2026).'
ORIGINAL = 'https://github.com/Aneeshers/tennis-sackmann-archive'


def fetch_season(tour: str, year: int, timeout: int = 22) -> tuple[pd.DataFrame, dict]:
    """Lee un año del archivo histórico y devuelve procedencia explícita."""
    tour = tour.lower()
    if tour not in ('atp','wta') or not (2000 <= int(year) <= 2026):
        raise ValueError('Circuito o año no permitido')
    path=f'{tour}/{tour}_matches_{year}.csv'
    url=f'{BASE}/{path}'
    response=requests.get(url,timeout=timeout,headers={'User-Agent':'TennisEdge-Research/0.3 (non-commercial attribution in UI)'})
    if response.status_code == 404:
        raise ValueError(f'Archivo no publicado: {tour.upper()} {year}')
    response.raise_for_status()
    content=response.content
    if len(content) > 8_000_000:
        raise ValueError('Archivo demasiado grande')
    df=pd.read_csv(BytesIO(content),low_memory=False)
    return df, {'tour':tour.upper(),'year':year,'source':url,'rows':len(df), 'retrieved_utc':datetime.now().isoformat(timespec='seconds')}


def collect_years(years: list[int], tours=('ATP','WTA')) -> tuple[dict, list[dict]]:
    datasets={}; log=[]
    for tour in tours:
        pieces=[]
        for year in years:
            try:
                df, meta=fetch_season(tour,year)
                pieces.append(df)
                log.append({**meta,'status':'ok'})
            except Exception as exc:
                log.append({'tour':tour,'year':year,'status':'error','message':str(exc)[:130]})
        if pieces:
            datasets[tour]=pd.concat(pieces,ignore_index=True)
    return datasets,log
