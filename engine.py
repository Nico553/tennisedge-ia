"""Motor estadístico de TennisEdge IA.

Elo cronológico (general + superficie), mercado de sets/juegos estimado mediante
partidos análogos ANTERIORES a la consulta. Uso experimental; no garantiza rendimiento.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math
import re

import numpy as np
import pandas as pd
from scipy.stats import beta

SURFACES = ("Hard", "Clay", "Grass", "Carpet")
SCORES = re.compile(r"^(\d+)-(\d+)(?:\([^)]*\))?$")


class DataError(ValueError):
    pass


def _col(data: pd.DataFrame, *options: str) -> str | None:
    cols = {c.lower().strip().replace(' ', '_'): c for c in data.columns}
    for option in options:
        key = option.lower().replace(' ', '_')
        if key in cols:
            return cols[key]
    return None


def _dates(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    compact = s.str.fullmatch(r"\d{8}")
    dt = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if compact.any():
        dt.loc[compact] = pd.to_datetime(s.loc[compact], format="%Y%m%d", errors="coerce")
    if (~compact).any():
        dt.loc[~compact] = pd.to_datetime(s.loc[~compact], errors="coerce", dayfirst=True)
    return dt


def _parse_score(value: Any) -> tuple[int, int, int, int] | None:
    if pd.isna(value):
        return None
    value = str(value).strip()
    if not value or re.search(r"\b(RET|W/O|WALKOVER|DEF|ABN|UNFINISHED|SUSP|DEFAULT)\b", value, re.I):
        return None
    sets: list[tuple[int,int]] = []
    for token in value.split():
        m = SCORES.match(token)
        if not m:
            return None
        a, b = int(m.group(1)), int(m.group(2))
        # Excluimos super-tiebreaks y marcadores improbables para no falsear juegos.
        if max(a,b)>7 or a==b or min(a,b)<0:
            return None
        sets.append((a,b))
    if len(sets) not in (2,3,4,5):
        return None
    wsets = sum(a>b for a,b in sets)
    lsets = sum(b>a for a,b in sets)
    if wsets <= lsets or wsets < 2:
        return None
    return sum(a for a,_ in sets), sum(b for _,b in sets), wsets, lsets


def _score_from_sets(df: pd.DataFrame) -> pd.Series:
    parts = []
    for i in range(1,6):
        w, l = _col(df, f"W{i}"), _col(df, f"L{i}")
        if w is not None and l is not None:
            parts.append((w,l))
    if len(parts)<2:
        raise DataError("Falta columna 'score' (formato 6-4 6-3) o columnas W1,L1,W2,L2...")
    output=[]
    for _, r in df.iterrows():
        tokens=[]
        for w,l in parts:
            if pd.notna(r[w]) and pd.notna(r[l]):
                try:
                    a,b=int(float(r[w])),int(float(r[l]))
                    if a or b:
                        tokens.append(f"{a}-{b}")
                except (ValueError,TypeError):
                    pass
        output.append(' '.join(tokens))
    return pd.Series(output,index=df.index)


def normalize_matches(df: pd.DataFrame, tour: str="ATP") -> pd.DataFrame:
    """Normaliza CSV Sackmann (winner_name/loser_name) y Tennis-Data (Winner/Loser)."""
    if df.empty:
        raise DataError("El archivo no contiene partidos.")
    w=_col(df,'winner_name','Winner','ganador')
    l=_col(df,'loser_name','Loser','perdedor')
    date=_col(df,'tourney_date','Date','fecha','match_date')
    surf=_col(df,'surface','superficie')
    score=_col(df,'score','Score','resultado')
    bo=_col(df,'best_of','Best of','bestof')
    comment=_col(df,'Comment','comment')
    missing=[name for name,key in [('winner/ganador',w),('loser/perdedor',l),('Date/tourney_date',date),('surface',surf)] if key is None]
    if missing:
        raise DataError("Faltan columnas: " + ', '.join(missing))
    out=pd.DataFrame({
        'date':_dates(df[date]),
        'winner':df[w].astype(str).str.strip(),
        'loser':df[l].astype(str).str.strip(),
        'surface':df[surf].astype(str).str.strip().str.title(),
        'score':df[score].astype(str) if score is not None else _score_from_sets(df),
        'tour':str(tour).upper(),
        'best_of':pd.to_numeric(df[bo], errors='coerce') if bo else 3,
    })
    if comment is not None:
        rejected=df[comment].astype(str).str.contains('retired|walkover|unfinished|abandoned|default',case=False,regex=True)
        out=out.loc[~rejected].copy()
    out['best_of']=out['best_of'].fillna(3).astype(int)
    out=out.dropna(subset=['date'])
    out=out.loc[(out.winner!='')&(out.loser!='')&(out.winner!=out.loser)&out.surface.isin(SURFACES)]
    records=out.score.map(_parse_score)
    out=out.loc[records.notna()].copy()
    records=records.loc[out.index]
    out['winner_games']=records.map(lambda a:a[0]).astype(int)
    out['loser_games']=records.map(lambda a:a[1]).astype(int)
    out['winner_sets']=records.map(lambda a:a[2]).astype(int)
    out['loser_sets']=records.map(lambda a:a[3]).astype(int)
    out=out.loc[out['winner_sets'].eq(out['best_of']//2+1)]
    out=out.drop_duplicates(subset=['date','winner','loser','score','tour'])
    out=out.sort_values(['date','winner','loser']).reset_index(drop=True)
    if out.empty:
        raise DataError('No quedaron partidos completos válidos. Revisa resultados/fechas y los marcadores.')
    return out


@dataclass
class Player:
    overall: float=1500.
    surfaces: dict[str,float]=field(default_factory=dict)
    n: int=0
    surf_n: dict[str,int]=field(default_factory=dict)
    last_played: pd.Timestamp|None=None

    def rating(self,surface:str)->float:
        count=self.surf_n.get(surface,0)
        weight=0.65*(count/(count+12))
        return (1-weight)*self.overall + weight*self.surfaces.get(surface,1500.)


class TennisModel:
    """Elo + partidos históricos similares, sin mezclar resultados futuros al entrenar."""
    def __init__(self, matches:pd.DataFrame):
        self.matches=matches.sort_values(['date','winner','loser']).reset_index(drop=True).copy()
        self.players:dict[tuple[str,str],Player]={}
        self.history:pd.DataFrame=pd.DataFrame()
        self._fit()

    def _player(self,tour:str,name:str)->Player:
        return self.players.setdefault((tour,name),Player())

    @staticmethod
    def _logistic(a:float,b:float)->float:
        return 1/(1+10**((b-a)/400))

    def probability(self,tour:str,a:str,b:str,surface:str)->dict:
        p1=self.players.get((tour,a),Player())
        p2=self.players.get((tour,b),Player())
        p=self._logistic(p1.rating(surface),p2.rating(surface))
        return {'p_win':p, 'elo_a':round(p1.rating(surface)), 'elo_b':round(p2.rating(surface)),
                'matches_a':p1.n, 'matches_b':p2.n,
                'surface_a':p1.surf_n.get(surface,0), 'surface_b':p2.surf_n.get(surface,0)}

    def _fit(self)->None:
        examples=[]
        for i,row in self.matches.iterrows():
            tour,surf=str(row.tour),str(row.surface)
            winner,loser=str(row.winner),str(row.loser)
            pw=self._player(tour,winner)
            pl=self._player(tour,loser)
            wr,lr=pw.rating(surf),pl.rating(surf)
            prob=self._logistic(wr,lr)
            # Elegimos el Jugador A por orden alfabético ANTES de conocer el resultado.
            # Orientar siempre por el ganador introduciría fuga de información grave.
            a_is_winner = winner.casefold() < loser.casefold()
            margin = int(row.winner_games-row.loser_games)
            examples.append({'date':row.date,'tour':tour,'surface':surf,'best_of':int(row.best_of),
                'p_model':prob if a_is_winner else 1-prob,
                'winner_flag':int(a_is_winner),
                'margin':margin if a_is_winner else -margin,
                'won_set':1 if a_is_winner else int(row.loser_sets>=1),
                'total_games':int(row.winner_games+row.loser_games),
                'n_a':pw.n if a_is_winner else pl.n,
                'n_b':pl.n if a_is_winner else pw.n,
            })
            # La actualización ocurre DESPUÉS de guardar las variables previas al partido.
            k=32.0
            delta=k*(1-prob)
            pw.overall+=delta; pl.overall-=delta
            sw=pw.surfaces.get(surf,1500.)
            sl=pl.surfaces.get(surf,1500.)
            ps=self._logistic(sw,sl)
            change=26*(1-ps)
            pw.surfaces[surf]=sw+change
            pl.surfaces[surf]=sl-change
            for player in (pw,pl):
                player.n+=1
                player.surf_n[surf]=player.surf_n.get(surf,0)+1
                player.last_played=row.date
        self.history=pd.DataFrame(examples)

    def _similar(self, tour:str, surface:str, p_win:float, best_of:int=3)->tuple[pd.DataFrame,np.ndarray,float]:
        h=self.history
        sel=h.loc[(h.tour==tour)&(h.best_of==best_of)].copy()
        if sel.empty:
            return sel,np.array([]),0.
        # Una muestra por partido: orientación determinista por nombre, nunca por resultado.
        weights=np.exp(-.5*((sel.p_model.to_numpy()-p_win)/0.09)**2)
        weights*=np.where(sel.surface.to_numpy()==surface,1.,0.22)
        weights=np.maximum(weights,1.e-15)
        effective=weights.sum()**2/(weights@weights)
        return sel,weights,float(effective)

    def estimate_market(self,tour:str, surface:str, p_win:float, market:str,
                        line:float=2.5, best_of:int=3) -> dict:
        h,w,n=self._similar(tour,surface,p_win,best_of)
        if len(h)==0 or n<20:
            return {'p':None,'lower':None,'upper':None,'n_eff':round(n,1), 'n_matches':len(h)}
        if market=='winner':
            outcomes=h.winner_flag.to_numpy(dtype=float)
        elif market=='set':
            outcomes=h.won_set.to_numpy(dtype=float)
        elif market=='handicap':
            # -2.5 significa margen>=3; +2.5 significa margen>=-2
            outcomes=(h.margin.to_numpy(dtype=float)+line>0).astype(float)
        elif market=='over':
            outcomes=(h.total_games.to_numpy(dtype=float)>line).astype(float)
        elif market=='under':
            outcomes=(h.total_games.to_numpy(dtype=float)<line).astype(float)
        else:
            raise ValueError('Mercado desconocido')
        observed=float(np.average(outcomes,weights=w))
        # Intervalos *aproximados*, no calibración real: el suavizado genera dependencias.
        alpha=0.5+observed*n
        beta_param=0.5+(1-observed)*n
        post=alpha/(alpha+beta_param)
        lo,hi=beta.ppf([0.05,0.95],alpha,beta_param)
        return {'p':round(float(post),4),'lower':round(float(lo),4),
                'upper':round(float(hi),4),'n_eff':round(n,1), 'n_matches':len(h)}

    def backtest(self)->dict:
        hist=self.history
        valid=hist[(hist.n_a>=10)&(hist.n_b>=10)].copy()
        if len(valid)<60:
            return {'valid':False,'reason':'Necesitas al menos 60 partidos evaluables con historial previo suficiente.'}
        # 20% más reciente; estados Elo calculados solo con resultados anteriores.
        test=valid.iloc[int(len(valid)*0.8):]
        y=test.winner_flag.to_numpy(dtype=int)
        pred=np.clip(test.p_model.to_numpy(),1.e-8,1-1.e-8)
        accuracy=float(np.mean((pred>=.5)==y))
        brier=float(np.mean((pred-y)**2))
        ll=float(-np.mean(y*np.log(pred)+(1-y)*np.log(1-pred)))
        return {'valid':True,'n_train':int(len(valid)-len(test)), 'n_test':int(len(test)),
                'accuracy':round(accuracy,4),'brier':round(brier,4), 'logloss':round(ll,4),
                'test':test.copy()}


def evaluate_odds(estimate:dict,odds:float)->dict:
    p=estimate.get('p')
    if p is None or odds<=1:
        return {'implied':round(1/odds,4) if odds>1 else None,'ev':None,'low_ev':None}
    return {'implied':round(1/odds,4),'ev':round(p*odds-1,4),
            'low_ev':round(estimate['lower']*odds-1,4)}
