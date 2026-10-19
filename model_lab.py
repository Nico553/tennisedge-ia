"""Modelos de tenis con características previas al partido y test temporal.

Predicciones experimentales, sin extrapolar su accuracy a la rentabilidad.
Las estadísticas de todos los partidos del mismo día se congelan ANTES de actualizar Elo.
"""
from __future__ import annotations
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression

FEATURES=['elo_diff','overall_diff','recent_form_diff','surf_recent_diff','experience_diff',
          'log_matches_a','log_matches_b','log_surface_a','log_surface_b',
          'rest_a','rest_b','rank_diff','rank_present','hard','clay','grass','carpet','atp','bo5']

@dataclass
class State:
    overall:float=1500.
    surface:dict=field(default_factory=dict)
    ns:dict=field(default_factory=dict)
    played:int=0
    rank:float|None=None
    last:pd.Timestamp|None=None
    recent:deque=field(default_factory=lambda:deque(maxlen=20))
    recent_surf:dict=field(default_factory=lambda:defaultdict(lambda:deque(maxlen=12)))

    def rating(self,s):
        n=self.ns.get(s,0)
        w=.65*n/(n+12)
        return (1-w)*self.overall+w*self.surface.get(s,1500.)

    def form(self,s=None):
        seq=list(self.recent_surf[s]) if s else list(self.recent)
        # shrink short streaks toward 50%
        return (sum(seq)+4)/(len(seq)+8)

class ChronologicalLab:
    def __init__(self, matches:pd.DataFrame):
        self.matches=matches.copy().sort_values(['date','tour','winner','loser']).reset_index(drop=True)
        self.states:Dict[tuple,State]={}
        self.examples=self._build_features()
        self.trained={}
        self.calibrators={}
        self.selected=None
        self.scoreboard=None
        self.market_metrics=None
        self.split_dates=None

    def _state(self,tour,name):
        return self.states.setdefault((tour,name),State())

    @staticmethod
    def _pair_features(sa,sb,tour,s,bo,date):
        last_a=min(365,(date-sa.last).days) if sa.last is not None else 365
        last_b=min(365,(date-sb.last).days) if sb.last is not None else 365
        return {'elo_diff':sa.rating(s)-sb.rating(s),
            'overall_diff':sa.overall-sb.overall,
            'recent_form_diff':sa.form()-sb.form(),
            'surf_recent_diff':sa.form(s)-sb.form(s),
            'experience_diff':np.log1p(sa.played)-np.log1p(sb.played),
            'log_matches_a':np.log1p(sa.played),'log_matches_b':np.log1p(sb.played),
            'log_surface_a':np.log1p(sa.ns.get(s,0)),'log_surface_b':np.log1p(sb.ns.get(s,0)),
            'rest_a':np.log1p(max(0,last_a)),'rest_b':np.log1p(max(0,last_b)),
            'rank_diff':np.log1p(sb.rank)-np.log1p(sa.rank) if sa.rank and sb.rank else 0.,
            'rank_present':int(bool(sa.rank and sb.rank)),
            'hard':int(s=='Hard'),'clay':int(s=='Clay'),
            'grass':int(s=='Grass'),'carpet':int(s=='Carpet'),
            'atp':int(tour=='ATP'),'bo5':int(bo==5)}

    def _build_features(self):
        records=[]
        for date,g in self.matches.groupby('date',sort=True):
            # Ningún resultado de este día modifica características del mismo día.
            batch=[]
            for r in g.itertuples(index=False):
                tour,s,bo=r.tour,r.surface,int(r.best_of)
                winner,loser=r.winner,r.loser
                a,b=sorted((winner,loser), key=str.casefold)
                sa,sb=self._state(tour,a),self._state(tour,b)
                f=self._pair_features(sa,sb,tour,s,bo,date)
                f.update({'date':date,'tour':tour,'player_a':a,'player_b':b,
                    'surface':s,'best_of':bo,'a_win':int(a==winner),
                    'a_set':int(a==winner or r.loser_sets>=1),
                    'b_set':int(b==winner or r.loser_sets>=1),
                    'a_margin':int(r.winner_games-r.loser_games)*(1 if a==winner else -1),
                    'total_games':int(r.winner_games+r.loser_games),
                    'a_n':sa.played,'b_n':sb.played,
                    'elo_p':1/(1+10**(-(sa.rating(s)-sb.rating(s))/400))})
                records.append(f)
                batch.append((tour,s,date,winner,loser,r.winner_rank,r.loser_rank))
            for tour,s,d,winner,loser,rankw,rankl in batch:
                sw,sl=self._state(tour,winner),self._state(tour,loser)
                p=1/(1+10**((sl.overall-sw.overall)/400))
                delta=30*(1-p)
                sw.overall+=delta;sl.overall-=delta
                ws,ls=sw.surface.get(s,1500.),sl.surface.get(s,1500.)
                ps=1/(1+10**((ls-ws)/400))
                delta_s=25*(1-ps)
                sw.surface[s]=ws+delta_s;sl.surface[s]=ls-delta_s
                if pd.notna(rankw) and rankw>0:sw.rank=float(rankw)
                if pd.notna(rankl) and rankl>0:sl.rank=float(rankl)
                for st,won in ((sw,1),(sl,0)):
                    st.played+=1;st.ns[s]=st.ns.get(s,0)+1;st.last=d
                    st.recent.append(won);st.recent_surf[s].append(won)
        return pd.DataFrame(records)

    @staticmethod
    def _augment(frame):
        """Duplica orientaciones solo en entrenamiento, nunca mezcla particiones temporales."""
        opposite=frame.copy()
        for k in ('elo_diff','overall_diff','recent_form_diff','surf_recent_diff','experience_diff','rank_diff'):
            opposite[k]=-opposite[k]
        for a,b in [('log_matches_a','log_matches_b'),('log_surface_a','log_surface_b'),
                    ('rest_a','rest_b'),('a_n','b_n')]:
            opposite[a],opposite[b]=frame[b].to_numpy(),frame[a].to_numpy()
        opposite['a_win']=1-frame.a_win.to_numpy()
        opposite['a_set']=frame.b_set.to_numpy()
        opposite['b_set']=frame.a_set.to_numpy()
        opposite['a_margin']=-frame.a_margin.to_numpy()
        opposite['elo_p']=1-frame.elo_p.to_numpy()
        return pd.concat([frame,opposite],ignore_index=True)

    @staticmethod
    def _classifier(name):
        if name=='Regresión logística':
            return make_pipeline(StandardScaler(),LogisticRegression(max_iter=600,C=0.3))
        if name=='Random Forest':
            return RandomForestClassifier(n_estimators=65,max_depth=9,min_samples_leaf=28,
                                          n_jobs=1,random_state=42)
        if name=='Gradient Boosting':
            return HistGradientBoostingClassifier(max_iter=85,max_leaf_nodes=12,
                         l2_regularization=5.,learning_rate=.055,min_samples_leaf=40,
                         random_state=42)
        raise ValueError(name)

    @staticmethod
    def _metrics(y,p):
        y=np.asarray(y).astype(int);p=np.clip(np.asarray(p,dtype=float),.00001,.99999)
        out={'N':len(y),'Exactitud':float(accuracy_score(y,p>=.5)),
            'Brier ↓':float(brier_score_loss(y,p)),
            'Log loss ↓':float(log_loss(y,p,labels=[0,1]))}
        try:out['AUC']=float(roc_auc_score(y,p))
        except ValueError:out['AUC']=np.nan
        return out

    def benchmark(self):
        df=self.examples
        good=df[(df.a_n>=8)&(df.b_n>=8)].copy().sort_values('date')
        if len(good)<400:
            raise ValueError('Se requieren al menos 400 partidos con antecedentes previos (8 por jugador).')
        dates=good.date.drop_duplicates().sort_values().to_numpy()
        if len(dates)<30:
            raise ValueError('Faltan fechas distintas para validar cronológicamente.')
        cut1=pd.Timestamp(dates[int(len(dates)*.70)])
        cut2=pd.Timestamp(dates[int(len(dates)*.85)])
        tr=good[good.date<cut1];val=good[(good.date>=cut1)&(good.date<cut2)];te=good[good.date>=cut2]
        if min(len(tr),len(val),len(te))<80:
            raise ValueError('Las particiones temporales tienen muy pocos partidos.')
        train_aug=self._augment(tr)
        xtr=train_aug[FEATURES].to_numpy();xval=val[FEATURES].to_numpy();xte=te[FEATURES].to_numpy()
        ytr=train_aug.a_win.to_numpy();yval=val.a_win.to_numpy();yte=te.a_win.to_numpy()
        classifiers={}; validation=[]; scores=[]
        for name in ('Elo por superficie','Regresión logística','Random Forest','Gradient Boosting'):
            if name=='Elo por superficie':
                pv=val.elo_p.to_numpy();pt=te.elo_p.to_numpy()
            else:
                model=self._classifier(name)
                model.fit(xtr,ytr)
                pv=model.predict_proba(xval)[:,1]
                pt=model.predict_proba(xte)[:,1]
                classifiers[name]=model
            # Isotonic fitted only on validation; independent chronological test
            # evaluates the calibrated predictions with no test-set feedback.
            cal=IsotonicRegression(out_of_bounds='clip',y_min=0.001,y_max=0.999)
            cal.fit(pv,yval)
            self.calibrators[name]=cal
            validation.append({'Modelo':name,**self._metrics(yval,pv)})
            scores.append({'Modelo':name,**self._metrics(yte,cal.predict(pt))})
        v=pd.DataFrame(validation).sort_values('Brier ↓')
        chosen=str(v.iloc[0].Modelo)
        self.selected=chosen
        self.validation=pd.DataFrame(validation)
        self.scoreboard=pd.DataFrame(scores).sort_values('Brier ↓')
        self.split_dates=(str(cut1.date()),str(cut2.date()))
        self.test=te.copy()
        # Modelos finales entrenados después de evaluar, sin introducir resultados futuros en el test.
        full_aug=self._augment(good)
        allx=full_aug[FEATURES].to_numpy()
        ally=full_aug.a_win.to_numpy()
        for name in ('Regresión logística','Random Forest','Gradient Boosting'):
            model=self._classifier(name)
            model.fit(allx,ally)
            self.trained[name]=model
        # Comparador adicional con mercados específicos, no inferir EV sin cuotas históricas.
        extra=[]
        for name,labels in [('A gana ≥1 set',lambda d:d.a_set),
                            ('A −2,5 juegos',lambda d:(d.a_margin>=3).astype(int)),
                            ('Más de 22,5 juegos',lambda d:(d.total_games>22.5).astype(int))]:
            for split_name,part in [('train',tr),('test',te)]:
                if len(set(labels(part)))<2:raise ValueError('Un mercado no tiene resultados de ambas clases')
            m=self._classifier('Regresión logística')
            m.fit(xtr,np.asarray(labels(train_aug)))
            p=m.predict_proba(xte)[:,1]
            extra.append({'Mercado':name,'Modelo':'Regresión logística','N':len(yte),
                          **{k:v for k,v in self._metrics(labels(te),p).items() if k!='N'}})
            m=self._classifier('Regresión logística')
            m.fit(allx,np.asarray(labels(full_aug)))
            self.trained[name]=m
        self.market_metrics=pd.DataFrame(extra)
        self.n_train=len(tr);self.n_val=len(val);self.n_test=len(te)
        return self

    def predict(self,tour:str,a:str,b:str,surface:str,best_of=3):
        tour=tour.upper();surface=surface.title()
        if a==b:raise ValueError('Selecciona dos jugadores distintos')
        if (tour,a) not in self.states or (tour,b) not in self.states:
            raise ValueError('Falta historial para alguno de los jugadores')
        dt=self.matches.date.max()+pd.Timedelta(days=1)
        sa,sb=self.states[tour,a],self.states[tour,b]
        features=self._pair_features(sa,sb,tour,surface,best_of,dt)
        x=np.array([[features[k] for k in FEATURES]])
        base=1/(1+10**(-(sa.rating(surface)-sb.rating(surface))/400))
        out={}
        for name in ('Elo por superficie','Regresión logística','Random Forest','Gradient Boosting'):
            raw=float(base if name=='Elo por superficie' else self.trained[name].predict_proba(x)[0,1])
            out[name]=float(self.calibrators[name].predict([raw])[0])
        for market in ('A gana ≥1 set','A −2,5 juegos','Más de 22,5 juegos'):
            out[market]=float(self.trained[market].predict_proba(x)[0,1])
        return {'probabilities':out,'details':{
            'matches_a':sa.played,'matches_b':sb.played,
            'surface_a':sa.ns.get(surface,0),'surface_b':sb.ns.get(surface,0),
            'last_a':str(sa.last.date()) if sa.last is not None else None,
            'last_b':str(sb.last.date()) if sb.last is not None else None,
            'data_max':str(self.matches.date.max().date())}}
