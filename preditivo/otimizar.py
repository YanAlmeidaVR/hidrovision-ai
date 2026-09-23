
"""
Otimização de hiperparâmetros dos modelos XGBoost com Optuna.
"""
import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    raise SystemExit("instale o optuna: pip install optuna")

FUSO = "America/Sao_Paulo"
HORIZONTES = (6, 12, 24)


# features e alvo (variação), idênticos ao treino de produção
def carregar(caminho):
    df = pd.read_csv(caminho, parse_dates=["datahora"])
    df["datahora"] = pd.to_datetime(df["datahora"], utc=True).dt.tz_convert(FUSO)
    df = df.drop_duplicates("datahora").sort_values("datahora").set_index("datahora")
    grade = pd.date_range(df.index.min(), df.index.max(), freq="1h", tz=FUSO)
    return df.reindex(grade).rename_axis("datahora")


def construir(df, horizonte):
    X = pd.DataFrame(index=df.index)
    nivel = df["nivel_cm"]
    chuva = df["chuva_mm"].fillna(0.0)
    X["nivel"] = nivel
    for h in (1, 2, 3, 6, 12, 24):
        X[f"nivel_lag{h}"] = nivel.shift(h)
    X["delta_1h"] = nivel - nivel.shift(1)
    X["delta_3h"] = nivel - nivel.shift(3)
    X["delta_6h"] = nivel - nivel.shift(6)
    X["tend_6h"] = X["delta_6h"] / 6.0
    X["chuva"] = chuva
    for j in (3, 6, 12, 24, 48, 72):
        X[f"chuva_acum{j}"] = chuva.rolling(j, min_periods=1).sum()
    X["chuva_max6"] = chuva.rolling(6, min_periods=1).max()
    houve = (chuva > 1.0).astype(int)
    g = houve.cumsum()
    X["horas_sem_chuva"] = houve.groupby(g).cumcount().where(g > 0, 72).clip(upper=72)
    X["chuva24_x_nivel"] = X["chuva_acum24"] * X["nivel"] / 100.0
    mes = X.index.month
    X["mes_sen"] = np.sin(2 * np.pi * mes / 12)
    X["mes_cos"] = np.cos(2 * np.pi * mes / 12)
    X["alvo"] = nivel.shift(-horizonte) - nivel
    return X.dropna()


def separar_temporal(X):
    """
    Três blocos SEM sobreposição, na ordem do tempo:
      treino    : 2023–2024  (ajusta o modelo)
      validação : 2025       (guia a busca do Optuna)
      teste     : 2026       (mede o resultado final, nunca visto pela busca)
    """
    feats = [c for c in X.columns if c != "alvo"]
    tr = X[X.index.year <= 2024]
    va = X[X.index.year == 2025]
    te = X[X.index.year >= 2026]
    return (tr[feats], tr["alvo"], va[feats], va["alvo"],
            te[feats], te["alvo"])


def mae(y, pred):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(y))))


def otimizar_horizonte(X, tentativas):
    Xtr, ytr, Xva, yva, Xte, yte = separar_temporal(X)

    def objetivo(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
        m = xgb.XGBRegressor(objective="reg:squarederror", n_jobs=-1,
                             random_state=42, **params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return mae(yva, m.predict(Xva))    # busca minimiza o erro em 2025

    estudo = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    estudo.optimize(objetivo, n_trials=tentativas, show_progress_bar=False)

    # baseline: os hiperparâmetros manuais de produção
    base = xgb.XGBRegressor(
        n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
        objective="reg:squarederror", n_jobs=-1, random_state=42)
    base.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

    # modelo otimizado, retreinado em treino+validação para o teste final
    melhor = xgb.XGBRegressor(objective="reg:squarederror", n_jobs=-1,
                              random_state=42, **estudo.best_params)
    Xtrva = pd.concat([Xtr, Xva]); ytrva = pd.concat([ytr, yva])
    melhor.fit(Xtrva, ytrva, verbose=False)

    # persistência como âncora
    base_persist = mae(yte, np.zeros(len(yte)))
    mae_base = mae(yte, base.predict(Xte))
    mae_otim = mae(yte, melhor.predict(Xte))

    return {
        "params": estudo.best_params,
        "mae_teste_manual": mae_base,
        "mae_teste_otimizado": mae_otim,
        "mae_persistencia": base_persist,
        "ganho_manual_pct": (1 - mae_base / base_persist) * 100,
        "ganho_otim_pct": (1 - mae_otim / base_persist) * 100,
        "melhora_pct": (1 - mae_otim / mae_base) * 100,
        "modelo": melhor,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", default="dados_treino.csv")
    ap.add_argument("--horizonte", type=int, choices=HORIZONTES,
                    help="otimizar só um horizonte (padrão: os três)")
    ap.add_argument("--tentativas", type=int, default=50)
    ap.add_argument("--salvar", action="store_true",
                    help="salva os modelos otimizados como modelo_delta_*.json")
    args = ap.parse_args()

    df = carregar(args.dados)
    horizontes = (args.horizonte,) if args.horizonte else HORIZONTES
    resumo = {}

    for h in horizontes:
        print(f"\n{'='*60}\nHORIZONTE t+{h}h — {args.tentativas} tentativas")
        X = construir(df, h)
        r = otimizar_horizonte(X, args.tentativas)
        print(f"  MAE persistência : {r['mae_persistencia']:.3f} cm")
        print(f"  MAE manual       : {r['mae_teste_manual']:.3f} cm  "
              f"(ganho {r['ganho_manual_pct']:+.1f}%)")
        print(f"  MAE otimizado    : {r['mae_teste_otimizado']:.3f} cm  "
              f"(ganho {r['ganho_otim_pct']:+.1f}%)")
        print(f"  >>> melhora do otimizado sobre o manual: {r['melhora_pct']:+.2f}%")
        print(f"  melhores hiperparâmetros:")
        for k, v in r["params"].items():
            print(f"    {k}: {v:.4g}" if isinstance(v, float) else f"    {k}: {v}")
        if args.salvar:
            r["modelo"].save_model(f"modelo_delta_{h}h.json")
            print(f"  salvo: modelo_delta_{h}h.json")
        resumo[f"t+{h}h"] = {k: v for k, v in r.items() if k != "modelo"}

    with open("otimizacao_resultado.json", "w") as f:
        json.dump(resumo, f, indent=2, default=float)
    print(f"\nresumo salvo em otimizacao_resultado.json")


if __name__ == "__main__":
    main()