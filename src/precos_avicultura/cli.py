"""Command line interface.

    python -m precos_avicultura.cli dados --sintetico
    python -m precos_avicultura.cli backtest --horizontes 1 2 3
    python -m precos_avicultura.cli treinar
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .backtest import run_full_backtest
from .config import DEFAULT_TEST_SIZE, HORIZONS, MODELS_DIR, PANEL_PATH, PROCESSED_DIR
from .data.loaders import build_real_panel, load_panel, save_panel
from .data.synthetic import make_synthetic_panel
from .models.train import train_all


def _cmd_dados(args: argparse.Namespace) -> None:
    if args.sintetico:
        panel = make_synthetic_panel(n_months=args.meses, seed=args.seed)
        print(f"Synthetic panel generated: {len(panel)} months")
    else:
        panel = build_real_panel(args.raw_dir)
        print(f"Real panel assembled from {args.raw_dir}: {len(panel)} months")
    path = save_panel(panel, args.saida)
    print(f"Saved to {path}")


def _cmd_backtest(args: argparse.Namespace) -> None:
    panel = load_panel(args.panel)
    predictions, scores = run_full_backtest(
        panel,
        horizons=tuple(args.horizontes),
        test_size=args.test_size,
        with_sarimax=args.sarimax,
    )

    out_dir = Path(args.saida)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / "backtest_previsoes.csv", index=False)
    scores.to_csv(out_dir / "backtest_metricas.csv", index=False)

    with pd.option_context("display.width", 120, "display.max_columns", 20):
        print("\nWalk-forward results (lower MAE is better; MASE < 1 beats seasonal naive)\n")
        print(scores.round(4).to_string(index=False))

    print(f"\nArtifacts written to {out_dir}")
    for horizon in sorted(scores["horizonte"].unique()):
        sub = scores[scores["horizonte"] == horizon].sort_values("mae")
        best = sub.iloc[0]
        print(f"  h={horizon}: best = {best['modelo']} (MAE {best['mae']:.4f})")


def _cmd_treinar(args: argparse.Namespace) -> None:
    panel = load_panel(args.panel)
    cards = train_all(
        panel,
        horizons=tuple(args.horizontes),
        estimator_name=args.estimador,
        target_mode=args.alvo,
        test_size=args.test_size,
        models_dir=Path(args.models_dir),
        evaluate=not args.rapido,
    )
    for horizon, card in cards.items():
        mae = card["metricas_backtest"].get("mae")
        mase = card["metricas_backtest"].get("mase")
        tem_metricas = mae is not None and mase is not None
        extra = f" | MAE {mae:.4f} | MASE {mase:.3f}" if tem_metricas else ""
        print(f"h={horizon}: {card['modelo']}{extra}")
    print(f"Artifacts in {args.models_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="precos-avicultura")
    sub = parser.add_subparsers(dest="comando", required=True)

    p_dados = sub.add_parser("dados", help="build the monthly panel")
    p_dados.add_argument("--sintetico", action="store_true", help="generate demo data")
    p_dados.add_argument("--meses", type=int, default=156)
    p_dados.add_argument("--seed", type=int, default=42)
    p_dados.add_argument("--raw-dir", default="data/raw")
    p_dados.add_argument("--saida", default=str(PANEL_PATH))
    p_dados.set_defaults(func=_cmd_dados)

    p_back = sub.add_parser("backtest", help="walk-forward evaluation")
    p_back.add_argument("--panel", default=str(PANEL_PATH))
    p_back.add_argument("--horizontes", type=int, nargs="+", default=list(HORIZONS))
    p_back.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    p_back.add_argument(
        "--sarimax", action="store_true", help="include the SARIMAX baseline (slow)"
    )
    p_back.add_argument("--saida", default=str(PROCESSED_DIR))
    p_back.set_defaults(func=_cmd_backtest)

    p_train = sub.add_parser("treinar", help="fit and persist production models")
    p_train.add_argument("--panel", default=str(PANEL_PATH))
    p_train.add_argument("--horizontes", type=int, nargs="+", default=list(HORIZONS))
    p_train.add_argument("--estimador", default="gbm", choices=["gbm", "ridge"])
    p_train.add_argument("--alvo", default="delta", choices=["delta", "level"])
    p_train.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    p_train.add_argument("--models-dir", default=str(MODELS_DIR))
    p_train.add_argument("--rapido", action="store_true", help="skip the backtest before fitting")
    p_train.set_defaults(func=_cmd_treinar)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
