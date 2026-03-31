# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 23:05:10 2026

@author: heidrich
"""

#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path
import yaml


def load_yaml(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(obj: dict, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def csv_to_run_id(csv_path: Path) -> str:
    return csv_path.stem


def apply_prefix_filter(files: list[Path], prefix: str | None) -> list[Path]:
    if not prefix:
        return files
    return [f for f in files if f.name.startswith(prefix)]


def apply_shard(
    files: list[Path], shard_idx: int | None, shard_total: int | None
) -> list[Path]:
    """
    Deterministisch: sortieren, dann round-robin auf shards verteilen.
    shard_idx: 0..shard_total-1
    """
    if shard_idx is None or shard_total is None:
        return files
    if shard_total <= 0 or not (0 <= shard_idx < shard_total):
        raise ValueError(
            "--shard_idx muss in [0, shard_total) liegen und shard_total > 0 sein"
        )

    out = []
    for i, f in enumerate(files):
        if i % shard_total == shard_idx:
            out.append(f)
    return out


def newest_subdir(p: Path) -> Path | None:
    if not p.exists():
        return None
    subdirs = [d for d in p.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    # neuester Ordner nach Änderungszeit
    return max(subdirs, key=lambda d: d.stat().st_mtime)


def get_best_params_file(repo_root: Path, tuning_name: str) -> Path | None:
    base = repo_root / "outputs" / "tuning" / tuning_name
    ts_dir = newest_subdir(base)
    if ts_dir is None:
        return None
    best = ts_dir / "gridsearch" / "best_params.yaml"
    return best if best.exists() else None


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--repo_root", type=Path, default=Path("."))
    ap.add_argument(
        "--data_dir", type=Path, default=Path("data/data_processed")
    )
    ap.add_argument(
        "--default_tuning",
        type=Path,
        default=Path("configs/tuning/tuning_Aalborg_SFH_NMAE.yaml"),
    )
    ap.add_argument(
        "--default_training",
        type=Path,
        default=Path("configs/training/train_Aalborg_SFH_hourly.yaml"),
    )
    ap.add_argument(
        "--out_tuning_dir",
        type=Path,
        default=Path("configs/tuning/_generated"),
    )
    ap.add_argument(
        "--out_training_dir",
        type=Path,
        default=Path("configs/training/_generated"),
    )

    # NEU: Prefix-Filter (für “Patches”)
    ap.add_argument(
        "--file_prefix",
        type=str,
        default=None,
        help="Nur CSVs deren Dateiname mit diesem Prefix beginnt (z.B. 'aggregated__consumertype_terraced_house').",
    )

    # NEU: Sharding (noch bequemer als Prefixes)
    ap.add_argument(
        "--shard_idx",
        type=int,
        default=None,
        help="Shard Index 0..shard_total-1 (z.B. 0)",
    )
    ap.add_argument(
        "--shard_total", type=int, default=None, help="Anzahl Shards (z.B. 4)"
    )
    ap.add_argument(
        "--glob",
        type=str,
        default="*.csv",
        help="Dateimuster für CSVs in data_dir, z.B. '*_temp_hourly.csv'",
    )

    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    data_dir = (repo_root / args.data_dir).resolve()
    default_tuning = (repo_root / args.default_tuning).resolve()
    default_training = (repo_root / args.default_training).resolve()
    out_tuning_dir = (repo_root / args.out_tuning_dir).resolve()
    out_training_dir = (repo_root / args.out_training_dir).resolve()

    tuning_tpl = load_yaml(default_tuning)
    train_tpl = load_yaml(default_training)

    files = sorted(data_dir.glob(args.glob))

    # NUR aggregated__*.csv zulassen
    files = [
        f for f in files if f.is_file() and f.name.startswith("aggregated__")
    ]

    if not files:
        raise SystemExit(
            "Keine passenden Dateien gefunden: Dateiname muss mit 'aggregated__' beginnen."
        )

        if not files:
            raise SystemExit(
                f"Keine CSVs gefunden in: {data_dir} (glob={args.glob})"
            )

        files = apply_prefix_filter(files, args.file_prefix)
        files = apply_shard(files, args.shard_idx, args.shard_total)

    if not files:
        raise SystemExit(
            "Nach Filter/Shard sind keine Dateien übrig geblieben."
        )

    print(f"CSV gesamt (glob): {len(sorted(data_dir.glob(args.glob)))}")
    if args.file_prefix:
        print(
            f"Nach prefix '{args.file_prefix}': {len(apply_prefix_filter(sorted(data_dir.glob(args.glob)), args.file_prefix))}"
        )
    if args.shard_idx is not None:
        print(f"Shard {args.shard_idx+1}/{args.shard_total}: {len(files)}")
    print(f"-> Zu verarbeiten: {len(files)}")

    for csv_path in files:
        run_id = csv_to_run_id(csv_path)

        # 1) tuning yaml
        tuning_cfg = yaml.safe_load(yaml.safe_dump(tuning_tpl))
        tuning_cfg["name"] = f"{run_id}_NMAE"
        tuning_cfg["data"]["path"] = f"data/data_processed/{csv_path.name}"

        tuning_out = out_tuning_dir / f"tuning_{run_id}_NMAE.yaml"
        dump_yaml(tuning_cfg, tuning_out)

        # 2) run tuning
        run_cmd(
            [
                "python",
                "-m",
                "scripts.tune_grid_seasonal",
                "--config",
                str(tuning_out.relative_to(repo_root)),
            ],
            cwd=repo_root,
        )

        # 3) best params deterministisch
        # outputs/tuning/{name}/gridsearch/best_params.yaml
        tuning_name = tuning_cfg["name"]
        tuning_name = tuning_cfg["name"]
        best_params_file = get_best_params_file(repo_root, tuning_name)
        if best_params_file is None:
            print(
                f"[WARN] best_params.yaml nicht gefunden für {tuning_name} unter {repo_root/'outputs'/'tuning'/tuning_name}"
            )
            print("       Überspringe Training.")
            continue
        best_params_obj = load_yaml(best_params_file)

        if not best_params_file.exists():
            print(
                f"[WARN] best_params.yaml nicht gefunden: {best_params_file}"
            )
            print("       Überspringe Training.")
            continue

        best_params_obj = load_yaml(best_params_file)

        # best_params.yaml kann entweder direkt params sein oder verschachtelt
        if (
            isinstance(best_params_obj, dict)
            and "best_params" in best_params_obj
            and isinstance(best_params_obj["best_params"], dict)
        ):
            best_params = best_params_obj["best_params"]
        else:
            best_params = best_params_obj

        # 4) training yaml
        train_cfg = yaml.safe_load(yaml.safe_dump(train_tpl))
        train_cfg["name"] = run_id
        train_cfg["data"]["path"] = f"data/data_processed/{csv_path.name}"

        # params updaten
        train_cfg["trainer"]["params"].update(best_params)

        train_out = out_training_dir / f"train_{run_id}.yaml"
        dump_yaml(train_cfg, train_out)

        # optional: params separat speichern
        best_params_out = out_training_dir / f"best_params_{run_id}.yaml"
        dump_yaml(best_params, best_params_out)

        # 5) run training
        run_cmd(
            [
                "python",
                "-m",
                "scripts.train",
                "--config",
                str(train_out.relative_to(repo_root)),
            ],
            cwd=repo_root,
        )

        print(f"✅ Fertig: {run_id}")
        print(f"   best_params: {best_params_file}")
        print(f"   train_cfg:   {train_out}")


if __name__ == "__main__":
    main()
