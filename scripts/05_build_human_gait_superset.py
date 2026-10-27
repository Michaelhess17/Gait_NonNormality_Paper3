#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import scipy.io as sio


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OLD_DATA = ROOT.parent / "Gait_NonNormality_Paper" / "data"
DEFAULT_ORIGINAL_REPO = Path("/tmp/gait_signatures")
DEFAULT_HYA_REPO = Path("/tmp/GaitSignatures_HealthyYoungAdultStudy")
DEFAULT_POSTSTROKE_REPO = Path("/tmp/GaitSignatures_PostStrokeGaitBiomechanics")
DEFAULT_OUTPUT = ROOT / "data" / "human_gait_superset"

DEFAULT_DUPLICATE_RMS = 0.75
DEFAULT_DUPLICATE_MARGIN = 10.0


@dataclass
class TrialRecord:
    source: str
    source_priority: int
    source_trial_index: int
    subject: str
    speed: float
    lf_or_hf: str
    paretic_side: str
    trial_key: str
    content_hash: str
    source_note: str = ""


@dataclass
class DuplicateRecord:
    skipped_source: str
    skipped_source_trial_index: int
    skipped_subject: str
    skipped_speed: float
    skipped_group: str
    kept_trial_index: int
    kept_source: str
    kept_source_trial_index: int
    kept_subject: str
    kept_speed: float
    kept_group: str
    nearest_rms: float
    second_nearest_rms: float
    second_to_best_rms_ratio: float
    same_trial_key: bool
    decision: str


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_single_column_csv(path: Path) -> list[str]:
    with path.open(newline="") as f:
        return ["".join(row).strip() for row in csv.reader(f) if row]


def normalize_subject(subject: str) -> str:
    subject = subject.strip()
    if subject.startswith("YA") and len(subject) == 5 and subject[2:].isdigit():
        return "YA" + subject[-2:]
    return subject


def trial_key(subject: str, speed: float, group: str) -> str:
    return f"{subject}|{speed:.6g}|{group}"


def hash_trial(x: np.ndarray) -> str:
    rounded = np.round(np.asarray(x, dtype=np.float64), 8)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def load_legacy_cache(old_data_dir: Path) -> tuple[np.ndarray, list[dict[str, str]]]:
    data = np.load(old_data_dir / "all_human_data.npy")
    rows = read_metadata(old_data_dir / "all_human_data_metadata.csv")
    if data.shape[0] != len(rows):
        raise ValueError("Legacy data/metadata length mismatch")
    return np.asarray(data, dtype=np.float64), rows


def legacy_subject_groups(old_rows: list[dict[str, str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in old_rows:
        out.setdefault(row["subject"], row["lf_or_hf"])
    return out


def group_for_subject(subject: str, subject_groups: dict[str, str]) -> str:
    if subject.startswith(("AB", "YA")):
        return "AB"
    if subject in subject_groups:
        return subject_groups[subject]
    raise ValueError(f"Could not infer HF/LF group for {subject!r}")


def make_records(
    data: np.ndarray,
    *,
    source: str,
    priority: int,
    subjects: list[str],
    speeds: list[float],
    groups: list[str],
    paretic_sides: list[str] | None = None,
    note: str = "",
) -> list[TrialRecord]:
    if data.shape[0] != len(subjects) or len(subjects) != len(speeds) or len(speeds) != len(groups):
        raise ValueError(f"Metadata length mismatch for {source}")
    if data.ndim != 3 or data.shape[1:] != (1500, 6):
        raise ValueError(f"{source} must have shape n_trials x 1500 x 6, got {data.shape}")
    paretic_sides = paretic_sides or [""] * len(subjects)
    return [
        TrialRecord(
            source=source,
            source_priority=priority,
            source_trial_index=i + 1,
            subject=subjects[i],
            speed=float(speeds[i]),
            lf_or_hf=groups[i],
            paretic_side=paretic_sides[i],
            trial_key=trial_key(subjects[i], float(speeds[i]), groups[i]),
            content_hash=hash_trial(data[i]),
            source_note=note,
        )
        for i in range(len(subjects))
    ]


def load_original_gait_signatures(repo: Path, subject_groups: dict[str, str]) -> tuple[np.ndarray, list[TrialRecord]]:
    subjects = [normalize_subject(x) for x in read_single_column_csv(repo / "Subjectlabels.csv")]
    speeds = [float(x) for x in sio.loadmat(repo / "SpeedLabels.mat", simplify_cells=True)["SpeedTrainTrials"]]
    raw = np.loadtxt(repo / "PareticvsNonP_RNNData.csv", delimiter=",", dtype=np.float64)
    data = raw.reshape(6, len(subjects), 1500).transpose(1, 2, 0)
    groups = [group_for_subject(subject, subject_groups) for subject in subjects]
    records = make_records(
        data,
        source="gait_signatures",
        priority=50,
        subjects=subjects,
        speeds=speeds,
        groups=groups,
        note="original public PareticvsNonP_RNNData.csv export",
    )
    return data, records


def load_poststroke(repo: Path, subject_groups: dict[str, str]) -> tuple[np.ndarray, list[TrialRecord]]:
    csv_dir = repo / "CSV"
    subjects = [normalize_subject(x) for x in read_single_column_csv(csv_dir / "SpeedSigs_labels.csv")]
    speeds = [float(x) for x in read_single_column_csv(csv_dir / "SpeedSigs_speeds.csv")]
    paretic_sides = read_single_column_csv(csv_dir / "SpeedSigs_isParetic.csv")
    raw = np.loadtxt(csv_dir / "SpeedSigs_data.csv", delimiter=",", dtype=np.float64)
    data = raw.reshape(len(subjects), 1500, 6)
    groups = [group_for_subject(subject, subject_groups) for subject in subjects]
    records = make_records(
        data,
        source="GaitSignatures_PostStrokeGaitBiomechanics",
        priority=90,
        subjects=subjects,
        speeds=speeds,
        groups=groups,
        paretic_sides=paretic_sides,
        note="public SpeedSigs_data.csv export",
    )
    return data, records


def load_hya(repo: Path, old_data_dir: Path) -> tuple[np.ndarray, list[TrialRecord]]:
    input_dir = repo / "Python RNN training code" / "input data"
    subjects = [normalize_subject(x) for x in read_single_column_csv(input_dir / "subjects_label.csv")]
    speeds = [float(x) for x in read_single_column_csv(input_dir / "speeds_label.csv")]
    groups = ["AB"] * len(subjects)

    public_raw = input_dir / "DataStorage_6000_2D_kinematics_clean.csv"
    if public_raw.is_file():
        raw = np.loadtxt(public_raw, delimiter=",", dtype=np.float64)
        if raw.shape == (6, len(subjects) * 1500):
            data = raw.reshape(6, len(subjects), 1500).transpose(1, 2, 0)
        elif raw.shape == (len(subjects) * 1500, 6):
            data = raw.reshape(len(subjects), 1500, 6)
        else:
            raise ValueError(f"Unexpected HYA public raw shape: {raw.shape}")
        note = "public HYA raw kinematics export"
    else:
        legacy_data, legacy_rows = load_legacy_cache(old_data_dir)
        by_key: dict[str, list[int]] = {}
        for i, row in enumerate(legacy_rows):
            key = trial_key(row["subject"], float(row["speed"]), row["lf_or_hf"])
            by_key.setdefault(key, []).append(i)
        idxs = []
        missing = []
        for subject, speed, group in zip(subjects, speeds, groups):
            key = trial_key(subject, speed, group)
            if key in by_key and by_key[key]:
                idxs.append(by_key[key].pop(0))
            else:
                missing.append(key)
        if missing:
            raise ValueError(f"Legacy cache is missing {len(missing)} HYA public label keys, e.g. {missing[:5]}")
        data = legacy_data[idxs]
        note = "HYA public labels matched to local legacy trajectories; public repo omits raw trajectory file"

    records = make_records(
        data,
        source="GaitSignatures_HealthyYoungAdultStudy",
        priority=100,
        subjects=subjects,
        speeds=speeds,
        groups=groups,
        note=note,
    )
    return data, records


def nearest_duplicate(candidate: np.ndarray, accepted: list[np.ndarray]) -> tuple[int, float, float, float]:
    if not accepted:
        return -1, float("inf"), float("inf"), float("inf")
    matrix = np.stack([x.reshape(-1) for x in accepted], axis=0)
    x = candidate.reshape(-1)
    rms = np.sqrt(np.mean((matrix - x) ** 2, axis=1))
    order = np.argsort(rms)
    best_i = int(order[0])
    best = float(rms[best_i])
    second = float(rms[int(order[1])]) if len(order) > 1 else float("inf")
    ratio = second / best if best > 0 else float("inf")
    return best_i, best, second, ratio


def build_unique_dataset(
    sources: list[tuple[np.ndarray, list[TrialRecord]]],
    duplicate_rms: float,
    duplicate_margin: float,
) -> tuple[np.ndarray, list[TrialRecord], list[DuplicateRecord], int]:
    candidates: list[tuple[np.ndarray, TrialRecord]] = []
    for data, records in sources:
        candidates.extend((data[i], record) for i, record in enumerate(records))
    candidates.sort(key=lambda item: (-item[1].source_priority, item[1].source, item[1].source_trial_index))

    accepted_data: list[np.ndarray] = []
    accepted_records: list[TrialRecord] = []
    duplicates: list[DuplicateRecord] = []
    exact_hash_duplicates = 0
    accepted_hashes: set[str] = set()

    for candidate, record in candidates:
        best_i, best, second, ratio = nearest_duplicate(candidate, accepted_data)
        is_duplicate = best <= duplicate_rms and ratio >= duplicate_margin
        if record.content_hash in accepted_hashes:
            exact_hash_duplicates += 1
            is_duplicate = True
        if is_duplicate:
            kept = accepted_records[best_i]
            duplicates.append(
                DuplicateRecord(
                    skipped_source=record.source,
                    skipped_source_trial_index=record.source_trial_index,
                    skipped_subject=record.subject,
                    skipped_speed=record.speed,
                    skipped_group=record.lf_or_hf,
                    kept_trial_index=best_i + 1,
                    kept_source=kept.source,
                    kept_source_trial_index=kept.source_trial_index,
                    kept_subject=kept.subject,
                    kept_speed=kept.speed,
                    kept_group=kept.lf_or_hf,
                    nearest_rms=best,
                    second_nearest_rms=second,
                    second_to_best_rms_ratio=ratio,
                    same_trial_key=record.trial_key == kept.trial_key,
                    decision="skip_numeric_duplicate" if record.content_hash not in accepted_hashes else "skip_exact_hash_duplicate",
                )
            )
            continue
        accepted_data.append(candidate)
        accepted_records.append(record)
        accepted_hashes.add(record.content_hash)

    return np.stack(accepted_data, axis=0), accepted_records, duplicates, exact_hash_duplicates


def write_records(path: Path, records: list[TrialRecord]) -> None:
    fieldnames = [
        "trial_index",
        "source",
        "source_priority",
        "source_trial_index",
        "subject",
        "speed",
        "lf_or_hf",
        "paretic_side",
        "trial_key",
        "content_hash",
        "source_note",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, record in enumerate(records, start=1):
            row = asdict(record)
            row["trial_index"] = i
            writer.writerow(row)


def write_duplicates(path: Path, duplicates: list[DuplicateRecord]) -> None:
    fieldnames = list(asdict(duplicates[0]).keys()) if duplicates else list(DuplicateRecord.__annotations__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in duplicates:
            writer.writerow(asdict(record))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the unique DMD-ready human gait trajectory dataset across gait-signatures repos.")
    parser.add_argument("--old-data-dir", type=Path, default=DEFAULT_OLD_DATA)
    parser.add_argument("--original-repo", type=Path, default=DEFAULT_ORIGINAL_REPO)
    parser.add_argument("--hya-repo", type=Path, default=DEFAULT_HYA_REPO)
    parser.add_argument("--poststroke-repo", type=Path, default=DEFAULT_POSTSTROKE_REPO)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duplicate-rms", type=float, default=DEFAULT_DUPLICATE_RMS)
    parser.add_argument("--duplicate-margin", type=float, default=DEFAULT_DUPLICATE_MARGIN)
    args = parser.parse_args()

    _, old_rows = load_legacy_cache(args.old_data_dir)
    subject_groups = legacy_subject_groups(old_rows)
    hya_data, hya_records = load_hya(args.hya_repo, args.old_data_dir)
    post_data, post_records = load_poststroke(args.poststroke_repo, subject_groups)
    original_data, original_records = load_original_gait_signatures(args.original_repo, subject_groups)

    unique_data, unique_records, duplicates, exact_hash_duplicates = build_unique_dataset(
        [(hya_data, hya_records), (post_data, post_records), (original_data, original_records)],
        args.duplicate_rms,
        args.duplicate_margin,
    )

    out_prefix = args.output_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_prefix.with_suffix(".npy"), unique_data)
    write_records(out_prefix.with_name(out_prefix.name + "_metadata.csv"), unique_records)
    write_duplicates(out_prefix.with_name(out_prefix.name + "_skipped_duplicates.csv"), duplicates)

    source_counts = {
        "GaitSignatures_HealthyYoungAdultStudy": len(hya_records),
        "GaitSignatures_PostStrokeGaitBiomechanics": len(post_records),
        "gait_signatures": len(original_records),
    }
    kept_counts: dict[str, int] = {}
    for record in unique_records:
        kept_counts[record.source] = kept_counts.get(record.source, 0) + 1
    skipped_counts: dict[str, int] = {}
    for record in duplicates:
        skipped_counts[record.skipped_source] = skipped_counts.get(record.skipped_source, 0) + 1

    duplicate_rms = [record.nearest_rms for record in duplicates if np.isfinite(record.nearest_rms)]
    duplicate_ratios = [record.second_to_best_rms_ratio for record in duplicates if np.isfinite(record.second_to_best_rms_ratio)]
    report_path = out_prefix.with_name(out_prefix.name + "_report.txt")
    with report_path.open("w") as f:
        f.write("Human gait unique-trial superset build report\n")
        f.write("=============================================\n\n")
        f.write(f"HealthyYoungAdultStudy repo: {args.hya_repo} @ {git_commit(args.hya_repo)}\n")
        f.write(f"PostStrokeGaitBiomechanics repo: {args.poststroke_repo} @ {git_commit(args.poststroke_repo)}\n")
        f.write(f"Original gait_signatures repo: {args.original_repo} @ {git_commit(args.original_repo)}\n")
        f.write(f"Legacy local cache for omitted HYA raw trajectories: {args.old_data_dir}\n\n")
        f.write("Candidate DMD-ready trajectory counts:\n")
        for source, count in source_counts.items():
            f.write(f"  {source}: {count}\n")
        f.write(f"  total candidates: {sum(source_counts.values())}\n\n")
        f.write(f"Duplicate RMS threshold: {args.duplicate_rms}\n")
        f.write(f"Duplicate second/best RMS margin threshold: {args.duplicate_margin}\n")
        f.write(f"Exact rounded content-hash duplicates: {exact_hash_duplicates}\n")
        f.write(f"Skipped numeric duplicates: {len(duplicates)}\n")
        if duplicate_rms:
            f.write(
                "Skipped duplicate nearest RMS: "
                f"min={min(duplicate_rms):.6g}, median={float(np.median(duplicate_rms)):.6g}, max={max(duplicate_rms):.6g}\n"
            )
            f.write(
                "Skipped duplicate second/best RMS ratio: "
                f"min={min(duplicate_ratios):.6g}, median={float(np.median(duplicate_ratios)):.6g}, max={max(duplicate_ratios):.6g}\n"
            )
        f.write("\nKept unique trials by source:\n")
        for source in source_counts:
            f.write(f"  {source}: {kept_counts.get(source, 0)}\n")
        f.write("Skipped duplicates by source:\n")
        for source in source_counts:
            f.write(f"  {source}: {skipped_counts.get(source, 0)}\n")
        f.write(f"\nUnique dataset shape: {tuple(unique_data.shape)}\n")
        f.write("\nNotes:\n")
        f.write("  The public HYA repository lists 150 trials but omits the raw trajectory CSV files; those trajectories are matched from the existing local legacy cache.\n")
        f.write("  PostStrokeGaitBiomechanics contains all 72 original gait_signatures public trajectories plus additional AB/ST trials, so PostStroke is kept preferentially.\n")

    print(f"candidates: {sum(source_counts.values())}")
    print(f"unique trials: {unique_data.shape[0]}")
    print(f"shape: {unique_data.shape}")
    print(f"skipped duplicates: {len(duplicates)}")
    print(f"wrote {out_prefix.with_suffix('.npy')}")
    print(f"wrote {out_prefix.with_name(out_prefix.name + '_metadata.csv')}")
    print(f"wrote {out_prefix.with_name(out_prefix.name + '_skipped_duplicates.csv')}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
