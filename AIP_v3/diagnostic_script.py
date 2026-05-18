"""
Dataset diagnostic script.
Run from AIP_v3/ directory: python diagnostic_script.py
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from label_mapping import (
    ALL_TAGS, PERIOD_TAGS, STYLE_TAGS, MOOD_TAGS,
    ELEMENT_TAGS, TECHNIQUE_TAGS, SUBJECT_TAGS,
    GENRE_TO_TAGS,
)

CSV_PATH    = "multilabel_dataset.csv"
ARCHIVE_DIR = "../archive"
INPUT_CSV   = "../archive/classes.csv"

TAG_GROUPS = {
    "Period":    PERIOD_TAGS,
    "Style":     STYLE_TAGS,
    "Mood":      MOOD_TAGS,
    "Element":   ELEMENT_TAGS,
    "Technique": TECHNIQUE_TAGS,
    "Subject":   SUBJECT_TAGS,
}

SEP = "-" * 70


def sep(title=""):
    if title:
        pad = (70 - len(title) - 2) // 2
        print(f"\n{'-'*pad} {title} {'-'*pad}")
    else:
        print(SEP)


# ── 1. CSV existence ──────────────────────────────────────────────────────────
def check_csv():
    sep("CSV file")
    if not os.path.exists(CSV_PATH):
        print(f"[MISSING] {CSV_PATH} — run generate_multilabel_csv.py first")
        sys.exit(1)
    size_mb = os.path.getsize(CSV_PATH) / 1024 / 1024
    print(f"Found: {CSV_PATH}  ({size_mb:.2f} MB)")
    return pd.read_csv(CSV_PATH)


# ── 2. Shape & columns ───────────────────────────────────────────────────────
def check_shape(df):
    sep("Shape & columns")
    tag_cols = [c for c in df.columns if c not in ("filename", "subset")]
    print(f"Rows        : {len(df):,}")
    print(f"Tag columns : {len(tag_cols)}  (expected {len(ALL_TAGS)})")
    missing_tags = set(ALL_TAGS) - set(tag_cols)
    extra_tags   = set(tag_cols) - set(ALL_TAGS)
    if missing_tags:
        print(f"[WARN] Tags missing from CSV : {sorted(missing_tags)}")
    if extra_tags:
        print(f"[WARN] Extra cols not in ALL_TAGS: {sorted(extra_tags)}")
    if not missing_tags and not extra_tags:
        print("Tag columns match ALL_TAGS exactly — OK")
    return tag_cols


# ── 3. Train / test split ────────────────────────────────────────────────────
def check_split(df):
    sep("Train / test split")
    counts = df["subset"].value_counts()
    total  = len(df)
    for subset, n in counts.items():
        print(f"  {subset:<8}: {n:>7,}  ({100*n/total:.1f}%)")
    if "train" not in counts or "test" not in counts:
        print("[WARN] Expected both 'train' and 'test' subsets")


# ── 4. Missing image files ───────────────────────────────────────────────────
def check_missing_files(df):
    sep("Image file existence")
    missing = [f for f in df["filename"] if not os.path.exists(os.path.join(ARCHIVE_DIR, f))]
    if missing:
        print(f"[WARN] {len(missing):,} filenames in CSV have no image on disk:")
        for f in missing[:10]:
            print(f"    {f}")
        if len(missing) > 10:
            print(f"    ... and {len(missing)-10} more")
    else:
        print(f"All {len(df):,} files exist on disk — OK")
    return missing


# ── 5. Labels-per-image distribution ────────────────────────────────────────
def check_label_counts(df, tag_cols):
    sep("Labels per image")
    counts = df[tag_cols].sum(axis=1)
    print(f"  Min   : {counts.min()}")
    print(f"  Max   : {counts.max()}")
    print(f"  Mean  : {counts.mean():.2f}")
    print(f"  Median: {counts.median():.0f}")
    zero_label = (counts == 0).sum()
    if zero_label:
        print(f"[WARN] {zero_label} rows have ZERO tags")
    else:
        print("No zero-label rows — OK")
    return counts


# ── 6. Tag frequency & class imbalance ───────────────────────────────────────
def check_tag_frequency(df, tag_cols):
    sep("Tag frequency (all tags)")
    tag_sums = df[tag_cols].sum().sort_values(ascending=False)
    total    = len(df)

    print(f"{'Tag':<35} {'Count':>7}  {'%':>6}")
    print("-" * 52)
    for tag, cnt in tag_sums.items():
        bar = "#" * int(30 * cnt / total)
        print(f"  {tag:<33} {cnt:>7,}  {100*cnt/total:>5.1f}%  {bar}")

    # imbalance ratio
    ratio = tag_sums.max() / max(tag_sums.min(), 1)
    print(f"\nImbalance ratio (max/min count): {ratio:.1f}x")
    if ratio > 50:
        print("[WARN] Very high imbalance — consider weighted loss or oversampling")
    return tag_sums


# ── 7. Per-group breakdown ───────────────────────────────────────────────────
def check_group_coverage(df, tag_cols):
    sep("Per-group tag coverage")
    for group, tags in TAG_GROUPS.items():
        present = [t for t in tags if t in tag_cols]
        group_df = df[present]
        # fraction of images that have at least one tag from this group
        covered = (group_df.sum(axis=1) > 0).sum()
        print(f"\n  {group} ({len(present)} tags):")
        print(f"    Images with ≥1 tag: {covered:,}  ({100*covered/len(df):.1f}%)")
        sums = group_df.sum().sort_values(ascending=False)
        for tag, cnt in sums.items():
            print(f"      {tag:<30} {cnt:>7,}  ({100*cnt/len(df):.1f}%)")


# ── 8. Tag co-occurrence spot-check ─────────────────────────────────────────
def check_cooccurrence(df, tag_cols, top_n=10):
    sep(f"Top-{top_n} co-occurring tag pairs")
    mat = df[tag_cols].values.astype(bool)
    n   = len(tag_cols)
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            co = (mat[:, i] & mat[:, j]).sum()
            if co > 0:
                pairs.append((co, tag_cols[i], tag_cols[j]))
    pairs.sort(reverse=True)
    print(f"  {'Tag A':<30} {'Tag B':<30} {'Co-occur':>8}")
    print("  " + "-" * 70)
    for co, a, b in pairs[:top_n]:
        print(f"  {a:<30} {b:<30} {co:>8,}")


# ── 9. Archive folder structure ───────────────────────────────────────────────
def check_archive():
    sep("Archive folder structure")
    if not os.path.isdir(ARCHIVE_DIR):
        print(f"[MISSING] Archive directory: {ARCHIVE_DIR}")
        return
    subdirs = [d for d in os.listdir(ARCHIVE_DIR)
               if os.path.isdir(os.path.join(ARCHIVE_DIR, d))]
    if subdirs:
        print(f"Subfolders found ({len(subdirs)}):")
        for d in sorted(subdirs):
            n = len(os.listdir(os.path.join(ARCHIVE_DIR, d)))
            print(f"    {d:<40} {n:>6} files")
    else:
        # flat structure — count files by extension
        files = os.listdir(ARCHIVE_DIR)
        exts  = {}
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            exts[ext] = exts.get(ext, 0) + 1
        print(f"Flat archive — {len(files):,} total files")
        for ext, cnt in sorted(exts.items(), key=lambda x: -x[1]):
            print(f"    {ext or '(no ext)':<15} {cnt:>7,}")


# ── 10. Input CSV genre coverage ─────────────────────────────────────────────
def check_genre_coverage():
    sep("Genre coverage in classes.csv")
    if not os.path.exists(INPUT_CSV):
        print(f"[SKIP] {INPUT_CSV} not found")
        return
    import csv, ast
    genre_counts = {}
    unknown = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                genres = ast.literal_eval(row.get("genre", "[]"))
                genres = [g.strip().replace(" ", "_") for g in genres]
            except Exception:
                genres = [row.get("genre", "").strip().replace(" ", "_")]
            for g in genres:
                genre_counts[g] = genre_counts.get(g, 0) + 1
                if g not in GENRE_TO_TAGS and g not in ("", "unknown"):
                    unknown.append(g)

    print(f"  {'Genre':<40} {'Images':>7}  {'Mapped?':>8}")
    print("  " + "-" * 60)
    for g, cnt in sorted(genre_counts.items(), key=lambda x: -x[1]):
        mapped = "YES" if g in GENRE_TO_TAGS else "NO"
        flag   = "  ← unmapped" if mapped == "NO" else ""
        print(f"  {g:<40} {cnt:>7,}  {mapped:>8}{flag}")
    if unknown:
        unique_unknown = sorted(set(unknown))
        print(f"\n[WARN] {len(unique_unknown)} genre name(s) not in GENRE_TO_TAGS:")
        for u in unique_unknown:
            print(f"    {u}")


# ── 11. Plots ─────────────────────────────────────────────────────────────────
def plot_diagnostics(df, tag_cols, label_counts):
    sep("Generating plots")
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # (a) Labels-per-image histogram
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.hist(label_counts, bins=range(int(label_counts.max()) + 2), edgecolor="black", color="steelblue")
    ax0.set_title("Labels per image")
    ax0.set_xlabel("Tag count"); ax0.set_ylabel("Images")

    # (b) Train / test bar
    ax1 = fig.add_subplot(gs[0, 1])
    split_counts = df["subset"].value_counts()
    ax1.bar(split_counts.index, split_counts.values, color=["#2196F3", "#FF9800"])
    for i, (_, val) in enumerate(split_counts.items()):
        ax1.text(i, val + 50, f"{val:,}", ha="center", fontsize=9)
    ax1.set_title("Train / Test split"); ax1.set_ylabel("Images")

    # (c) Top-30 tag frequencies
    ax2 = fig.add_subplot(gs[1, :])
    tag_sums = df[tag_cols].sum().sort_values(ascending=False)
    top30 = tag_sums.head(30)
    colors = []
    for tag in top30.index:
        if tag in PERIOD_TAGS:    colors.append("#9C27B0")
        elif tag in STYLE_TAGS:   colors.append("#2196F3")
        elif tag in MOOD_TAGS:    colors.append("#FF9800")
        elif tag in ELEMENT_TAGS: colors.append("#4CAF50")
        elif tag in TECHNIQUE_TAGS: colors.append("#F44336")
        else:                     colors.append("#607D8B")
    ax2.bar(range(len(top30)), top30.values, color=colors)
    ax2.set_xticks(range(len(top30)))
    ax2.set_xticklabels(top30.index, rotation=45, ha="right", fontsize=8)
    ax2.set_title("Top 30 tag frequencies  (purple=Period, blue=Style, orange=Mood, green=Element, red=Technique, grey=Subject)")
    ax2.set_ylabel("Count")

    # (d) Per-group image coverage
    ax3 = fig.add_subplot(gs[2, 0])
    group_coverage = {}
    for group, tags in TAG_GROUPS.items():
        present = [t for t in tags if t in tag_cols]
        group_coverage[group] = (df[present].sum(axis=1) > 0).mean() * 100
    ax3.barh(list(group_coverage.keys()), list(group_coverage.values()), color="teal")
    ax3.set_xlabel("% of images with ≥1 tag")
    ax3.set_title("Group coverage")
    ax3.set_xlim(0, 105)
    for i, (_, v) in enumerate(group_coverage.items()):
        ax3.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)

    # (e) Bottom-20 tag frequencies (rare tags)
    ax4 = fig.add_subplot(gs[2, 1])
    bottom20 = tag_sums.tail(20)
    ax4.barh(range(len(bottom20)), bottom20.values, color="salmon")
    ax4.set_yticks(range(len(bottom20)))
    ax4.set_yticklabels(bottom20.index, fontsize=8)
    ax4.set_title("Bottom 20 tags (rarest)")
    ax4.set_xlabel("Count")

    out = "dataset_diagnostics.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  DATASET DIAGNOSTIC REPORT")
    print("=" * 70)

    df       = check_csv()
    tag_cols = check_shape(df)
    check_split(df)
    missing  = check_missing_files(df)
    lcounts  = check_label_counts(df, tag_cols)
    tag_sums = check_tag_frequency(df, tag_cols)
    check_group_coverage(df, tag_cols)
    check_cooccurrence(df, tag_cols)
    check_archive()
    check_genre_coverage()
    plot_diagnostics(df, tag_cols, lcounts)

    sep("Summary")
    issues = []
    if missing:
        issues.append(f"{len(missing)} files in CSV are missing on disk")
    if (lcounts == 0).any():
        issues.append(f"{(lcounts==0).sum()} rows have zero tags")
    ratio = tag_sums.max() / max(tag_sums.min(), 1)
    if ratio > 50:
        issues.append(f"High tag imbalance ({ratio:.0f}x)")
    if issues:
        print("[Issues found]")
        for i in issues:
            print(f"  - {i}")
    else:
        print("No critical issues found.")
    print()


if __name__ == "__main__":
    main()
