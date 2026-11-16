"""Sub-response analysis of three PISA 2015 items with nonmonotone ICCs.

Reads the public PISA 2015 cognitive file (CY6_MS_CMB_STU_COG, SAS format) and tabulates, for
Reading 22 (CR466Q03), Science 81 (CS649Q01) and Mathematics 42 (CM998Q04):
  1. the response patterns of the raw sub-responses against the scored response, and
  2. the proportion correct of every part (or the share of every option) by decile of the rest score.

Rest score: proportion of full-credit responses on the other computer-scored items ("C...S", Scored Response)
of the same domain; respondents with fewer than MIN_REST such items are left out. Full credit is the highest
valid code of an item; partial credit counts as not correct. Special missing codes are read as missing.

Usage:
    python pisa2015_subresponse.py /path/to/cy6_ms_cmb_stu_cog.sas7bdat [--out DIR] [--cache FILE]

Requires pandas and numpy only. One pass over the 5 GB file takes about a minute.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ITEMS = [  # (label, scored response, raw sub-responses)
    ("Reading 22 (CR466Q03, Work Right Q03)", "CR466Q03S", ["DR466Q03RA", "DR466Q03RB", "DR466Q03RC"]),
    ("Science 81 (CS649Q01, Weather Balloon Q01)", "CS649Q01S", ["DS649Q01R"]),
    ("Mathematics 42 (CM998Q04, Bike Rental Q04)", "CM998Q04S", ["DM998Q04RA", "DM998Q04RB"]),
]
SCORED = re.compile(r"^C[RSM]\d{3}Q\d+S$")
MIN_REST = 8
N_GROUPS = 10
CHUNK = 4000


def extract(path: Path) -> pd.DataFrame:
    """Scored responses of all computer-based items and the raw sub-responses of the three items, for every
    student with a scored response to at least one of them."""
    targets = [scored for _, scored, _ in ITEMS]
    raw = [column for _, _, parts in ITEMS for column in parts]
    columns, kept = None, []
    for chunk in pd.read_sas(path, format="sas7bdat", chunksize=CHUNK, encoding=None):
        if columns is None:
            columns = [c for c in chunk.columns if SCORED.match(c)] + raw
        part = chunk[columns]
        kept.append(part[part[targets].notna().any(axis=1)].astype("float32"))
    return pd.concat(kept).reset_index(drop=True)


def rest_score(full: pd.DataFrame, scored: str) -> pd.Series:
    others = [c for c in full.columns if c[1] == scored[1] and c != scored]
    count = full[others].notna().sum(axis=1)
    return (full[others].sum(axis=1) / count.replace(0, np.nan)).where(count >= MIN_REST)


def analyze(df: pd.DataFrame, full: pd.DataFrame, label: str, scored: str, parts: list[str]) -> dict:
    valid = df[scored].isin([0, 1])
    credit = df[scored].eq(1)
    values = {part: sorted(df.loc[valid, part].dropna().unique().astype(int)) for part in parts}

    # 1. response patterns against the scored response (every respondent with a scored response)
    code = lambda v: "-" if pd.isna(v) else str(int(v))
    pattern = df.loc[valid, parts].apply(lambda row: "".join(code(v) for v in row), axis=1)
    patterns = pd.crosstab(pattern, credit[valid].map({True: "full credit", False: "no credit"}))
    patterns["share"] = patterns.sum(axis=1) / valid.sum()
    patterns = patterns.sort_values("share", ascending=False)
    key = patterns["full credit"].idxmax()
    answered = patterns.index[~patterns.index.str.contains("-")]
    modal_error = patterns.loc[answered].drop(index=key)["share"].idxmax()

    # 2. rest-score groups (respondents who answered every part and have a rest score)
    rest = rest_score(full, scored)
    complete = valid & rest.notna() & df[parts].notna().all(axis=1)
    d = df.loc[complete, parts].astype(int)
    group = pd.qcut(rest[complete].rank(method="first"), N_GROUPS, labels=False) + 1
    table = pd.DataFrame({"n": d.groupby(group).size(), "rest_score": rest[complete].groupby(group).mean()})
    if len(parts) == 1:  # multiple choice: share of every option
        for option in values[parts[0]]:
            name = f"option_{option}" + (" (key)" if str(option) == key else "")
            table[name] = d[parts[0]].eq(option).groupby(group).mean()
    else:                # multiple yes/no judgments: proportion correct of every part
        correct = pd.DataFrame({part: d[part].eq(int(k)) for part, k in zip(parts, key)})
        for part in parts:
            table[f"{part} correct"] = correct[part].groupby(group).mean()
        for part in parts:  # each part among respondents correct on all the other parts
            others = correct.drop(columns=part).all(axis=1)
            table[f"{part} | others correct"] = correct.loc[others, part].groupby(group[others]).mean()
        joined = d.astype(str).agg("".join, axis=1)
        table[f"pattern {modal_error}"] = joined.eq(modal_error).groupby(group).mean()
    table["full credit"] = credit[complete].groupby(group).mean()
    table.index.name = "rest-score decile"

    return {"label": label, "n_scored": int(valid.sum()), "p_full": float(credit[valid].mean()), "key": key,
            "values": values, "patterns": patterns, "n_complete": int(complete.sum()), "deciles": table}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cog_file", type=Path, help="cy6_ms_cmb_stu_cog.sas7bdat")
    parser.add_argument("--out", type=Path, default=None, help="directory for the tables as CSV files")
    parser.add_argument("--cache", type=Path, default=None, help="pickle of the extracted columns (made if missing)")
    args = parser.parse_args()

    if args.cache is not None and args.cache.exists():
        df = pd.read_pickle(args.cache)
    else:
        df = extract(args.cog_file)
        if args.cache is not None:
            df.to_pickle(args.cache)

    scored = [c for c in df.columns if SCORED.match(c)]
    full = df[scored].eq(df[scored].max()).astype("float32").where(df[scored].notna())

    pd.set_option("display.width", 250, "display.max_columns", 30)
    for label, item, parts in ITEMS:
        result = analyze(df, full, label, item, parts)
        print(f"\n=== {result['label']} ===")
        print(f"scored respondents: {result['n_scored']:,}; full credit: {result['p_full']:.4f}; "
              f"key (codes of {', '.join(parts)}): {result['key']}")
        print("\nresponse patterns ('-': no response)")
        print(result["patterns"].head(10).round(4).to_string())
        print(f"\nby rest-score decile (n = {result['n_complete']:,}; all parts answered, "
              f"at least {MIN_REST} other items of the domain)")
        print(result["deciles"].round(3).to_string())
        if args.out is not None:
            args.out.mkdir(parents=True, exist_ok=True)
            result["patterns"].to_csv(args.out / f"{item}_patterns.csv")
            result["deciles"].to_csv(args.out / f"{item}_deciles.csv")


if __name__ == "__main__":
    main()
