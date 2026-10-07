"""Report-derived weak labels from a local LLM served by llama.cpp.

Each report is read by the model, which assigns one category per target
(present / mild / uncertain / absent / not_mentioned) under a JSON-schema
constraint. Categories are turned into probabilities calibrated on the gold
studies, whose reports are labelled too. Only aggregate numbers are printed,
never report text.

    # start the model server first (see README), then:
    python src/weak_labels.py run --tag gemma4-12b --studies gold
    python src/weak_labels.py eval --tag gemma4-12b
    python src/weak_labels.py run --tag gemma4-12b --studies all
    python src/weak_labels.py export --tag gemma4-12b --out E:/Documents/Projects/data/subset/weak_labels.csv
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import TARGETS

CATEGORIES = ["present", "mild", "uncertain", "absent", "not_mentioned"]
# Ordinal score per category, used to rank studies when comparing models.
ORDINAL = {"present": 1.0, "mild": 0.6, "uncertain": 0.5, "not_mentioned": 0.15, "absent": 0.0}

SYSTEM = """You label knee MRI radiology reports. Reports come from many countries and may be written in any \
language (English, Spanish, Turkish, Croatian, Greek, German, Bulgarian, Dutch, French, ...). Read the whole report, \
including findings and impression, and return one category for each of 12 targets.

Categories:
- present: the report states the finding is there.
- mild: the finding is reported but described as minimal, trace, mild, low-grade, small, early, or as a precursor \
form (sprain without tear, intrasubstance degeneration without a tear reaching the surface, cartilage softening or \
thinning only).
- uncertain: the report says possible, probable, suspected, equivocal, cannot be excluded, or similar.
- absent: the report explicitly says the structure is normal or intact, or that the finding is not present.
- not_mentioned: the report does not address this target at all.

Targets:
- ACL: tear of the anterior cruciate ligament (partial or complete, including a torn ACL graft).
- MCL: injury of the medial collateral ligament (sprain, partial tear, complete tear).
- Medial Meniscus: tear of the medial meniscus (any type: horizontal, radial, longitudinal, complex, root, \
bucket-handle, flap, maceration).
- Lateral Meniscus: tear of the lateral meniscus (same types).
- Medial OA: osteoarthritis of the medial femorotibial compartment (cartilage loss or defects, osteophytes, \
joint space narrowing, subchondral sclerosis or cysts in the medial compartment).
- Lateral OA: the same for the lateral femorotibial compartment.
- PF OA: the same for the patellofemoral compartment (patella, trochlea), including chondromalacia patellae.
- Effusion: joint effusion. Trace, minimal or physiological fluid counts as mild; small, moderate or large \
counts as present.
- Synovitis: synovitis, synovial thickening or proliferation, or Hoffa fat pad oedema/synovitis.
- Baker's: Baker's (popliteal) cyst.
- Contusion: bone contusion, bone bruise, or traumatic bone marrow oedema (not degenerative subchondral oedema \
from osteoarthritis).
- Fracture: any fracture, including osteochondral, impaction, avulsion (e.g. Segond), stress and subchondral \
insufficiency fractures.

Judge each target on its own; a finding in one compartment says nothing about the others. Use the report's own \
wording, not what you would expect clinically.

Example 1 (English): "Complete ACL tear. Bone bruises of the lateral femoral condyle and posterolateral tibial \
plateau. Moderate effusion. Menisci intact. MCL grade 1 sprain." ->
ACL present, MCL mild, Medial Meniscus absent, Lateral Meniscus absent, Medial OA not_mentioned, Lateral OA \
not_mentioned, PF OA not_mentioned, Effusion present, Synovitis not_mentioned, Baker's not_mentioned, Contusion \
present, Fracture not_mentioned.

Example 2 (German): "Horizontalriss im Innenmeniskushinterhorn. Höhergradige Chondropathie medial femorotibial mit \
Osteophyten. Kleine Baker-Zyste. Kreuz- und Kollateralbänder intakt. Kein Gelenkerguss." ->
ACL absent, MCL absent, Medial Meniscus present, Lateral Meniscus not_mentioned, Medial OA present, Lateral OA \
not_mentioned, PF OA not_mentioned, Effusion absent, Synovitis not_mentioned, Baker's mild, Contusion \
not_mentioned, Fracture not_mentioned.

Example 3 (Spanish): "Posible rotura del cuerno posterior del menisco externo. Condromalacia rotuliana grado II. \
Mínimo derrame articular. Sin fracturas." ->
ACL not_mentioned, MCL not_mentioned, Medial Meniscus not_mentioned, Lateral Meniscus uncertain, Medial OA \
not_mentioned, Lateral OA not_mentioned, PF OA present, Effusion mild, Synovitis not_mentioned, Baker's \
not_mentioned, Contusion not_mentioned, Fracture absent."""

SCHEMA = {
    "type": "object",
    "properties": {t: {"type": "string", "enum": CATEGORIES} for t in TARGETS},
    "required": TARGETS,
    "additionalProperties": False,
}

DATA = Path(r"E:\Documents\Projects\data")
CACHE = DATA / "weak_labels"


def label_report(client, url, report):
    body = {
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Report:\n{report}"}],
        "temperature": 0,
        "max_tokens": 400,
        "response_format": {"type": "json_schema", "json_schema": {"name": "labels", "schema": SCHEMA}},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = client.post(f"{url}/v1/chat/completions", json=body)
    r.raise_for_status()
    labels = json.loads(r.json()["choices"][0]["message"]["content"])
    if set(labels) != set(TARGETS) or not set(labels.values()) <= set(CATEGORIES):
        raise ValueError("output does not match the schema")
    return labels


def load_cache(path):
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["id"]: r for r in rows if "labels" in r}


def select_studies(train, which):
    gold = train[TARGETS].notna().all(axis=1)
    if which == "gold":
        return train[gold]
    if which == "subset":
        ids = pd.read_csv(DATA / "subset" / "train.csv", usecols=["StudyInstanceUID"])["StudyInstanceUID"]
        return train[train["StudyInstanceUID"].isin(ids)]
    return train


def cmd_run(args, train):
    path = CACHE / f"{args.tag}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = load_cache(path)
    todo = select_studies(train, args.studies)
    todo = todo[~todo["StudyInstanceUID"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"[{args.tag}] {len(done)} cached, {len(todo)} to label with {args.parallel} parallel requests")

    t0, n_ok, n_err = time.time(), 0, 0
    with httpx.Client(timeout=300) as client, ThreadPoolExecutor(args.parallel) as pool, \
            open(path, "a", encoding="utf-8") as out:
        futures = {pool.submit(label_report, client, args.url, r.Report): r.StudyInstanceUID
                   for r in todo.itertuples()}
        for i, fut in enumerate(as_completed(futures), 1):
            sid = futures[fut]
            try:
                out.write(json.dumps({"id": sid, "labels": fut.result()}) + "\n")
                n_ok += 1
            except Exception as ex:
                print(f"  error on study #{i}: {type(ex).__name__}: {str(ex)[:120]}")
                n_err += 1
            out.flush()
            if i % 50 == 0 or i == len(todo):
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(todo)}  {rate * 60:.1f} reports/min  eta {(len(todo) - i) / rate / 60:.0f} min",
                      flush=True)
    print(f"[{args.tag}] labelled {n_ok}, errors {n_err}, {time.time() - t0:.0f}s")


def gold_table(train, cache):
    """Long table (study, target, category, gold) over gold studies the model has labelled."""
    gold = train[train[TARGETS].notna().all(axis=1)].set_index("StudyInstanceUID")
    rows = [(sid, t, cache[sid]["labels"][t], int(gold.loc[sid, t]))
            for sid in gold.index if sid in cache for t in TARGETS]
    return pd.DataFrame(rows, columns=["id", "target", "cat", "gold"])


def calibrate(tab, k=4.0):
    """P(gold = 1 | target, category), shrunk towards the category's rate pooled over all targets."""
    pooled = tab.groupby("cat")["gold"].agg(["sum", "size"])
    base = tab["gold"].mean()
    pooled_p = (pooled["sum"] + base) / (pooled["size"] + 1)
    per = tab.groupby(["target", "cat"])["gold"].agg(["sum", "size"])
    table = {}
    for t in TARGETS:
        for c in CATEGORIES:
            p0 = pooled_p.get(c, base)
            s, n = per.loc[(t, c)] if (t, c) in per.index else (0, 0)
            table[(t, c)] = float((s + k * p0) / (n + k))
    return table


def cmd_eval(args, train):
    cache = load_cache(CACHE / f"{args.tag}.jsonl")
    tab = gold_table(train, cache)
    n = tab["id"].nunique()
    print(f"[{args.tag}] gold studies labelled: {n}/58")
    if n == 0:
        return
    tab["score"] = tab["cat"].map(ORDINAL)
    tab["pred"] = tab["cat"].isin(["present", "mild"]).astype(int)

    rows = []
    for t, g in tab.groupby("target", sort=False):
        auc = roc_auc_score(g["gold"], g["score"]) if g["gold"].nunique() == 2 else np.nan
        tp = int(((g.pred == 1) & (g.gold == 1)).sum())
        rows.append({"target": t, "pos": int(g.gold.sum()), "AUC": auc,
                     "sens": tp / max(1, g.gold.sum()), "spec": int(((g.pred == 0) & (g.gold == 0)).sum()) /
                     max(1, (g.gold == 0).sum()), "acc": float((g.pred == g.gold).mean())})
    res = pd.DataFrame(rows).set_index("target")
    print(res.round(3).to_string())
    print(f"macro AUC {res['AUC'].mean():.3f} | macro sens {res['sens'].mean():.3f} | "
          f"macro spec {res['spec'].mean():.3f} | accuracy {tab['pred'].eq(tab['gold']).mean():.3f}")
    print("\ncategory vs gold (all targets pooled):")
    print(pd.crosstab(tab["cat"], tab["gold"]).reindex(CATEGORIES).fillna(0).astype(int).to_string())


def cmd_export(args, train):
    cache = load_cache(CACHE / f"{args.tag}.jsonl")
    cal = calibrate(gold_table(train, cache))
    rows = []
    for sid, r in cache.items():
        labels = r["labels"]
        if all(v == "not_mentioned" for v in labels.values()):
            continue  # empty or technique-only report: no information
        rows.append({"StudyInstanceUID": sid, **{t: cal[(t, labels[t])] for t in TARGETS}})
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} studies to {args.out}")
    print("calibrated P(positive | category):")
    print(pd.DataFrame({c: {t: cal[(t, c)] for t in TARGETS} for c in CATEGORIES}).round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "eval", "export"])
    ap.add_argument("--tag", required=True, help="name of the model run, used for the cache file")
    ap.add_argument("--data", default=str(DATA / "rsna-knee-abnormality-detection" / "train.csv"))
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--studies", default="gold", choices=["gold", "subset", "all"])
    ap.add_argument("--parallel", type=int, default=4, help="match llama-server --parallel")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(DATA / "subset" / "weak_labels.csv"))
    args = ap.parse_args()
    train = pd.read_csv(args.data)
    {"run": cmd_run, "eval": cmd_eval, "export": cmd_export}[args.cmd](args, train)


if __name__ == "__main__":
    main()
