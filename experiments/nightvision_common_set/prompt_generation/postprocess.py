#!/usr/bin/env python3
"""Build and describe the unified prompt sets from prompt_generation results.

Reads every result JSON under the input directories (default: results/5_char
and results/8_char, next to this script), groups them by model stem, and for
each stem:

  1. merge   - concatenate every file for the stem (raw search outputs named
               <stem>_<N>tokens.json, or merged <stem>.json files)
               and drop exact-duplicate prompt strings, keeping the lowest
               logprob occurrence
  2. dedup   - load the model's tokenizer and drop prompts whose token-id
               sequence duplicates a lower-logprob prompt (skip with
               --no-token-dedup)
  3. select  - sort by logprob ascending and keep the lowest --top-n
  4. write   - ../prompts/<stem>.json as {"prompts": [...]}, the unified prompt
               directory read by collect_logits and (through the collected
               logits) sampling. An existing file is kept unless
               --overwrite is given. --merged-dir additionally writes the full
               result entries with config and summary stats.
  5. count   - render the prompt set's first --count-prompts prompts through the
               model's chat template as a single user message, exactly like
               collect_logits/collect.py does, and average their token counts
               (skip with --no-token-count). The count always describes the
               prompts file that is on disk after step 4.
  6. stats   - logprob / probability percentiles of the selected prompts.

Per-stem counts and stats accumulate across runs in results/prompt_set_summary.json
(keyed by stem, so --models runs update only their stems); from it the script
regenerates results/avg_prompt_tokens.json ({"models": [{"model",
"architecture", "avg_prompt_tokens"}, ...]}) and the probability table
results/probability_summary.txt. A per-run report lands in
results/postprocess_report.json.

--summarize DIR skips the pipeline and only tabulates the summary.probability
block of every result JSON in DIR (raw search outputs or --merged-dir files) to
DIR/<DIR name>_probability_summary.txt.

Usage:
  python postprocess.py                                  # results/{5,8}_char -> ../prompts
  python postprocess.py -d results/8_char                # only one result folder
  python postprocess.py --models qwen3_0.6b smol2_135m   # only these stems
  python postprocess.py --no-token-dedup --overwrite
  python postprocess.py --summarize results/8_char       # just the probability table
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
# Unified prompt directory shared by every nightvision_common_set experiment.
PROMPTS_DIR = SCRIPT_DIR.parent / "prompts"
# architecture / hidden_size per HF model, at the repository root.
MODEL_CONFIGS = SCRIPT_DIR.parents[2] / "model_configs.json"
# The prompt sets are the union of the 5- and 8-character searches.
DEFAULT_DIRS = [RESULTS_DIR / "5_char", RESULTS_DIR / "8_char"]
DEFAULT_TOP_N = 80000
DEFAULT_COUNT_PROMPTS = 20000       # prompts averaged for avg_prompt_tokens
DEFAULT_COUNT_CHUNK = 512           # prompts rendered / tokenized per call
DEFAULT_REPORT = RESULTS_DIR / "postprocess_report.json"
SUMMARY_JSON = RESULTS_DIR / "prompt_set_summary.json"
AVG_TOKENS_JSON = RESULTS_DIR / "avg_prompt_tokens.json"
PROBABILITY_TXT = RESULTS_DIR / "probability_summary.txt"

# Model stem = everything before the "_<N>tokens" part of a raw result filename,
# e.g. "granite_3.1_3b_a800m_8tokens.json" -> "granite_3.1_3b_a800m".
# Files without that suffix (merged outputs) use the whole stem.
STEM_RE = re.compile(r"^(?P<stem>.+?)_\d+tokens(?:_.*)?$")

PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]
PERCENTILE_KEYS = [f"p{p}" for p in PERCENTILES]
TABLE_COLUMNS = ["model", "prompts", "avg_tokens", "lowest", "highest", *PERCENTILE_KEYS]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def model_stem(path):
    """Stem a result file belongs to."""
    name = os.path.splitext(os.path.basename(path))[0]
    m = STEM_RE.match(name)
    return m.group("stem") if m else name


def model_name_from_config(config):
    """HF model name recorded by the search (raw: config.model; merged: config.models)."""
    if not config:
        return None
    models = config.get("models") or ([config["model"]] if config.get("model") else [])
    return models[0] if models else None


def percentile(sorted_values, p):
    if not sorted_values:
        return None
    idx = round((len(sorted_values) - 1) * p / 100)
    return sorted_values[idx]


def stats(values):
    values = sorted(values)
    out = {"lowest": values[0], "highest": values[-1]}
    for p in PERCENTILES:
        out[f"p{p}"] = percentile(values, p)
    return out


def build_summary(results):
    if not results:
        return {"count": 0}
    return {
        "count": len(results),
        "logprob": stats([r["logprob"] for r in results]),
        "probability": stats([r["probability"] for r in results]),
    }


def _tokenizer_classes(model_name, trust_remote_code, local_files_only):
    """AutoTokenizer first, then the class named in tokenizer_config.json.

    transformers 5 maps some model types (e.g. granitemoe) to a generic fast
    backend that needs tokenizer.json. Repos that only ship vocab.json/merges.txt
    (ibm-granite/granite-3.0-3b-a800m-instruct) still load through the concrete
    class in their tokenizer_config.json (GPT2Tokenizer), which rebuilds the BPE
    model from those files."""
    import transformers
    from transformers import AutoTokenizer
    from transformers.utils import cached_file

    yield AutoTokenizer
    try:
        cfg_path = cached_file(model_name, "tokenizer_config.json",
                               local_files_only=local_files_only)
        with open(cfg_path, encoding="utf-8") as f:
            cls_name = json.load(f).get("tokenizer_class")
    except Exception:
        return
    cls = getattr(transformers, cls_name, None) if cls_name else None
    if cls is not None and cls is not AutoTokenizer:
        yield cls


def _load_tokenizer(model_name, trust_remote_code, local_files_only):
    last_err = None
    for cls in _tokenizer_classes(model_name, trust_remote_code, local_files_only):
        try:
            return cls.from_pretrained(model_name, local_files_only=local_files_only,
                                       trust_remote_code=trust_remote_code)
        except Exception as err:
            if last_err is not None:
                print(f"  {cls.__name__} failed too ({type(err).__name__})", flush=True)
            last_err = err
    raise last_err


def load_tokenizer(model_name, trust_remote_code):
    # Prefer the staged local snapshot; fall back to downloading if it isn't there.
    try:
        return _load_tokenizer(model_name, trust_remote_code, local_files_only=True)
    except Exception as local_err:
        print(f"  local tokenizer not found ({type(local_err).__name__}); trying download", flush=True)
        return _load_tokenizer(model_name, trust_remote_code, local_files_only=False)


def tokenize_prompts(tokenizer, prompts, batch_size=4096):
    ids = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        ids.extend(tokenizer(batch, add_special_tokens=False)["input_ids"])
    return ids


def load_model_configs():
    try:
        with open(MODEL_CONFIGS) as f:
            return json.load(f)
    except OSError as err:
        print(f"WARNING: cannot read {MODEL_CONFIGS} ({err}); architecture will be null")
        return {}


def write_merged_json(merged, out_path):
    """Write config/summary indented but each result entry on one compact line,
    so token_ids don't blow up the file size."""
    with open(out_path, "w") as f:
        f.write("{\n")
        f.write('  "config": ' + json.dumps(merged["config"], indent=2, ensure_ascii=True).replace("\n", "\n  ") + ",\n")
        f.write('  "summary": ' + json.dumps(merged["summary"], indent=2, ensure_ascii=True).replace("\n", "\n  ") + ",\n")
        f.write('  "results": [\n')
        results = merged["results"]
        for i, entry in enumerate(results):
            f.write("    " + json.dumps(entry, ensure_ascii=True, separators=(",", ":")))
            f.write(",\n" if i < len(results) - 1 else "\n")
        f.write("  ]\n}\n")


def format_table(columns, rows):
    widths = [max(len(r[i]) for r in [columns, *rows]) for i in range(len(columns))]

    def fmt(row):
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    lines = [fmt(columns), fmt(["-" * w for w in widths])]
    lines.extend(fmt(r) for r in rows)
    return "\n".join(lines) + "\n"


def _fmt_num(x, digits=6):
    return "" if x is None else f"{x:.{digits}f}"


# ---------------------------------------------------------------------------
# pipeline stages
# ---------------------------------------------------------------------------

def collect_result_files(dirs):
    """stem -> list of result JSON paths, scanning only the top level of each dir."""
    paths_by_stem = defaultdict(list)
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            print(f"Skipping missing directory: {d}")
            continue
        for path in sorted(d.glob("*.json")):
            if not path.is_file():
                continue
            paths_by_stem[model_stem(path)].append(path)
    return paths_by_stem


def merge_stem(stem, paths):
    """Load every file for a stem, drop exact-duplicate prompts (lowest logprob wins)."""
    results = []
    sources = []
    model_names = []
    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            print(f"  skipping {path}: not a search results file")
            continue
        config = data.get("config")
        name = model_name_from_config(config)
        if name and name not in model_names:
            model_names.append(name)
        file_results = data["results"]
        for entry in file_results:
            entry.setdefault("source", str(path))
        results.extend(file_results)
        sources.append({"path": str(path), "config": config, "count": len(file_results)})
        print(f"  {path}: {len(file_results)} entries", flush=True)
        del data

    best = {}
    for entry in results:
        p = entry["prompt"]
        if p not in best or entry["logprob"] < best[p]["logprob"]:
            best[p] = entry
    string_removed = len(results) - len(best)
    merged = sorted(best.values(), key=lambda x: x["logprob"])
    return merged, sources, model_names, string_removed


def dedup_by_tokens(results, tokenizer, keep_token_ids):
    """Keep the first (lowest-logprob) prompt for every distinct token-id sequence.

    `results` must already be sorted by logprob ascending."""
    token_ids = tokenize_prompts(tokenizer, [r["prompt"] for r in results])
    seen = set()
    kept = []
    for entry, ids in zip(results, token_ids):
        key = tuple(ids)
        if key in seen:
            continue
        seen.add(key)
        entry["num_tokens"] = len(key)
        if keep_token_ids:
            entry["token_ids"] = list(key)
        kept.append(entry)
    return kept


def average_prompt_tokens(tokenizer, prompts, chunk):
    """Mean token count of every prompt rendered through the model's chat template
    (single user message, so the template's own default system prompt is included),
    matching the way collect_logits/collect.py feeds prompts to the model."""
    kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    total = 0
    for start in range(0, len(prompts), chunk):
        conversations = [[{"role": "user", "content": p}] for p in prompts[start:start + chunk]]
        templated = tokenizer.apply_chat_template(conversations, **kwargs)
        # The template already emits BOS / role tokens, so no extra special tokens.
        ids = tokenizer(templated, add_special_tokens=False)["input_ids"]
        total += sum(len(x) for x in ids)
    return total / len(prompts)


def process_stem(stem, paths, args, model_configs):
    t0 = time.time()
    prompts_path = args.prompts_dir / f"{stem}.json"

    merged, sources, model_names, string_removed = merge_stem(stem, paths)
    if not merged:
        print("  no results, skipping")
        return {"stem": stem, "status": "empty"}
    if len(model_names) > 1:
        print(f"  WARNING: {len(model_names)} model names for {stem}, using {model_names[0]}")
    model_name = model_names[0] if model_names else None
    merged_count = len(merged)

    tokenizer = None
    if (args.token_dedup or args.token_count) and model_name is not None:
        tokenizer = load_tokenizer(model_name, args.trust_remote_code)
    elif args.token_dedup or args.token_count:
        print("  WARNING: no model name in any config; tokenizer dedup and token count skipped")

    token_removed = 0
    if args.token_dedup and tokenizer is not None:
        before = len(merged)
        merged = dedup_by_tokens(merged, tokenizer, keep_token_ids=args.merged_dir is not None)
        token_removed = before - len(merged)

    selected = merged[: args.top_n] if args.top_n > 0 else merged
    selected_prompts = [entry["prompt"] for entry in selected]

    # -- write the prompt set (or keep the one on disk) ----------------------
    if prompts_path.exists() and not args.overwrite:
        prompts_status = "kept_existing"
        with open(prompts_path, "r", encoding="utf-8") as f:
            on_disk = json.load(f)["prompts"]
        if on_disk != selected_prompts:
            print(f"  NOTE: existing {prompts_path.name} differs from this run's selection "
                  f"({len(on_disk)} vs {len(selected_prompts)} prompts); kept the existing file "
                  "(use --overwrite to replace it)")
    else:
        args.prompts_dir.mkdir(parents=True, exist_ok=True)
        with open(prompts_path, "w") as f:
            json.dump({"prompts": selected_prompts}, f, indent=2, ensure_ascii=True)
        prompts_status = "written"
        on_disk = selected_prompts

    merged_path = None
    if args.merged_dir is not None:
        args.merged_dir.mkdir(parents=True, exist_ok=True)
        merged_path = args.merged_dir / f"{stem}.json"
        write_merged_json({
            "config": {
                "model_stem": stem,
                "models": model_names,
                "sources": sources,
                "string_dedup": {"before": sum(s["count"] for s in sources),
                                 "after": merged_count, "removed": string_removed},
                "token_dedup": ({"tokenizer": model_name, "key": "token_ids",
                                 "add_special_tokens": False,
                                 "before": merged_count, "after": merged_count - token_removed,
                                 "removed": token_removed}
                                if args.token_dedup and tokenizer is not None else None),
                "top_n": args.top_n,
                "sort": "logprob ascending",
            },
            "summary": build_summary(selected),
            "results": selected,
        }, merged_path)

    # -- describe the prompt set ---------------------------------------------
    avg_tokens = None
    counted = 0
    if args.token_count and tokenizer is not None and on_disk:
        to_count = on_disk[: args.count_prompts] if args.count_prompts > 0 else on_disk
        counted = len(to_count)
        avg_tokens = average_prompt_tokens(tokenizer, to_count, args.count_chunk)

    summary = build_summary(selected)
    arch = model_configs.get(model_name, {}).get("architecture") if model_name else None

    print(f"  {sum(s['count'] for s in sources)} -> {merged_count} (string dedup -{string_removed})"
          f" -> {merged_count - token_removed} (token dedup -{token_removed})"
          f" -> {len(selected)} selected; prompts file {prompts_status}"
          + (f"; avg prompt tokens={avg_tokens:.2f} over {counted}" if avg_tokens is not None else "")
          + f" ({time.time() - t0:.1f}s) -> {prompts_path}", flush=True)
    return {
        "stem": stem, "status": "ok", "model": model_name, "architecture": arch,
        "sources": [s["path"] for s in sources],
        "input": sum(s["count"] for s in sources),
        "string_dedup_removed": string_removed,
        "token_dedup_removed": token_removed,
        "selected": len(selected),
        "prompts_file": prompts_status,
        "prompts_on_disk": len(on_disk),
        "avg_prompt_tokens": avg_tokens,
        "avg_prompt_tokens_over": counted,
        "summary": summary,
        "output": str(prompts_path),
        "merged_output": str(merged_path) if merged_path else None,
    }


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------

def load_summary_json(path):
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {m["stem"]: m for m in data.get("models", [])}


def update_summaries(report, args):
    """Merge this run's per-stem records into prompt_set_summary.json and
    regenerate avg_prompt_tokens.json and probability_summary.txt from it."""
    by_stem = load_summary_json(args.summary_json)
    for r in report:
        if r["status"] != "ok":
            continue
        rec = by_stem.get(r["stem"], {})
        rec.update({
            "stem": r["stem"], "model": r["model"], "architecture": r["architecture"],
            "prompts": r["prompts_on_disk"],
            "logprob": r["summary"].get("logprob"),
            "probability": r["summary"].get("probability"),
        })
        if r["avg_prompt_tokens"] is not None:
            rec["avg_prompt_tokens"] = r["avg_prompt_tokens"]
            rec["avg_prompt_tokens_over"] = r["avg_prompt_tokens_over"]
        by_stem[r["stem"]] = rec
    if not by_stem:
        return
    records = [by_stem[s] for s in sorted(by_stem)]

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_json, "w") as f:
        json.dump({"models": records}, f, indent=2)

    with open(args.avg_tokens_json, "w") as f:
        json.dump({"models": [
            {"model": m["model"], "architecture": m.get("architecture"),
             "avg_prompt_tokens": m["avg_prompt_tokens"]}
            for m in records if m.get("avg_prompt_tokens") is not None
        ]}, f, indent=2)

    rows = []
    for m in records:
        prob = m.get("probability") or {}
        rows.append([m["stem"], str(m.get("prompts", "")),
                     "" if m.get("avg_prompt_tokens") is None else f"{m['avg_prompt_tokens']:.2f}",
                     _fmt_num(prob.get("lowest")), _fmt_num(prob.get("highest")),
                     *(_fmt_num(prob.get(k)) for k in PERCENTILE_KEYS)])
    args.probability_txt.write_text(format_table(TABLE_COLUMNS, rows), encoding="utf-8")
    print(f"Summaries: {args.summary_json}, {args.avg_tokens_json}, {args.probability_txt}")


def summarize_directory(directory, output=None):
    """Tabulate summary.probability of every result JSON in a folder."""
    directory = Path(directory)
    if not directory.is_dir():
        sys.exit(f"error: {directory} is not a directory")
    output = output or directory / f"{directory.name}_probability_summary.txt"
    suffix_re = re.compile(r"_\d+tokens(?:_.*)?$")

    rows = []
    for json_path in sorted(directory.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        summary = data.get("summary") if isinstance(data, dict) else None
        prob = (summary or {}).get("probability")
        if prob is None:
            print(f"skipping {json_path.name}: no probability summary", file=sys.stderr)
            continue
        rows.append([suffix_re.sub("", json_path.stem), str(summary.get("count", "")),
                     _fmt_num(prob["lowest"]), _fmt_num(prob["highest"]),
                     *(_fmt_num(prob[k]) for k in PERCENTILE_KEYS)])
    if not rows:
        sys.exit(f"error: no JSON files with probability summaries found in {directory}")

    columns = ["model", "prompts", "lowest", "highest", *PERCENTILE_KEYS]
    Path(output).write_text(format_table(columns, rows), encoding="utf-8")
    print(f"wrote {output} ({len(rows)} models)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--dirs", nargs="+", type=Path, default=DEFAULT_DIRS,
                        help="Directories of search result JSONs; only top-level *.json files are read "
                             f"(default: {' '.join(str(d) for d in DEFAULT_DIRS)}).")
    parser.add_argument("-o", "--prompts-dir", type=Path, default=PROMPTS_DIR,
                        help=f"Unified prompt directory to write <stem>.json into (default: {PROMPTS_DIR}).")
    parser.add_argument("--merged-dir", type=Path, default=None,
                        help="Also write the selected entries in full result format "
                             "(config, summary, results with num_tokens/token_ids) to this directory.")
    parser.add_argument("-n", "--top-n", type=int, default=DEFAULT_TOP_N,
                        help=f"Number of lowest-logprob prompts to keep per model; 0 keeps all (default: {DEFAULT_TOP_N}).")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Only process these model stems (default: every stem found).")
    parser.add_argument("--no-token-dedup", dest="token_dedup", action="store_false",
                        help="Skip tokenizer-based deduplication (string dedup always runs).")
    parser.add_argument("--no-token-count", dest="token_count", action="store_false",
                        help="Skip the average chat-templated prompt token count.")
    parser.add_argument("--count-prompts", type=int, default=DEFAULT_COUNT_PROMPTS,
                        help=f"Prompts (from the top of the set) averaged for the token count; 0 = all "
                             f"(default: {DEFAULT_COUNT_PROMPTS}).")
    parser.add_argument("--count-chunk", type=int, default=DEFAULT_COUNT_CHUNK,
                        help=f"Prompts rendered/tokenized per chunk when counting (default: {DEFAULT_COUNT_CHUNK}).")
    parser.add_argument("--hub", default=None,
                        help="HuggingFace hub cache holding the models--* folders (sets HF_HUB_CACHE; "
                             "default: leave the environment's HF_HOME / HF_HUB_CACHE alone).")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Pass trust_remote_code=True when loading tokenizers.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace prompt files that already exist (default: keep them; "
                             "counts and stats are refreshed either way).")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT,
                        help=f"Where to write the per-run report (default: {DEFAULT_REPORT}).")
    parser.add_argument("--summary-json", type=Path, default=SUMMARY_JSON,
                        help=f"Accumulated per-stem counts and stats (default: {SUMMARY_JSON}).")
    parser.add_argument("--avg-tokens-json", type=Path, default=AVG_TOKENS_JSON,
                        help=f"Average prompt tokens per model, regenerated from --summary-json "
                             f"(default: {AVG_TOKENS_JSON}).")
    parser.add_argument("--probability-txt", type=Path, default=PROBABILITY_TXT,
                        help=f"Probability percentile table, regenerated from --summary-json "
                             f"(default: {PROBABILITY_TXT}).")
    parser.add_argument("--summarize", type=Path, default=None, metavar="DIR",
                        help="Only tabulate summary.probability of every result JSON in DIR "
                             "to DIR/<DIR name>_probability_summary.txt (or --summarize-output) and exit.")
    parser.add_argument("--summarize-output", type=Path, default=None,
                        help="Output path for --summarize.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.summarize is not None:
        summarize_directory(args.summarize, args.summarize_output)
        return

    if args.hub:
        os.environ.setdefault("HF_HUB_CACHE", args.hub)
    model_configs = load_model_configs()

    paths_by_stem = collect_result_files(args.dirs)
    if args.models:
        wanted = set(args.models)
        missing = wanted - set(paths_by_stem)
        if missing:
            print(f"No result files for: {' '.join(sorted(missing))}")
        paths_by_stem = {s: p for s, p in paths_by_stem.items() if s in wanted}
    if not paths_by_stem:
        sys.exit(f"No result JSONs found in {' '.join(str(d) for d in args.dirs)}")

    report, failed = [], []
    for stem in sorted(paths_by_stem):
        print(f"{stem}", flush=True)
        try:
            row = process_stem(stem, paths_by_stem[stem], args, model_configs)
            report.append(row)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
            failed.append({"stem": stem, "error": f"{type(e).__name__}: {e}"})

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump({"dirs": [str(d) for d in args.dirs], "prompts_dir": str(args.prompts_dir),
                   "top_n": args.top_n, "token_dedup": args.token_dedup,
                   "token_count": args.token_count, "count_prompts": args.count_prompts,
                   "processed": report, "failed": failed}, f, indent=2)
    update_summaries(report, args)

    print("\nmodel_stem                   input  str_dup  tok_dup  selected  prompts_file   avg_tokens")
    for r in report:
        if r["status"] == "ok":
            avg = "" if r["avg_prompt_tokens"] is None else f"{r['avg_prompt_tokens']:.2f}"
            print(f"{r['stem']:<26} {r['input']:>8} {r['string_dedup_removed']:>8} "
                  f"{r['token_dedup_removed']:>8} {r['selected']:>9}  {r['prompts_file']:<13} {avg:>10}")
        else:
            print(f"{r['stem']:<26} {'':>8} {'':>8} {'':>8} {'':>9}  {r['status']}")
    if failed:
        print(f"\n{len(failed)} stem(s) failed:")
        for fl in failed:
            print(f"  {fl['stem']}: {fl['error']}")
    print(f"\nPrompt files in {args.prompts_dir}; report at {args.report}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
