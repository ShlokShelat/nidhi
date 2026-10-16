"""
evidence_vote_test.py

Build the TEST evidence column by voting across the evidence tagger runs: a
span is emitted only when at least --k of the runs propose it.

WHY AGREEMENT RATHER THAN UNION. The scorer matches predicted spans to gold
spans one-to-one, so an extra span that matches no gold span is a full false
positive and cannot be offset by the spans that do match. Pooling several
taggers by union therefore imports each tagger's private errors on top of the
correct spans they share. Requiring agreement discards proportionally more bad
spans than good ones -- a wrong span tends to be wrong in a way peculiar to one
tagger, while a right one is found by several -- so the vote trades a little
recall for more precision. Raising --k further keeps only the spans every
tagger finds, which costs more recall than the remaining precision is worth.

Each run contributes spans at ITS OWN evidence_min_conf, read from that run's
run_config.json, so the test build uses the same per-run operating points the
vote was validated under rather than re-tuning anything.

Usage:
  python src/evidence_vote_test.py --k 2
  python src/patch_submission_risk.py \
      --base submissions/NIDHI_evidence_consistent.csv \
      --evidence artifacts/evidence_vote2_test.csv \
      --out submissions/NIDHI_evidence_vote2b.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scoring import (_span_pair_matches, normalize_label, normalize_span,
                     parse_evidence, token_len)

RUNS = ["evidence_v2", "evidence_v1", "evidence_s14", "evidence_s15"]


DEDUPE_KEEP = "long"     # set from --dedupe-keep; "long" keeps the longer
                         # span of a containment pair


def dedupe(spans):
    """Drop duplicates and spans contained in a longer kept span. Matching is
    one-to-one, so near-duplicates can never both score -- keeping them only
    costs precision."""
    # Tie-break on CHARACTER length, not just tokens. Two spans can carry the
    # same token count while one is a character-level prefix of the other (a
    # clipped first word), so with a token-only key their order is arbitrary;
    # when the shorter-by-characters one is visited first, the longer is not
    # contained in it and BOTH survive. Matching is one-to-one, so the extra
    # span is a guaranteed false positive.
    # DEDUPE_KEEP = "short" reverses the visit order so the SHORTEST survives a
    # containment pair instead of the longest. It matters because it runs
    # BEFORE clustering, so it decides which string each run contributes to
    # the vote, which is then what `--rep modal` picks among.
    # A shorter survivor is also safer under the scorer's 3x token cap: a
    # prediction shorter than gold is contained by it, while a longer one can
    # breach the cap and match nothing.
    rev = DEDUPE_KEEP == "short"
    key = ((lambda x: (token_len(x), len(normalize_span(x)))) if rev
           else (lambda x: (-token_len(x), -len(normalize_span(x)))))
    out = []
    for s in sorted(spans, key=key):
        n = normalize_span(s)
        if not n:
            continue
        # keep-longest: drop s if an already-kept span CONTAINS it.
        # keep-shortest: drop s if it CONTAINS an already-kept span.
        if rev:
            if any(normalize_span(k) in n for k in out):
                continue
        elif any(n in normalize_span(k) for k in out):
            continue
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(RUNS))
    ap.add_argument("--dir", default="artifacts/evidence_runs")
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--split", default="test", choices=["test", "val"],
                    help="which candidate dump to vote over: reads "
                         "{run}_{split}_candidates.csv. Use `val` to SCORE a "
                         "vote configuration against gold before shipping it; "
                         "`test` to build a submission.")
    ap.add_argument("--risk-from",
                    default="submissions/NIDHI_evidence_consistent.csv",
                    help="submission whose risk column drives the consistency "
                         "rule; must be the same submission this evidence "
                         "column will be patched into")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--cap", type=int, default=None,
                    help="max spans per post, ranked by cluster agreement. "
                         "Unset leaves the span set unchanged. NOTE the "
                         "exact-vote instruments (evidence_member_loo, "
                         "apply_span_reranker) score a DIFFERENT decode from "
                         "this one, because they do not cluster, so a cap must "
                         "be validated HERE with --split val, not there.")
    ap.add_argument("--suppress-prob", type=float, default=None,
                    help="suppress evidence where prob_indicator >= this "
                         "instead of where the risk ARGMAX is indicator. The "
                         "argmax rule fires wherever p_indicator happens to "
                         "beat three other classes, which is not a chosen "
                         "operating point; this makes the decision boundary "
                         "explicit and tunable. None = argmax rule.")
    ap.add_argument("--merge-overlap", type=float, default=None,
                    help="drop a span that OVERLAPS a higher-agreement span "
                         "on the same post without either containing the "
                         "other (clustering only handles containment). Value "
                         "is the required shared-token fraction of the "
                         "shorter span, e.g. 0.4. None = no overlap merge.")
    ap.add_argument("--dedupe-keep", default="long", choices=["long", "short"],
                    help="which span survives a containment pair WITHIN a run, "
                         "before clustering. 'long' (default) keeps the "
                         "containing span, 'short' the contained one.")
    ap.add_argument("--rel-agree", type=float, default=None,
                    help="per-post relative agreement floor in (0,1]: keep a "
                         "span only if agree(s) >= alpha * max_agree(post). "
                         "Unlike --k this adapts to how much the taggers "
                         "agreed on THIS post. Composes with --cap, applied "
                         "first. None = no relative floor.")
    ap.add_argument("--min-conf-override", type=float, default=None,
                    help="state every run's evidence_min_conf explicitly when "
                         "runs/*/run_config.json is unavailable (e.g. scoring "
                         "from dumps on a machine without the checkpoints). "
                         "Without this a missing config REFUSES rather than "
                         "silently defaulting the gate to 0.0.")
    ap.add_argument("--allow-missing", action="store_true",
                    help="build even if some requested runs lack test "
                         "candidates (default: refuse, since a silently "
                         "smaller ensemble is not what was validated)")
    ap.add_argument("--out", default="artifacts/evidence_vote2_test.csv")
    ap.add_argument("--rep", default="modal",
                    choices=["modal", "capsafe", "median", "short"],
                    help="which variant of a cluster to emit. `modal` "
                         "(default) emits the boundary the most runs agreed "
                         "on. `capsafe` falls back to the shortest variant "
                         "when the modal one exceeds 3x it, respecting the "
                         "scorer's 3x token cap.")
    ap.add_argument("--cluster", action="store_true",
                    help="Count agreement under the SCORER's containment+3x "
                         "equivalence instead of exact string equality, so "
                         "runs that pick the same clause but different "
                         "boundaries count as agreeing rather than splitting "
                         "their votes and losing both spans.")
    args = ap.parse_args()
    global DEDUPE_KEEP
    DEDUPE_KEEP = args.dedupe_keep

    te = pd.read_csv(args.test)
    rows = list(te["row_id"])

    ch = pd.read_csv(args.risk_from, keep_default_na=False)
    # OOF prediction files name the column pred_risk_level; submissions name it
    # risk_level. Both are valid risk sources for the consistency rule.
    rcol = "risk_level" if "risk_level" in ch.columns else "pred_risk_level"
    risk = {r: normalize_label(x)
            for r, x in zip(ch["row_id"], ch[rcol])}
    missing = [r for r in rows if r not in risk]
    if missing:
        raise SystemExit(f"no risk label for {len(missing)} {args.split} rows")

    # PROBABILITY-BASED SUPPRESSION. The default rule suppresses iff the risk
    # ARGMAX is `indicator` -- i.e. at whatever point p_indicator happens to
    # beat three other classes. That is not a chosen threshold.
    #
    # Suppression is all-or-nothing under the metric, so it dominates span
    # quality: a post whose gold evidence is empty scores 1.0 when nothing is
    # emitted and 0.0 otherwise, and a post whose gold is non-empty scores 0.0
    # if suppressed. Each suppression error costs a whole post, where a
    # mis-bounded span costs only part of one. --suppress-prob exposes that
    # decision as an explicit threshold on the calibrated p_indicator instead
    # of leaving it wherever the argmax boundary falls.
    #
    # Any threshold set here must be tuned on an instrument that suppresses on
    # PREDICTED risk, as this script does; tuning it against gold risk measures
    # a decision the deployed pipeline never makes.
    supp_p = None
    if args.suppress_prob is not None:
        pcol = next((c for c in ("prob_indicator", "p_Indicator", "p_indicator")
                     if c in ch.columns), None)
        if pcol is None:
            raise SystemExit(
                f"--suppress-prob needs a prob_indicator column in "
                f"{args.risk_from}; found {list(ch.columns)}")
        supp_p = {str(r): float(v) for r, v in zip(ch["row_id"], ch[pcol])}
        miss2 = [r for r in rows if str(r) not in supp_p]
        if miss2:
            raise SystemExit(f"no prob_indicator for {len(miss2)} rows")

    per_run = {}
    for run in [r.strip() for r in args.runs.split(",")]:
        cp = os.path.join(args.dir, f"{run}_{args.split}_candidates.csv")
        cfgp = os.path.join("runs", run, "run_config.json")
        if not os.path.exists(cp):
            print(f"  skip {run}: no {cp}")
            continue
        # A MISSING run_config.json MUST NOT DEFAULT min_conf TO 0.0.
        # The per-run gates are non-trivial (0.70, 0.65, 0.60, 0.55, 0.0), so
        # a 0.0 fallback admits every tagger's full low-confidence tail and
        # emits a materially over-emitting evidence column -- a different model
        # from the validated one, with no error raised and only a printed
        # "min_conf=0.0" as the tell. Refuse instead.
        if args.min_conf_override is not None:
            mc = float(args.min_conf_override)
        elif not os.path.exists(cfgp):
            raise SystemExit(
                f"REFUSING to vote: {cfgp} is missing, so {run}'s "
                f"evidence_min_conf is unknown. Defaulting it to 0.0 would "
                f"silently admit that tagger's entire low-confidence tail and "
                f"change the evidence column. Restore runs/{run}/ or pass "
                f"--min-conf-override to state the gate explicitly.")
        else:
            mc = float(json.load(open(cfgp)).get("evidence_min_conf", 0.0))
        d = pd.read_csv(cp)
        d = d[d["conf"] >= mc]
        m = {}
        for r, s in zip(d["row_id"], d["span"]):
            m.setdefault(r, []).append(str(s))
        per_run[run] = {r: dedupe(m.get(r, [])) for r in rows}
        n = sum(len(v) for v in per_run[run].values())
        print(f"  {run}: min_conf={mc}  {n} spans ({n / len(rows):.2f}/post)")

    if len(per_run) < args.k:
        raise SystemExit(f"only {len(per_run)} runs loaded, need >= {args.k}")

    # Loudly refuse to silently build a smaller ensemble than asked for. A
    # missing candidate file would otherwise turn an 8-model vote into a
    # 4-model one with nothing but a 'skip' line as the tell, and a partial
    # ensemble is a different model from the validated one.
    asked = [r.strip() for r in args.runs.split(",")]
    lost = [r for r in asked if r not in per_run]
    if lost and not args.allow_missing:
        raise SystemExit(
            f"{len(lost)} of {len(asked)} requested runs have no test "
            f"candidates: {lost}\nRun scripts/evidence_dump.sh first, or pass "
            f"--allow-missing to build with {len(per_run)} runs anyway.")

    def cluster_kept(r, k):
        """Count agreement under the SCORER'S equivalence, not string equality.

        The exact-key vote below requires two runs to emit byte-identical text
        before they count as agreeing. The official scorer
        (`_span_pair_matches`) treats spans as equivalent when EITHER CONTAINS
        the other within a 3x token-length cap. So runs that agree on WHICH
        clause matters but differ on where it STARTS OR ENDS -- one emitting a
        core phrase, another the same phrase plus a trailing qualifier -- split
        their votes, neither reaches k, and BOTH are dropped, even though
        either one would have matched gold.

        Containment is substring-based, so it covers EDGE extensions only: a
        word inserted in the MIDDLE breaks containment in both directions, and
        those variants still split their votes -- correctly, since the scorer
        also treats them as different spans.

        Containment is not transitive, so grouping is order-dependent;
        candidates are sorted longest-first then lexicographically so the
        result does not depend on dict iteration order.
        """
        items = [(run, s) for run in per_run for s in per_run[run][r]]
        items.sort(key=lambda x: (-token_len(x[1]), normalize_span(x[1])))
        groups = []
        for run, s in items:
            for g in groups:
                if any(_span_pair_matches(s, s2) or _span_pair_matches(s2, s)
                       for _, s2 in g):
                    g.append((run, s))
                    break
            else:
                groups.append([(run, s)])
        out = []
        for g in groups:
            agree = len({run for run, _ in g})
            if agree < k:
                continue
            cnt = {}
            for _, s in g:
                cnt[s] = cnt.get(s, 0) + 1
            modal = max(cnt.items(),
                        key=lambda kv: (kv[1], -token_len(kv[0])))[0]
            if args.rep == "capsafe":
                # The scorer rejects a prediction longer than 3x the gold it
                # contains, so an over-long span matches nothing even when it
                # contains the right clause. A prediction CONTAINED IN gold
                # always passes the cap, so erring short is one-sided-safe: it
                # can only be too specific, never disqualified. Fall back to
                # the shortest variant only when the modal one exceeds 3x it:
                # the 3 is the SCORER'S OWN constant, so this stays
                # 0-parameter.
                shortest = min((s for _, s in g), key=token_len)
                out.append((agree, shortest
                            if token_len(modal) > 3 * max(token_len(shortest), 1)
                            else modal))
            elif args.rep == "median":
                ss = sorted((s for _, s in g), key=token_len)
                out.append((agree, ss[len(ss) // 2]))
            elif args.rep == "short":
                out.append((agree, min((s for _, s in g), key=token_len)))
            else:
                out.append((agree, modal))
        # (agreement, span) pairs. The agreement count is carried out so --cap
        # can drop the WEAKEST span on a post; the span ORDER is unchanged, so
        # dedupe() downstream sees exactly what it saw before.
        return out

    out, n_forced, n_suppressed = [], 0, 0
    for r in rows:
        if args.cluster:
            pairs = cluster_kept(r, args.k)
            agree_of = {}
            for a, s in pairs:
                agree_of[s] = max(agree_of.get(s, 0), a)
            kept = dedupe([s for _, s in pairs])
        else:
            counts = {}
            for run in per_run:
                for s in per_run[run][r]:
                    key = normalize_span(s)
                    counts.setdefault(key, [0, s])
                    counts[key][0] += 1
            agree_of = {s: c for c, s in counts.values()}
            kept = dedupe([s for c, s in counts.values() if c >= args.k])

        # PER-POST RELATIVE AGREEMENT. The metric is averaged over posts, and
        # how much the taggers agree varies from post to post, so a selection
        # rule that adapts per post can fit both an easy post and a hard one;
        # `k` cannot, being a single global agreement floor. This is the
        # continuous, per-post version of `--cap`: keep a span only if its
        # agreement is within `alpha` of the BEST agreement on its own post.
        #
        # Why it can beat a count cap: a post whose spans all agree 6/7 keeps
        # them all, while a post with one 7/7 span and two 2/7 stragglers loses
        # the stragglers -- a cap cannot tell those two posts apart, because it
        # only counts. The pool emits several near-duplicate candidates per
        # gold span and one-to-one bipartite matching charges the redundant
        # ones as FP, so the stragglers are exactly what is worth dropping.
        #
        # Applied BEFORE the cap so the two compose: rel-agree prunes on
        # quality, cap then bounds what survives.
        if args.rel_agree:
            if kept:
                best = max((agree_of.get(s, 0) for s in kept), default=0)
                if best > 0:
                    lim = args.rel_agree * best
                    kept = [s for s in kept if agree_of.get(s, 0) >= lim]

        if args.cap:
            kept = sorted(kept, key=lambda s: -agree_of.get(s, 0))[:args.cap]

        # OVERLAP MERGE. Clustering groups spans where one CONTAINS the other,
        # because that is the scorer's equivalence. It cannot see two spans
        # that OVERLAP without either containing the other -- two crops of the
        # same clause taken from different word boundaries -- which land in
        # separate clusters, both survive the vote, and both match the SAME
        # gold span. Matching is one-to-one, so the second is a guaranteed
        # false positive: on a post with one gold and two predictions where one
        # matches, F1 is 2*1/(2+1)=0.667 instead of 1.0. Overlapping crops of
        # one clause are exactly the redundancy a pool of taggers produces.
        # Drops the LOWER-agreement span of each pair.
        if args.merge_overlap:
            order = sorted(kept, key=lambda s: (-agree_of.get(s, 0),
                                                -token_len(s)))
            out2 = []
            for s in order:
                ns = set(normalize_span(s).split())
                if not ns:
                    continue
                red = False
                for k in out2:
                    nk = set(normalize_span(k).split())
                    ov = len(ns & nk)
                    if ov >= 2 and ov >= args.merge_overlap * min(len(ns),
                                                                  len(nk)):
                        red = True
                        break
                if not red:
                    out2.append(s)
            kept = out2

        # the consistency rule: a post whose risk level is `indicator` carries
        # no evidence by definition, so any spans found for it are suppressed.
        # --suppress-prob replaces the argmax test with an explicit threshold
        # on p_indicator.
        _sup = (supp_p[str(r)] >= args.suppress_prob) if supp_p is not None \
            else (risk[r] == "indicator")
        if _sup:
            if kept:
                n_suppressed += 1
            kept = []
        elif not kept:
            pool = [s for run in per_run for s in per_run[run][r]]
            if pool:
                kept = [min(pool, key=token_len)]
                n_forced += 1
        out.append("; ".join(kept))

    res = pd.DataFrame({"row_id": rows, "evidence": out})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    res.to_csv(args.out, index=False)

    n = sum(len(parse_evidence(x)) for x in out)
    # A risk source without an evidence column (an OOF prediction file, as
    # opposed to a submission) supports the vote but not the diff against the
    # base submission.
    if "evidence" in ch.columns:
        cur = {r: parse_evidence(x)
               for r, x in zip(ch["row_id"], ch["evidence"])}
        n_cur = sum(len(v) for v in cur.values())
        changed = sum(1 for r, x in zip(rows, out)
                      if [normalize_span(s) for s in parse_evidence(x)]
                      != [normalize_span(s) for s in cur[r]])
    else:
        n_cur, changed = None, None

    print(f"\nvote>={args.k} over {len(per_run)} runs ({args.split})")
    if n_cur is None:
        print(f"  spans: {n} ({n / len(rows):.2f}/post)")
    else:
        print(f"  spans: {n} ({n / len(rows):.2f}/post)   "
              f"champion has {n_cur} ({n_cur / len(rows):.2f}/post)")
    print(f"  suppressed on indicator posts: {n_suppressed}")
    print(f"  forced a span where vote gave none: {n_forced}")
    if changed is not None:
        print(f"  evidence differs from champion on {changed} of "
              f"{len(rows)} posts")
    print(f"\nwrote {args.out}")
    print("\nNext:")
    print("  python src/patch_submission_risk.py \\")
    print(f"      --base {args.risk_from} \\")
    print(f"      --evidence {args.out} \\")
    print("      --out submissions/NIDHI_evidence_vote2b.csv")


if __name__ == "__main__":
    main()
