from typing import Any, Dict, List


def _merge_signal_metadata(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        if key.startswith(("vector_", "fts_keywords_", "fts_summaries_")):
            if isinstance(value, bool):
                target[key] = bool(target.get(key)) or value
            elif isinstance(value, (int, float)):
                target[key] = max(float(target.get(key, 0.0)), float(value))
            elif key not in target or not target.get(key):
                target[key] = value


def merge_vector_results(per_shard: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for shard_results in per_shard:
        for row in shard_results:
            kbid = str(row.get("KBID", "")).strip()
            if not kbid:
                continue
            dist = float(row.get("vector_distance", 1e9))
            if kbid not in best or dist < float(best[kbid].get("vector_distance", 1e9)):
                best[kbid] = row
    return sorted(best.values(), key=lambda x: x.get("vector_distance", 1e9))


def merge_fts_keyword_results(per_shard: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for shard_results in per_shard:
        for row in shard_results:
            kbid = str(row.get("KBID", "")).strip()
            if not kbid:
                continue
            rank = float(row.get("fts_keywords_rank", 0.0))
            if kbid not in best or rank > float(best[kbid].get("fts_keywords_rank", 0.0)):
                best[kbid] = row
    return sorted(best.values(), key=lambda x: x.get("fts_keywords_rank", 0.0), reverse=True)


def merge_fts_summary_results(per_shard: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for shard_results in per_shard:
        for row in shard_results:
            kbid = str(row.get("KBID", "")).strip()
            if not kbid:
                continue
            rank = float(row.get("fts_summaries_rank", 0.0))
            if kbid not in best or rank > float(best[kbid].get("fts_summaries_rank", 0.0)):
                best[kbid] = row
    return sorted(best.values(), key=lambda x: x.get("fts_summaries_rank", 0.0), reverse=True)


def weighted_rrf(
    *,
    vector_results: List[Dict[str, Any]],
    fts_kw_results: List[Dict[str, Any]],
    fts_sm_results: List[Dict[str, Any]],
    vector_weight: float,
    fts_kw_weight: float,
    fts_sm_weight: float,
    match_bonus: float,
    ranked_lists: List[List[Dict[str, Any]]] = None,
    weights: List[float] = None,
) -> List[Dict[str, Any]]:
    if ranked_lists is not None:
        resolved_weights = weights or [vector_weight, fts_kw_weight, fts_sm_weight]
        if len(ranked_lists) >= 1:
            vector_results = ranked_lists[0]
            vector_weight = resolved_weights[0] if len(resolved_weights) > 0 else vector_weight
        if len(ranked_lists) >= 2:
            fts_kw_results = ranked_lists[1]
            fts_kw_weight = resolved_weights[1] if len(resolved_weights) > 1 else fts_kw_weight
        if len(ranked_lists) >= 3:
            fts_sm_results = ranked_lists[2]
            fts_sm_weight = resolved_weights[2] if len(resolved_weights) > 2 else fts_sm_weight

    merged: Dict[str, Dict[str, Any]] = {}
    vec_scores: Dict[str, float] = {}
    fts_kw_scores: Dict[str, float] = {}
    fts_sm_scores: Dict[str, float] = {}

    for row in vector_results or []:
        kbid = str(row.get("KBID", "")).strip()
        if not kbid:
            continue
        score = float(row.get("vector_distance", 1.0))
        if kbid not in vec_scores or score < vec_scores[kbid]:
            vec_scores[kbid] = score
        if kbid not in merged:
            merged[kbid] = dict(row)
        else:
            _merge_signal_metadata(merged[kbid], row)

    for row in fts_kw_results or []:
        kbid = str(row.get("KBID", "")).strip()
        if not kbid:
            continue
        score = float(row.get("fts_keywords_rank", 0.0))
        if kbid not in fts_kw_scores or score > fts_kw_scores[kbid]:
            fts_kw_scores[kbid] = score
        if kbid not in merged:
            merged[kbid] = dict(row)
        else:
            _merge_signal_metadata(merged[kbid], row)

    for row in fts_sm_results or []:
        kbid = str(row.get("KBID", "")).strip()
        if not kbid:
            continue
        score = float(row.get("fts_summaries_rank", 0.0))
        if kbid not in fts_sm_scores or score > fts_sm_scores[kbid]:
            fts_sm_scores[kbid] = score
        if kbid not in merged:
            merged[kbid] = dict(row)
        else:
            _merge_signal_metadata(merged[kbid], row)

    if not merged:
        return []

    def norm_lower_better(scores_map):
        vals = list(scores_map.values())
        if not vals:
            return {}
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return {k: 1.0 for k in scores_map}
        return {k: 1.0 - (v - lo) / (hi - lo) for k, v in scores_map.items()}

    def norm_higher_better(scores_map):
        vals = list(scores_map.values())
        if not vals:
            return {}
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return {k: 1.0 for k in scores_map}
        return {k: (v - lo) / (hi - lo) for k, v in scores_map.items()}

    norm_vec = norm_lower_better(vec_scores)
    norm_fts_kw = norm_higher_better(fts_kw_scores)
    norm_fts_sm = norm_higher_better(fts_sm_scores)

    for kbid, base in merged.items():
        n_signals = (
            (1 if kbid in vec_scores else 0)
            + (1 if kbid in fts_kw_scores else 0)
            + (1 if kbid in fts_sm_scores else 0)
        )
        bonus = match_bonus * (n_signals - 1)
        exact_bonus = 0.0
        phrase_bonus = 0.0
        if base.get("vector_exact_match") or base.get("fts_keywords_exact_match") or base.get("fts_summaries_exact_match"):
            exact_bonus += 0.20
        if base.get("vector_phrase_match") or base.get("fts_keywords_phrase_match") or base.get("fts_summaries_phrase_match"):
            phrase_bonus += 0.08
        base["rrf_score"] = (
            vector_weight * norm_vec.get(kbid, 0.0)
            + fts_kw_weight * norm_fts_kw.get(kbid, 0.0)
            + fts_sm_weight * norm_fts_sm.get(kbid, 0.0)
            + bonus
            + exact_bonus
            + phrase_bonus
        )

    return sorted(merged.values(), key=lambda x: x.get("rrf_score", 0.0), reverse=True)
