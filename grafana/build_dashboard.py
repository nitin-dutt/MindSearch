# -*- coding: utf-8 -*-
"""Generates grafana/dashboards/mindsearch.json (the provisioned MindSearch dashboard).   Run:  python grafana/build_dashboard.py

The dashboard is generated rather than hand-edited so each panel is one readable line here (title, Prometheus query, unit) and the JSON
Grafana loads always matches. Metric names come from backend/metrics.py (mindsearch_*), the API's HTTP instrumentation (http_*) and vLLM (vllm:*).
"""
import json
import os

DS = {"type": "prometheus", "uid": "prometheus"}
_id = [0]
_y = [0]


def _next_id():
    _id[0] += 1
    return _id[0]


def target(expr, legend="", ref="A"):
    return {"datasource": DS, "expr": expr, "legendFormat": legend, "refId": ref}


def stat(title, expr, unit="short", x=0, w=6, h=4, desc="", thresholds=None):
    steps = thresholds or [{"color": "green", "value": None}]
    return {"id": _next_id(), "type": "stat", "title": title, "description": desc, "datasource": DS,
            "gridPos": {"x": x, "y": _y[0], "w": w, "h": h},
            "targets": [target(expr)],
            "fieldConfig": {"defaults": {"unit": unit, "thresholds": {"mode": "absolute", "steps": steps}}, "overrides": []},
            "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "value", "graphMode": "area"}}


def series(title, targets, unit="short", x=0, w=12, h=8, desc="", stack=False):
    return {"id": _next_id(), "type": "timeseries", "title": title, "description": desc, "datasource": DS,
            "gridPos": {"x": x, "y": _y[0], "w": w, "h": h},
            "targets": [target(e, l, chr(65 + i)) for i, (e, l) in enumerate(targets)],
            "fieldConfig": {"defaults": {"unit": unit, "custom": {"lineWidth": 2, "fillOpacity": 15 if stack else 0,
                                                                  "stacking": {"mode": "normal" if stack else "none"}}}, "overrides": []},
            "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}}}


def row(title):
    return {"id": _next_id(), "type": "row", "title": title, "collapsed": False, "gridPos": {"x": 0, "y": _y[0], "w": 24, "h": 1}, "panels": []}


def q(metric, p, by, window="1m", extra=""):
    return f"histogram_quantile({p}, sum by (le{', ' + by if by else ''}) (rate({metric}_bucket{extra}[{window}])))"


panels = []


def add(*items, height=8):
    panels.extend(items)
    _y[0] += height


add(row("Chat requests"), height=1)
add(stat("Chat requests / min", "sum(rate(mindsearch_chat_requests_total[2m])) * 60", "short", 0, desc="All chat requests, however they ended."),
    stat("Time to first token p95", q("mindsearch_time_to_first_token_seconds", 0.95, "", "5m"), "s", 6,
         desc="From the request arriving to the first word of the answer being ready.",
         thresholds=[{"color": "green", "value": None}, {"color": "orange", "value": 3}, {"color": "red", "value": 8}]),
    stat("Refused as 'I don't know' (share)", '(sum(rate(mindsearch_chat_requests_total{outcome=~"refused_gate|abstained_model"}[5m])) or vector(0)) / sum(rate(mindsearch_chat_requests_total[5m]))',
         "percentunit", 12, desc="Gate refusals plus the model saying it does not know."),
    stat("Failed or interrupted (share)", '(sum(rate(mindsearch_chat_requests_total{outcome=~"failed|interrupted"}[5m])) or vector(0)) / sum(rate(mindsearch_chat_requests_total[5m]))',
         "percentunit", 18, desc="The language model was unavailable, or died mid-answer.",
         thresholds=[{"color": "green", "value": None}, {"color": "red", "value": 0.01}]), height=4)
add(series("How chat requests end (per minute)", [("sum by (outcome) (rate(mindsearch_chat_requests_total[1m])) * 60", "{{outcome}}")], "short", 0, 12, stack=True,
           desc="answered, abstained_model, refused_gate, no_context, failed (HTTP 503), interrupted."),
    series("Time to first token (p50 / p95)", [(q("mindsearch_time_to_first_token_seconds", 0.5, ""), "p50"), (q("mindsearch_time_to_first_token_seconds", 0.95, ""), "p95")],
           "s", 12, 12))

add(row("Where a request spends its time"), height=1)
add(series("Stage latency p95", [(q("mindsearch_stage_seconds", 0.95, "stage"), "{{stage}}")], "s", 0, 12,
           desc="rewrite, embed_query, vector_search, decrypt, rerank, graph, retrieve_total (all retrieval), generation (the model)."),
    series("Stage latency p50", [(q("mindsearch_stage_seconds", 0.5, "stage"), "{{stage}}")], "s", 12, 12))
add(series("API latency p95 by route", [(q("http_request_duration_seconds", 0.95, "handler", extra='{handler!="/metrics"}'), "{{handler}}")], "s", 0, 12,
           desc="Standard HTTP timing of every API route."),
    series("API requests per second by status", [('sum by (status) (rate(http_requests_total{handler!="/metrics"}[1m]))', "{{status}}")], "reqps", 12, 12))

add(row("Language model (vLLM)"), height=1)
add(series("Requests running / waiting", [("vllm:num_requests_running", "running"), ("vllm:num_requests_waiting", "waiting")], "short", 0, 8, 7),
    series("KV-cache usage", [("vllm:kv_cache_usage_perc", "KV cache")], "percentunit", 8, 8, 7,
           desc="How full the model's working memory is; near 100% means requests queue."),
    series("Tokens per second", [("sum(rate(vllm:generation_tokens_total[1m]))", "generated"), ("sum(rate(vllm:prompt_tokens_total[1m]))", "prompt")], "short", 16, 8, 7), height=7)
add(series("vLLM time to first token p95", [(q("vllm:time_to_first_token_seconds", 0.95, ""), "p95")], "s", 0, 12, 7),
    series("vLLM inter-token latency p95", [(q("vllm:inter_token_latency_seconds", 0.95, ""), "p95")], "s", 12, 12, 7), height=7)

add(row("Ingestion (worker) and integrity"), height=1)
add(series("Documents ingested per minute", [('sum by (status) (rate(mindsearch_ingest_documents_total{job="worker"}[5m])) * 60', "{{status}}")], "short", 0, 8, 7,
           desc="done, failed (after retries), retried."),
    series("Ingest time per document p95", [(q("mindsearch_ingest_seconds", 0.95, "", "10m", '{job="worker"}'), "p95")], "s", 8, 8, 7),
    series("Chunks written per minute", [('sum(rate(mindsearch_ingest_chunks_total{job="worker"}[5m])) * 60', "chunks")], "short", 16, 8, 7), height=7)
add(stat("Chunks that failed integrity checks", "sum(mindsearch_decrypt_failures_total)", "short", 0, 6,
         desc="A stored chunk failed decryption and was skipped (wrong key or tampering). Should stay 0.",
         thresholds=[{"color": "green", "value": None}, {"color": "red", "value": 1}]),
    stat("Invalid citation numbers", "sum(mindsearch_invalid_citations_total)", "short", 6, 6, desc="Answers citing a passage that was not shown."),
    stat("Graph chunks that failed extraction", 'sum(mindsearch_kg_chunks_failed_total{job="worker"})', "short", 12, 6),
    stat("Worker up", 'min(up{job="worker"})', "short", 18, 6, desc="1 = Prometheus can scrape the ingestion worker.",
         thresholds=[{"color": "red", "value": None}, {"color": "green", "value": 1}]), height=4)

dashboard = {
    "uid": "mindsearch-overview", "title": "MindSearch overview", "tags": ["mindsearch"], "timezone": "browser", "schemaVersion": 39, "version": 2,
    "refresh": "10s", "time": {"from": "now-30m", "to": "now"}, "editable": True, "panels": panels,
    "templating": {"list": []}, "annotations": {"list": []},
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboards", "mindsearch.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8", newline="\n") as f:
    json.dump(dashboard, f, indent=1)
print(f"Wrote {out}: {sum(1 for p in panels if p['type'] != 'row')} panels")
