"""
Assignment 11 — Monitoring & Alerts starter (TODO).

Tracks block rate, rate-limit hits, judge fail rate.
Fires alerts when thresholds are exceeded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def default_metrics_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "metrics.json")


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MonitoringAlert:
    """Aggregate counters from pipeline plugins and emit alerts."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    alerts: list[Alert] = field(default_factory=list)

    # Counters — update these from your pipeline after each request
    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0

    # Breakdown theo lớp chặn (rate_limiter / input_guardrail / output_guardrail …)
    blocked_by_layer: dict = field(default_factory=dict)
    redacted_responses: int = 0

    def record(self, *, blocked: bool, layer: str | None = None, redacted: bool = False):
        """Update counters after each request."""
        self.total_requests += 1
        if blocked:
            self.blocked_requests += 1
            key = layer or "unknown"
            self.blocked_by_layer[key] = self.blocked_by_layer.get(key, 0) + 1
            if layer == "rate_limiter":
                self.rate_limit_hits += 1
        if redacted:
            self.redacted_responses += 1

    def check_metrics(self) -> list[Alert]:
        """Compute rates, append Alert objects when thresholds exceeded."""
        snap = self.snapshot()
        new_alerts: list[Alert] = []
        existing = {a.metric for a in self.alerts}

        if self.total_requests and snap["block_rate"] > self.block_rate_threshold:
            new_alerts.append(Alert(
                metric="block_rate",
                value=round(snap["block_rate"], 4),
                threshold=self.block_rate_threshold,
                message=(
                    f"Block rate {snap['block_rate']:.0%} > "
                    f"{self.block_rate_threshold:.0%} — possible attack wave or false positives."
                ),
            ))
        if self.rate_limit_hits >= self.rate_limit_hit_threshold:
            new_alerts.append(Alert(
                metric="rate_limit_hits",
                value=float(self.rate_limit_hits),
                threshold=float(self.rate_limit_hit_threshold),
                message=(
                    f"{self.rate_limit_hits} rate-limit hits ≥ "
                    f"{self.rate_limit_hit_threshold} — possible flooding / cost attack."
                ),
            ))
        if self.judge_checks and snap["judge_fail_rate"] > self.judge_fail_rate_threshold:
            new_alerts.append(Alert(
                metric="judge_fail_rate",
                value=round(snap["judge_fail_rate"], 4),
                threshold=self.judge_fail_rate_threshold,
                message="LLM judge fail rate above threshold — review model outputs.",
            ))

        # Không nhân đôi cảnh báo cùng metric
        self.alerts = [a for a in self.alerts if a.metric not in {n.metric for n in new_alerts}]
        self.alerts.extend(new_alerts)
        for a in new_alerts:
            if a.metric not in existing:
                print(f"[ALERT] {a.message}")
        return new_alerts

    def export_json(self, filepath: str | None = None):
        """Write metrics + alerts to JSON under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_metrics_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.snapshot()
        data["blocked_by_layer"] = dict(self.blocked_by_layer)
        data["redacted_responses"] = self.redacted_responses
        data["thresholds"] = {
            "block_rate": self.block_rate_threshold,
            "rate_limit_hits": self.rate_limit_hit_threshold,
            "judge_fail_rate": self.judge_fail_rate_threshold,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }
