#!/usr/bin/env python3
"""Semi-real GUI testbed for the forget gate under scroll misalignment.

The synthetic forget-gate fixtures were pixel-aligned; real GUIs scroll.
This probe renders 5 realistic HTML templates with headless Chrome
(classic --headless; Chrome 91 has no --headless=new), produces 120
old/new screenshot pairs in a factorial design

    {content changed, unchanged} x {scroll offset 0, small 30-80px,
                                    large 200-600px}   (20 pairs/cell)

and measures whether cheap vertical alignment (row-profile normalized
cross-correlation) + the reference tile-diff detector still separates
changed from unchanged, versus the naive no-alignment detector.

Every page has a position:fixed 64px header carrying a clock, a
"last sync" line and a snapshot id, all of which differ in EVERY pair,
so unchanged pairs still differ in pixels. Scroll is simulated by a
negative top margin on the content wrapper (verified to shift pixels);
the true offset is recorded. For changed pairs exactly one labeled field
value is mutated; its ground-truth bbox in the NEW screenshot comes from
re-rendering the same page with the mutated span painted magenta and
diffing against the plain render.

Detector variants:
    naive_unmasked    max tile diff, raw images
    naive_masked      same, fixed header rows (y<64) excluded
    aligned_unmasked  row-profile NCC offset estimate on full image,
                      tile diff on the aligned overlap
    aligned_masked    header cropped first, then align + diff

Reuses tile_diff / in_box / auroc from gate_forget_detector.py.

Outputs:
    results/discovery/gates/realgui_forget_gate.json   (metrics)
    results/discovery/gates/realgui/                   (pngs, manifest,
                                                        per-pair rows,
                                                        bbox verify crops)
CPU only; no VLM.
"""

from __future__ import annotations

import importlib.util
import json
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "results" / "discovery" / "gates" / "realgui"
HTML_DIR = OUTDIR / "html"
PNG_DIR = OUTDIR / "png"
VERIFY_DIR = OUTDIR / "verify"
MANIFEST = OUTDIR / "pairs_manifest.json"
PER_PAIR = OUTDIR / "per_pair_results.json"
RESULT_JSON = ROOT / "results" / "discovery" / "gates" / "realgui_forget_gate.json"

W, H = 1280, 1600
HEADER_H = 64
MAX_SCROLL = 1400          # content ~3300px, viewport 1600
SEED = 20260820
PAIRS_PER_CELL = 20
CELLS = [("zero", 0, 0), ("small", 30, 80), ("large", 200, 600)]
MAX_SHIFT = 700
MIN_OVERLAP = 300

# ---- reference detector helpers (tile_diff, in_box, auroc) -----------------
_spec = importlib.util.spec_from_file_location(
    "gate_forget_detector",
    ROOT / "vlm_diagnosis" / "scripts" / "gate_forget_detector.py")
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)
tile_diff, in_box, auroc, TILE = _ref.tile_diff, _ref.in_box, _ref.auroc, _ref.TILE


# ============================================================================
# 1. Templates
# ============================================================================

CSS = """
*{box-sizing:border-box}
body{margin:0;font-family:Arial,Helvetica,sans-serif;background:#f4f5f7;color:#20242c}
.hdr{position:fixed;top:0;left:0;right:0;height:64px;background:#1d2733;color:#e8edf3;
     z-index:100;padding:8px 28px;border-bottom:3px solid #3a86c8}
.hdr-top{display:flex;justify-content:space-between;font-size:17px;font-weight:bold}
.hdr-sub{font-size:12px;color:#9db0c4;margin-top:5px}
.wrap{max-width:1080px;margin:0 auto;padding:14px 24px 60px}
h1{font-size:26px;margin:18px 0 6px}
h2{font-size:19px;margin:26px 0 4px;color:#2b3a55}
p{line-height:1.55;color:#3c4452;margin:10px 0}
.card{background:#fff;border:1px solid #dde2e9;border-radius:8px;padding:14px 20px;margin:16px 0}
.card h3{margin:2px 0 10px;font-size:16px;color:#1d2733}
.row{display:flex;padding:9px 6px;border-bottom:1px solid #eef1f5;font-size:14px}
.row:last-child{border-bottom:none}
.lbl{color:#5a6472;width:280px;flex:none}
.mail{display:flex;padding:10px 8px;border-bottom:1px solid #e8ecf1;font-size:13px;background:#fff}
.mail .frm{width:170px;flex:none;font-weight:bold;color:#2b3a55}
.mail .sub{width:300px;flex:none}
.mail .snp{color:#7b8494;flex:1;overflow:hidden;white-space:nowrap}
.mail .tim{width:70px;flex:none;text-align:right;color:#9aa3b2}
table{border-collapse:collapse;width:100%;font-size:13px;background:#fff}
th{background:#e9eef5;text-align:left;padding:8px 10px;border:1px solid #d5dbe4}
td{padding:8px 10px;border:1px solid #e2e7ee}
.stat{background:#eef4fb;border-left:4px solid #3a86c8;padding:12px 18px;margin:16px 0;font-size:15px}
.desc{font-size:12px;color:#8a93a2;margin:2px 0 0}
.feed{font-size:13px;padding:7px 4px;border-bottom:1px dashed #e4e8ee;color:#495261}
"""

LOREM = [
    "The migration to the shared control plane finished ahead of the review "
    "cycle, and most residual traffic now flows through the regional relays "
    "without manual pinning. Remaining exceptions are tracked in the ledger.",
    "Operators reported that the reconciliation loop occasionally lagged "
    "behind bursty enqueue periods; the queue depth alarm was retuned and the "
    "worker pool now scales on the p95 dequeue latency rather than raw depth.",
    "Quarterly access reviews closed with three findings, all low severity. "
    "Two related to stale service tokens and one to an over-broad read scope "
    "on the archival bucket, revoked the same week.",
    "The caching layer was switched to a two-tier arrangement where warm "
    "entries live in process memory and colder entries spill to the shared "
    "store. Hit rates recovered to their pre-incident baseline within days.",
    "Documentation for the rollout playbook was consolidated into a single "
    "page with per-region checklists. Teams no longer need to cross-reference "
    "the legacy wiki, which will be archived at the end of the quarter.",
    "Budget variance stayed inside the agreed corridor for the fourth "
    "consecutive month. The main driver was lower egress spend after the "
    "compression change, partially offset by the new observability tier.",
    "A tabletop exercise rehearsed the loss of the primary metadata store. "
    "Failover completed inside the objective, though the runbook needed two "
    "corrections around credential rotation ordering.",
    "Hiring for the platform rotation closed with two accepted offers. "
    "Onboarding cohorts start on alternating weeks so that mentors are never "
    "assigned more than one new engineer at a time.",
]


def V(fid, val, debug_fid):
    st = "display:inline-block;min-width:110px;white-space:nowrap;font-weight:bold;"
    if fid == debug_fid:
        st += "background:#ff00ff;color:#ff00ff;outline:4px solid #ff00ff;"
    return f'<span id="f_{fid}" style="{st}">{val}</span>'


def frow(label, fid, vals, dbg):
    return (f'<div class="row"><span class="lbl">{label}</span>'
            f'{V(fid, vals[fid], dbg)}</div>')


def mailrow(frm, sub, snp, tim):
    return (f'<div class="mail"><span class="frm">{frm}</span>'
            f'<span class="sub">{sub}</span><span class="snp">{snp}</span>'
            f'<span class="tim">{tim}</span></div>')


# --- template 1: project dashboard ---
T1_FIELDS = [
    ("release_ver", "Release version",
     ["v4.2.19", "v4.3.02", "v4.1.87", "v4.4.10", "v4.2.55"]),
    ("deploy_env", "Environment",
     ["production", "staging-eu", "staging-us", "canary-01", "preview-2"]),
    ("build_status", "Build status",
     ["passing", "failing", "queued", "blocked", "expired"]),
    ("budget_used", "Error budget used",
     ["12.4%", "57.1%", "33.8%", "74.2%", "91.3%"]),
    ("cpu_quota", "CPU quota",
     ["640 vCPU", "512 vCPU", "768 vCPU", "896 vCPU", "384 vCPU"]),
    ("mem_quota", "Memory quota",
     ["2.0 TiB", "1.5 TiB", "3.2 TiB", "2.8 TiB", "4.1 TiB"]),
    ("storage_used", "Storage in use",
     ["48.2 TB", "61.7 TB", "39.4 TB", "72.9 TB", "55.1 TB"]),
    ("kpi_users", "Active users (24h)",
     ["18,204", "22,917", "15,486", "27,338", "19,062"]),
    ("oncall_name", "On-call engineer",
     ["Dana K.", "Femi A.", "Lars B.", "Mira T.", "Joon P."]),
    ("tickets_open", "Open tickets",
     ["23", "41", "17", "58", "36"]),
    ("region", "Primary region",
     ["eu-west-1", "us-east-2", "ap-neo-1a", "eu-north1", "us-west-4"]),
    ("kpi_uptime", "Uptime (30d)",
     ["99.982%", "99.917%", "99.874%", "99.995%", "99.940%"]),
    ("alerts_7d", "Pager alerts (7d)",
     ["14", "27", "9", "33", "21"]),
    ("mttr", "Mean time to recovery",
     ["42 min", "18 min", "67 min", "25 min", "93 min"]),
    ("queue_depth", "Peak queue depth (24h)",
     ["12,480", "8,915", "17,203", "6,542", "21,077"]),
    ("dlq_events", "Dead-letter events (7d)",
     ["312", "87", "540", "129", "764"]),
    ("cdn_hit", "CDN hit ratio",
     ["94.1%", "88.7%", "97.3%", "82.5%", "91.6%"]),
]

def t1_body(vals, dbg):
    feed = "".join(f'<div class="feed">{t}</div>' for t in [
        "09:02 deploy pipeline promoted build 8841 to canary",
        "08:47 autoscaler added 6 workers in pool general-b",
        "08:31 config change: retry budget raised for relay-eu",
        "08:12 nightly compaction finished in 41m 22s",
        "07:58 certificate rotation completed for edge tier",
        "07:40 replay of dead-letter queue drained 312 events",
        "07:15 sandbox refresh completed for 14 tenants",
        "06:59 slow-query report generated (11 offenders)",
        "06:30 backup verification passed for shard 0-7",
        "06:02 traffic mirror to staging disabled after test",
    ])
    deploys = "".join(
        f"<tr><td>{b}</td><td>{w}</td><td>{d}</td><td>{r}</td></tr>"
        for b, w, d, r in [
            ("8836", "api-core", "Mon 14:02", "success"),
            ("8829", "relay", "Mon 09:44", "success"),
            ("8821", "api-core", "Fri 16:20", "rolled back"),
            ("8817", "billing", "Fri 11:05", "success"),
            ("8810", "web", "Thu 15:33", "success"),
            ("8804", "relay", "Thu 10:12", "success"),
            ("8799", "api-core", "Wed 17:41", "success"),
            ("8791", "web", "Wed 09:27", "success"),
        ])
    F = lambda l, f: frow(l, f, vals, dbg)
    return f"""
<h1>Atlas Deployment Console</h1>
<p>{LOREM[0]}</p>
<div class="card"><h3>Service overview</h3>
{F("Release version","release_ver")}{F("Environment","deploy_env")}
{F("Build status","build_status")}{F("Error budget used","budget_used")}
</div>
<div class="card"><h3>Recent activity</h3>{feed}</div>
<p>{LOREM[1]}</p>
<div class="card"><h3>Capacity</h3>
{F("CPU quota","cpu_quota")}{F("Memory quota","mem_quota")}
{F("Storage in use","storage_used")}{F("Active users (24h)","kpi_users")}
</div>
<p>{LOREM[3]}</p><p>{LOREM[4]}</p>
<div class="card"><h3>Operations</h3>
{F("On-call engineer","oncall_name")}{F("Open tickets","tickets_open")}
{F("Primary region","region")}{F("Uptime (30d)","kpi_uptime")}
</div>
<div class="card"><h3>Deploy history</h3>
<table><tr><th>Build</th><th>Service</th><th>When</th><th>Result</th></tr>{deploys}</table>
</div>
<p>{LOREM[6]}</p>
<div class="card"><h3>Alerting</h3>
{F("Pager alerts (7d)","alerts_7d")}{F("Mean time to recovery","mttr")}
</div>
<p>{LOREM[2]}</p><p>{LOREM[7]}</p>
<div class="card"><h3>Incident review queue</h3>
<table><tr><th>Id</th><th>Title</th><th>Severity</th><th>State</th></tr>
<tr><td>INC-2214</td><td>latency regression in relay-eu</td><td>S2</td><td>review scheduled</td></tr>
<tr><td>INC-2209</td><td>stale cache served after rollout</td><td>S3</td><td>action items open</td></tr>
<tr><td>INC-2201</td><td>alert storm from retuned monitor</td><td>S4</td><td>closed</td></tr>
<tr><td>INC-2196</td><td>partial write outage, shard 3</td><td>S2</td><td>closed</td></tr>
<tr><td>INC-2190</td><td>expired cert on internal edge</td><td>S3</td><td>closed</td></tr>
<tr><td>INC-2184</td><td>runaway batch job exhausted pool</td><td>S3</td><td>closed</td></tr>
</table></div>
<p>{LOREM[5]}</p>
<div class="card"><h3>Queues &amp; delivery</h3>
{F("Peak queue depth (24h)","queue_depth")}{F("Dead-letter events (7d)","dlq_events")}
{F("CDN hit ratio","cdn_hit")}
</div>
<p>{LOREM[4]}</p><p>{LOREM[6]}</p><p>{LOREM[3]}</p>
"""


# --- template 2: mail / list view ---
T2_FIELDS = [
    ("unread_total", "Unread messages", ["128", "94", "211", "67", "153"]),
    ("flagged", "Flagged", ["17", "9", "31", "24", "5"]),
    ("drafts", "Drafts", ["6", "12", "3", "9", "15"]),
    ("spam_count", "Spam caught (7d)", ["482", "391", "560", "227", "648"]),
    ("rule_hits", "Rule matches (7d)", ["1,204", "877", "1,592", "643", "2,018"]),
    ("quota_used", "Quota used", ["61.3%", "48.9%", "77.6%", "35.2%", "84.1%"]),
    ("archive_count", "Archived threads", ["8,417", "9,203", "7,655", "11,048", "6,392"]),
    ("last_backup", "Last backup size", ["3.9 GB", "4.6 GB", "3.1 GB", "5.4 GB", "2.8 GB"]),
    ("list_subs", "List subscriptions", ["37", "52", "28", "61", "44"]),
    ("bounce_rate", "Bounce rate", ["0.42%", "0.87%", "0.19%", "1.13%", "0.65%"]),
    ("digest_day", "Digest day", ["Monday", "Tuesday", "Thursday", "Friday", "Sunday"]),
    ("retention", "Retention window", ["180 days", "365 days", "90 days", "540 days", "270 days"]),
    ("tls_rate", "TLS delivery rate", ["99.2%", "97.8%", "99.9%", "95.4%", "98.6%"]),
    ("dkim_pass", "DKIM pass rate", ["98.4%", "96.1%", "99.7%", "93.8%", "97.2%"]),
]

_MAILS = [
    ("Priya Nair", "Re: relay cutover checklist", "the two remaining items are the cert swap and the", "09:41"),
    ("Build Bot", "nightly build 8841 finished", "all suites green, artifact uploaded to the shared", "08:55"),
    ("Tomas Ekberg", "Q3 vendor review notes", "attaching the comparison sheet we discussed, the", "08:12"),
    ("Alice Wong", "office move logistics", "desks on the 4th floor will be available from the", "07:58"),
    ("Ops Alerts", "[resolved] latency regression", "p95 returned to baseline after the cache change", "07:31"),
    ("Marta Silva", "interview loop for platform role", "can you take the systems session on thursday, the", "07:02"),
    ("Ken Osei", "draft: postmortem for INC-2214", "please review section 3, the timeline still has a", "06:47"),
    ("Newsletter", "weekly infra digest #142", "this week: profile-guided rollouts, a note on the", "06:15"),
    ("Priya Nair", "budget corridor check-in", "we are still inside the corridor, egress is down", "Tue"),
    ("Sam Porter", "keys for the demo tenant", "rotated this morning, the old ones stop working", "Tue"),
    ("Build Bot", "flaky test quarantine report", "4 tests entered quarantine, 2 exited after fixes", "Tue"),
    ("Legal Team", "updated data processing terms", "the redlines from the customer are minor, mostly", "Mon"),
    ("Alice Wong", "catering for the offsite", "final headcount needed by wednesday noon, please", "Mon"),
    ("Ops Alerts", "[info] maintenance window set", "regional relays will drain for 20 minutes on the", "Mon"),
    ("Tomas Ekberg", "contract renewal timeline", "procurement wants the summary two weeks before", "Mon"),
    ("Ken Osei", "runbook corrections merged", "the credential rotation ordering is fixed now in", "Fri"),
    ("Marta Silva", "offer accepted!", "great news, the second candidate signed today so", "Fri"),
    ("Sam Porter", "demo script walkthrough", "recorded a dry run, the pacing works better with", "Fri"),
    ("Newsletter", "security bulletin 2026-08", "one advisory relevant to our stack, patched in", "Thu"),
    ("Priya Nair", "sync notes: storage tiering", "warm tier hit rates recovered, the spill policy", "Thu"),
    ("Build Bot", "dependency audit summary", "3 updates pending, none with breaking changes in", "Thu"),
    ("Alice Wong", "travel approvals for october", "submit requests before the 15th to make the batch", "Wed"),
    ("Ops Alerts", "[resolved] queue depth alarm", "retuned to p95 dequeue latency, no action needed", "Wed"),
    ("Ken Osei", "tabletop exercise recap", "failover met the objective, two runbook fixes are", "Wed"),
]

def t2_body(vals, dbg):
    F = lambda l, f: frow(l, f, vals, dbg)
    m = lambda a, b: "".join(mailrow(*r) for r in _MAILS[a:b])
    return f"""
<h1>Relay Mail — Team Inbox</h1>
<div class="card"><h3>Mailbox summary</h3>
{F("Unread messages","unread_total")}{F("Flagged","flagged")}{F("Drafts","drafts")}
</div>
<div class="card"><h3>Today</h3>{m(0,8)}</div>
<div class="card"><h3>Filtering</h3>
{F("Spam caught (7d)","spam_count")}{F("Rule matches (7d)","rule_hits")}
</div>
<div class="card"><h3>Earlier this week</h3>{m(8,16)}</div>
<div class="card"><h3>Storage</h3>
{F("Quota used","quota_used")}{F("Archived threads","archive_count")}
{F("Last backup size","last_backup")}
</div>
<div class="card"><h3>Last week</h3>{m(16,24)}</div>
<div class="card"><h3>Lists &amp; retention</h3>
{F("List subscriptions","list_subs")}{F("Bounce rate","bounce_rate")}
{F("Digest day","digest_day")}{F("Retention window","retention")}
</div>
<div class="card"><h3>Older</h3>{"".join(mailrow(*r) for r in reversed(_MAILS[2:14]))}</div>
<div class="card"><h3>Delivery health</h3>
{F("TLS delivery rate","tls_rate")}{F("DKIM pass rate","dkim_pass")}
</div>
<p>{LOREM[4]}</p><p>{LOREM[6]}</p>
"""


# --- template 3: settings page ---
T3_FIELDS = [
    ("plan", "Subscription plan", ["Business", "Standard", "Premium", "Starter", "Scale"]),
    ("seats", "Licensed seats", ["120", "85", "240", "60", "175"]),
    ("twofa", "Two-factor auth", ["Enforced", "Optional", "Disabled", "SSO only", "Per-team"]),
    ("session_timeout", "Session timeout", ["8 hours", "24 hours", "1 hour", "12 hours", "4 hours"]),
    ("theme", "Interface theme", ["System", "Light", "Dark", "Contrast", "Compact"]),
    ("language", "Display language", ["English", "Deutsch", "Français", "한국어", "Español"]),
    ("email_digest", "Email digest", ["Weekly", "Daily", "Off", "Monthly", "Realtime"]),
    ("push_alerts", "Push alerts", ["Mentions", "All", "None", "Critical", "Custom"]),
    ("data_region", "Data residency", ["EU (Frankfurt)", "US (Oregon)", "APAC (Seoul)", "EU (Dublin)", "US (Ohio)"]),
    ("telemetry", "Usage telemetry", ["Minimal", "Standard", "Off", "Extended", "Anonymous"]),
    ("api_limit", "API rate limit", ["600 req/min", "1200 req/min", "300 req/min", "2400 req/min", "150 req/min"]),
    ("webhook_status", "Webhook delivery", ["Healthy", "Degraded", "Paused", "Failing", "Retrying"]),
    ("log_level", "Audit log level", ["Info", "Debug", "Warning", "Verbose", "Error"]),
    ("cache_ttl", "Cache TTL", ["15 min", "60 min", "5 min", "240 min", "30 min"]),
    ("billing_cycle", "Billing cycle", ["Annual", "Monthly", "Quarterly", "Biennial", "Custom"]),
    ("invoice_count", "Invoices on file", ["36", "12", "48", "24", "60"]),
    ("sso_provider", "SSO provider", ["Okta", "Entra ID", "OneLogin", "Ping", "JumpCloud"]),
    ("scim_sync", "SCIM directory sync", ["Hourly", "Daily", "Realtime", "Paused", "Weekly"]),
    ("export_fmt", "Scheduled export format", ["Parquet", "CSV", "JSONL", "Avro", "ORC"]),
]

_T3_DESC = {
    "plan": "Determines feature availability and the support tier attached to the workspace.",
    "seats": "Members beyond this count are placed in a read-only waiting state.",
    "twofa": "Applies at next sign-in; active sessions are not interrupted.",
    "session_timeout": "Idle sessions are signed out after this period on shared devices.",
    "theme": "System follows the operating system preference where reported.",
    "language": "Applies to menus and notifications; documents keep their own language.",
    "email_digest": "A summary of workspace activity delivered to each member.",
    "push_alerts": "Mobile and desktop notification scope for this workspace.",
    "data_region": "Where primary data is stored; changing it requires a migration window.",
    "telemetry": "Controls what usage data is shared with the vendor.",
    "api_limit": "Applies per token; bursts above the limit receive HTTP 429.",
    "webhook_status": "Aggregated delivery health over the trailing hour.",
    "log_level": "Verbosity of entries written to the audit trail.",
    "cache_ttl": "How long derived views may be served without revalidation.",
    "billing_cycle": "Invoices are issued at the start of each cycle.",
    "invoice_count": "Historical invoices retained and downloadable from this page.",
    "sso_provider": "Identity provider used for single sign-on across the workspace.",
    "scim_sync": "How often group membership is reconciled with the directory.",
    "export_fmt": "Format used by the nightly warehouse export job.",
}

def t3_row(fid, label, vals, dbg):
    return (f'<div class="row"><span class="lbl">{label}'
            f'<div class="desc">{_T3_DESC[fid]}</div></span>'
            f'{V(fid, vals[fid], dbg)}</div>')

def t3_body(vals, dbg):
    R = lambda f, l: t3_row(f, l, vals, dbg)
    return f"""
<h1>Workspace Settings</h1>
<p>Changes save automatically and are recorded in the audit trail. Some
settings require an administrator role to modify.</p>
<div class="card"><h3>Account</h3>
{R("plan","Subscription plan")}{R("seats","Licensed seats")}
{R("twofa","Two-factor auth")}{R("session_timeout","Session timeout")}
</div>
<p>{LOREM[2]}</p>
<div class="card"><h3>Appearance &amp; language</h3>
{R("theme","Interface theme")}{R("language","Display language")}
</div>
<div class="card"><h3>Notifications</h3>
{R("email_digest","Email digest")}{R("push_alerts","Push alerts")}
</div>
<p>{LOREM[5]}</p>
<div class="card"><h3>Privacy &amp; compliance</h3>
{R("data_region","Data residency")}{R("telemetry","Usage telemetry")}
</div>
<div class="card"><h3>Developer</h3>
{R("api_limit","API rate limit")}{R("webhook_status","Webhook delivery")}
{R("log_level","Audit log level")}{R("cache_ttl","Cache TTL")}
</div>
<p>{LOREM[7]}</p>
<div class="card"><h3>Billing</h3>
{R("billing_cycle","Billing cycle")}{R("invoice_count","Invoices on file")}
</div>
<p>{LOREM[0]}</p><p>{LOREM[3]}</p>
<div class="card"><h3>Integrations</h3>
{R("sso_provider","SSO provider")}{R("scim_sync","SCIM directory sync")}
{R("export_fmt","Scheduled export format")}
</div>
<p>{LOREM[1]}</p><p>{LOREM[6]}</p>
"""


# --- template 4: table report ---
T4_FIELDS = [
    ("varA_eng", "Engineering variance", ["+2.4%", "-1.8%", "+5.1%", "-3.6%", "+0.9%"]),
    ("varA_ops", "Operations variance", ["-0.7%", "+3.2%", "-2.5%", "+1.6%", "-4.1%"]),
    ("varA_sec", "Security variance", ["+1.1%", "-2.9%", "+4.3%", "-0.4%", "+2.8%"]),
    ("varA_dat", "Data variance", ["-3.3%", "+0.6%", "-1.2%", "+2.1%", "-5.0%"]),
    ("total_spend", "Total spend (Q3)", ["$4.82M", "$5.14M", "$4.37M", "$5.66M", "$3.98M"]),
    ("forecast_q4", "Forecast (Q4)", ["$5.05M", "$4.61M", "$5.48M", "$4.20M", "$5.92M"]),
    ("reqs", "Approved requisitions", ["142", "118", "167", "95", "203"]),
    ("audits", "Pending audits", ["7", "12", "4", "18", "9"]),
    ("vendors", "Active vendors", ["58", "43", "71", "36", "64"]),
    ("renewals", "Renewals due (90d)", ["11", "6", "17", "23", "8"]),
    ("open_roles", "Open requisitioned roles", ["19", "8", "27", "14", "33"]),
    ("attrition", "Trailing attrition", ["6.2%", "9.8%", "4.1%", "11.5%", "7.7%"]),
]

_T4A = [  # dept, headcount, budget, spend (static); variance cell mutable for 4 rows
    ("Engineering", "184", "$1.92M", "$1.88M", "varA_eng"),
    ("Product", "46", "$0.51M", "$0.49M", None),
    ("Operations", "72", "$0.83M", "$0.86M", "varA_ops"),
    ("Design", "18", "$0.22M", "$0.21M", None),
    ("Security", "31", "$0.44M", "$0.41M", "varA_sec"),
    ("Support", "55", "$0.47M", "$0.48M", None),
    ("Data", "39", "$0.58M", "$0.61M", "varA_dat"),
    ("Marketing", "27", "$0.36M", "$0.35M", None),
]
_T4B = [
    ("Compute", "$1.41M", "$1.38M", "$1.52M", "-2.1%"),
    ("Storage", "$0.62M", "$0.66M", "$0.71M", "+6.5%"),
    ("Network egress", "$0.48M", "$0.39M", "$0.36M", "-18.8%"),
    ("Observability", "$0.21M", "$0.27M", "$0.30M", "+28.6%"),
    ("Licenses", "$0.55M", "$0.55M", "$0.57M", "0.0%"),
    ("Facilities", "$0.33M", "$0.34M", "$0.34M", "+3.0%"),
    ("Travel", "$0.09M", "$0.14M", "$0.12M", "+55.6%"),
    ("Training", "$0.07M", "$0.08M", "$0.10M", "+14.3%"),
    ("Recruiting", "$0.18M", "$0.16M", "$0.15M", "-11.1%"),
    ("Contingency", "$0.10M", "$0.05M", "$0.08M", "-50.0%"),
]

def t4_body(vals, dbg):
    rowsA = ""
    for dept, hc, bud, sp, fid in _T4A:
        var = V(fid, vals[fid], dbg) if fid else "&mdash;"
        rowsA += (f"<tr><td>{dept}</td><td>{hc}</td><td>{bud}</td>"
                  f"<td>{sp}</td><td>{var}</td></tr>")
    rowsB = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td><td>{e}</td></tr>"
                    for a, b, c, d, e in _T4B)
    F = lambda l, f: frow(l, f, vals, dbg)
    return f"""
<h1>Q3 Operating Report</h1>
<p>{LOREM[5]}</p><p>{LOREM[1]}</p>
<div class="card"><h3>Table A — spend by department</h3>
<table><tr><th>Department</th><th>Headcount</th><th>Budget</th><th>Spend</th><th>Variance</th></tr>
{rowsA}</table></div>
<p>{LOREM[3]}</p>
<div class="card"><h3>Summary figures</h3>
{F("Total spend (Q3)","total_spend")}{F("Forecast (Q4)","forecast_q4")}
{F("Approved requisitions","reqs")}{F("Pending audits","audits")}
</div>
<div class="card"><h3>Table B — spend by category (Q1&ndash;Q3)</h3>
<table><tr><th>Category</th><th>Q1</th><th>Q2</th><th>Q3</th><th>&Delta; QoQ</th></tr>
{rowsB}</table></div>
<p>{LOREM[6]}</p><p>{LOREM[2]}</p>
<div class="card"><h3>Procurement</h3>
{F("Active vendors","vendors")}{F("Renewals due (90d)","renewals")}
</div>
<p>{LOREM[7]}</p><p>{LOREM[0]}</p>
<div class="card"><h3>Table C — regional allocation</h3>
<table><tr><th>Region</th><th>Budget</th><th>Spend</th><th>Utilization</th></tr>
<tr><td>EMEA</td><td>$1.62M</td><td>$1.58M</td><td>97.5%</td></tr>
<tr><td>Americas</td><td>$2.04M</td><td>$2.11M</td><td>103.4%</td></tr>
<tr><td>APAC</td><td>$0.88M</td><td>$0.79M</td><td>89.8%</td></tr>
<tr><td>LATAM</td><td>$0.31M</td><td>$0.28M</td><td>90.3%</td></tr>
<tr><td>Central</td><td>$0.47M</td><td>$0.51M</td><td>108.5%</td></tr>
</table></div>
<p>{LOREM[4]}</p>
<div class="card"><h3>Headcount planning</h3>
{F("Open requisitioned roles","open_roles")}{F("Trailing attrition","attrition")}
</div>
<p>{LOREM[1]}</p><p>{LOREM[5]}</p>
"""


# --- template 5: article with stats ---
T5_FIELDS = [
    ("survey_n", "Survey responses", ["1,204", "1,487", "982", "1,731", "1,058"]),
    ("adoption", "Toolchain adoption", ["78%", "64%", "83%", "57%", "91%"]),
    ("build_time", "Median build time", ["6m 12s", "4m 48s", "8m 03s", "5m 27s", "7m 39s"]),
    ("flake_rate", "Test flake rate", ["1.9%", "3.4%", "0.8%", "2.7%", "4.6%"]),
    ("review_latency", "Median review latency", ["4.1 h", "7.8 h", "2.6 h", "5.9 h", "9.2 h"]),
    ("incidents", "Tooling incidents (12mo)", ["23", "37", "15", "44", "29"]),
    ("teams_migrated", "Teams migrated", ["61", "48", "74", "35", "82"]),
    ("loc_scanned", "Lines scanned daily", ["48M", "62M", "35M", "77M", "54M"]),
    ("cost_per_seat", "Cost per seat", ["$118", "$142", "$97", "$165", "$126"]),
    ("satisfaction", "Developer satisfaction", ["7.4/10", "6.8/10", "8.1/10", "6.2/10", "7.9/10"]),
    ("oncall_load", "Median on-call pages/week", ["3.2", "5.7", "1.8", "7.4", "4.6"]),
    ("docs_fresh", "Docs updated within 90d", ["58%", "71%", "44%", "83%", "62%"]),
]

def t5_stat(fid, label, vals, dbg):
    return (f'<div class="stat"><span class="lbl" style="width:auto">{label}: </span>'
            f'{V(fid, vals[fid], dbg)}</div>')

def t5_body(vals, dbg):
    S = lambda f, l: t5_stat(f, l, vals, dbg)
    return f"""
<h1>State of Internal Tooling, 2026</h1>
<p style="color:#7b8494">Platform research group &middot; annual survey report</p>
<p>{LOREM[0]}</p>
{S("survey_n","Survey responses")}
<p>{LOREM[1]}</p><p>{LOREM[2]}</p>
{S("adoption","Toolchain adoption")}
<h2>Build and test</h2>
<p>{LOREM[3]}</p>
{S("build_time","Median build time")}
<p>{LOREM[4]}</p>
{S("flake_rate","Test flake rate")}
<h2>Code review</h2>
<p>{LOREM[5]}</p>
{S("review_latency","Median review latency")}
<p>{LOREM[6]}</p>
{S("incidents","Tooling incidents (12mo)")}
<h2>Migration progress</h2>
<p>{LOREM[7]}</p>
{S("teams_migrated","Teams migrated")}
<p>{LOREM[1]}</p>
{S("loc_scanned","Lines scanned daily")}
<h2>Cost and sentiment</h2>
<p>{LOREM[3]}</p>
{S("cost_per_seat","Cost per seat")}
<p>{LOREM[5]}</p>
{S("satisfaction","Developer satisfaction")}
<p>{LOREM[6]}</p>
<h2>Operational load</h2>
<p>{LOREM[2]}</p>
{S("oncall_load","Median on-call pages/week")}
<p>{LOREM[7]}</p>
{S("docs_fresh","Docs updated within 90d")}
<p>{LOREM[0]}</p><p>{LOREM[4]}</p>
"""


TEMPLATES = {
    "dashboard": (T1_FIELDS, t1_body, "Atlas Deployment Console"),
    "inbox": (T2_FIELDS, t2_body, "Relay Mail"),
    "settings": (T3_FIELDS, t3_body, "Workspace Settings"),
    "table_report": (T4_FIELDS, t4_body, "Q3 Operating Report"),
    "article": (T5_FIELDS, t5_body, "Tooling Report 2026"),
}


def page_html(tname, vals, scroll, clock, sync, snap_id, dbg=None):
    fields, body_fn, title = TEMPLATES[tname]
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>{CSS}</style></head><body>
<div class="hdr"><div class="hdr-top"><span>{title}</span><span>{clock}</span></div>
<div class="hdr-sub">Last sync: {sync} &middot; snapshot {snap_id}</div></div>
<div style="margin-top:{HEADER_H - scroll}px;min-height:3300px">
<div class="wrap">{body_fn(vals, dbg)}</div>
</div></body></html>"""


# ============================================================================
# 2. Rendering
# ============================================================================

def render(html_str: str, png_path: Path):
    png_path.parent.mkdir(parents=True, exist_ok=True)
    html_path = HTML_DIR / (png_path.stem + ".html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_str)
    prof = tempfile.mkdtemp(prefix="chrprof_")
    try:
        cmd = ["google-chrome", "--headless", "--disable-gpu", "--no-sandbox",
               "--hide-scrollbars", f"--window-size={W},{H}",
               f"--user-data-dir={prof}", f"--screenshot={png_path}",
               f"file://{html_path}"]
        r = subprocess.run(cmd, capture_output=True, timeout=90)
        if not png_path.exists():
            raise RuntimeError(f"chrome failed for {png_path.name}: "
                               f"{r.stderr.decode()[-400:]}")
    finally:
        shutil.rmtree(prof, ignore_errors=True)
    return png_path


def render_many(jobs):
    """jobs: list of (html_str, png_path). Parallel, skip existing."""
    todo = [(h, p) for h, p in jobs if not p.exists()]
    if todo:
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(lambda j: render(*j), todo))


def gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def diff_bbox(plain_png, debug_png, thresh=25):
    a = np.asarray(Image.open(plain_png).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(debug_png).convert("RGB"), dtype=np.int16)
    m = (np.abs(a - b).max(axis=2) > thresh)
    m[:HEADER_H, :] = False   # header clock identical anyway; belt & braces
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


# ============================================================================
# 3. Calibration: field y-positions per template
# ============================================================================

def calibrate():
    """Render each template plain + all-fields-highlighted at two scrolls;
    diff -> per-field content-space y ranges (fields are on distinct rows,
    DOM order == top-to-bottom order)."""
    cal = {}
    jobs = []
    meta = []
    for tname, (fields, _, _) in TEMPLATES.items():
        vals = {fid: pool[0] for fid, _, pool in fields}
        for s in (0, MAX_SCROLL):
            for mode in ("plain", "dball"):
                dbg = "__ALL__" if mode == "dball" else None
                # __ALL__: highlight every field
                html = page_html(tname, vals, s, "12:00:00", "12:00:00",
                                 "cal", dbg=None)
                if dbg:
                    # brute force: highlight all by marking each fid
                    fields_html = html
                    for fid, _, pool in fields:
                        fields_html = fields_html.replace(
                            f'id="f_{fid}" style="',
                            f'id="f_{fid}" style="background:#ff00ff;'
                            f'color:#ff00ff;outline:4px solid #ff00ff;')
                    html = fields_html
                p = PNG_DIR / "calib" / f"{tname}_s{s}_{mode}.png"
                jobs.append((html, p))
                meta.append((tname, s, mode, p))
    render_many(jobs)

    seen = {}   # (tname, scroll) -> list of (y0,y1) screen space
    for tname in TEMPLATES:
        for s in (0, MAX_SCROLL):
            plain = PNG_DIR / "calib" / f"{tname}_s{s}_plain.png"
            dball = PNG_DIR / "calib" / f"{tname}_s{s}_dball.png"
            a = np.asarray(Image.open(plain).convert("RGB"), dtype=np.int16)
            b = np.asarray(Image.open(dball).convert("RGB"), dtype=np.int16)
            m = (np.abs(a - b).max(axis=2) > 25)
            m[:HEADER_H, :] = False
            rows = np.where(m.any(axis=1))[0]
            segs = []
            if len(rows):
                start = prev = rows[0]
                for y in rows[1:]:
                    if y - prev > 6:
                        segs.append((start, prev + 1))
                        start = y
                    prev = y
                segs.append((start, prev + 1))
            seen[(tname, s)] = segs

    for tname, (fields, _, _) in TEMPLATES.items():
        n = len(fields)
        seg0 = seen[(tname, 0)]                 # first k fields (content y asc)
        seg1 = seen[(tname, MAX_SCROLL)]        # last m fields
        k, m = len(seg0), len(seg1)
        assert k + m >= n, f"{tname}: calibration covers {k}+{m} < {n} fields"

        def to_content(seg, s):
            y0, y1 = seg
            clipped = (y0 <= HEADER_H + 2) or (y1 >= H - 2)
            return (y0 - HEADER_H + s, y1 - HEADER_H + s, clipped)

        ys = {}                                 # idx -> (y0, y1, clipped)
        for i, seg in enumerate(seg0):
            ys[i] = to_content(seg, 0)
        for j, seg in enumerate(seg1):
            idx = n - m + j
            got = to_content(seg, MAX_SCROLL)
            if idx in ys:
                have = ys[idx]
                if not (have[2] or got[2]):     # both unclipped: must agree
                    assert abs(have[0] - got[0]) <= 4, \
                        f"{tname} field {idx}: {have} vs {got}"
                if have[2] and not got[2]:      # prefer the unclipped view
                    ys[idx] = got
            else:
                ys[idx] = got
        assert len(ys) == n, f"{tname}: mapped {len(ys)}/{n} fields"
        assert not any(v[2] for v in ys.values()), \
            f"{tname}: field visible only clipped: " \
            f"{[i for i, v in ys.items() if v[2]]}"
        cal[tname] = {fields[i][0]: (ys[i][0], ys[i][1]) for i in range(n)}
    return cal


# ============================================================================
# 4. Pair generation
# ============================================================================

def rand_clock(rnd):
    return f"2026-08-{rnd.randint(10,20):02d} {rnd.randint(0,23):02d}:" \
           f"{rnd.randint(0,59):02d}:{rnd.randint(0,59):02d} UTC"


def make_pairs(cal):
    tnames = list(TEMPLATES)
    design = []
    for cell, lo, hi in CELLS:
        for changed in (True, False):
            for k in range(PAIRS_PER_CELL):
                design.append((cell, lo, hi, changed))
    pairs = []
    jobs = []
    for i, (cell, lo, hi, changed) in enumerate(design):
        tname = tnames[i % len(tnames)]
        fields, _, _ = TEMPLATES[tname]
        rnd = random.Random(SEED + 1000 * i)
        vals_old = {fid: rnd.choice(pool) for fid, _, pool in fields}
        s_old = rnd.randint(0, 1200)
        frac = False
        if lo == hi == 0:
            delta = 0
        else:
            mag = rnd.randint(lo, hi)
            sgn = rnd.choice([-1, 1])
            if s_old + sgn * mag < 0 or s_old + sgn * mag > MAX_SCROLL - 1:
                sgn = -sgn
            delta = sgn * mag
            # half of the nonzero-offset pairs get a 0.5px subpixel component
            # (momentum scrolling / DPR scaling leave non-integer positions;
            # Chrome then rasterizes with antialiasing, so unchanged pairs
            # stop being pixel-identical after integer alignment)
            frac = (i % 2 == 1)
            if frac:
                delta += 0.5
        s_new = s_old + delta
        clock_o, clock_n = rand_clock(rnd), rand_clock(rnd)
        sync_o, sync_n = rand_clock(rnd), rand_clock(rnd)
        sid_o = f"{rnd.randrange(16**8):08x}"
        sid_n = f"{rnd.randrange(16**8):08x}"

        vals_new = dict(vals_old)
        mut_fid = None
        if changed:
            # field must be fully visible in BOTH viewports (else the change
            # is undetectable by construction)
            cands = []
            for fid, _, pool in fields:
                y0, y1 = cal[tname][fid]
                vis = lambda s: (y0 - s >= 12 and
                                 y1 - s <= H - HEADER_H - 12)
                if vis(s_old) and vis(s_new):
                    cands.append((fid, pool))
            assert cands, f"pair {i}: no field visible in both viewports " \
                          f"({tname} s_old={s_old} s_new={s_new})"
            mut_fid, pool = rnd.choice(cands)
            vals_new[mut_fid] = rnd.choice(
                [v for v in pool if v != vals_old[mut_fid]])

        pid = f"p{i:03d}_{tname}_{cell}_{'chg' if changed else 'unc'}"
        p_old = PNG_DIR / f"{pid}_old.png"
        p_new = PNG_DIR / f"{pid}_new.png"
        jobs.append((page_html(tname, vals_old, s_old, clock_o, sync_o, sid_o),
                     p_old))
        jobs.append((page_html(tname, vals_new, s_new, clock_n, sync_n, sid_n),
                     p_new))
        rec = {"pair_id": pid, "template": tname, "cell": cell,
               "changed": changed, "scroll_old": s_old, "scroll_new": s_new,
               "dy_true": s_new - s_old, "fractional": frac,
               "mutated_field": mut_fid,
               "old_png": str(p_old), "new_png": str(p_new)}
        if changed:
            p_dbg = PNG_DIR / "debug" / f"{pid}_newdbg.png"
            jobs.append((page_html(tname, vals_new, s_new, clock_n, sync_n,
                                   sid_n, dbg=mut_fid), p_dbg))
            rec["dbg_png"] = str(p_dbg)
            rec["old_value"] = vals_old[mut_fid]
            rec["new_value"] = vals_new[mut_fid]
        pairs.append(rec)

    render_many(jobs)

    for rec in pairs:                       # ground-truth bbox from debug diff
        if rec["changed"]:
            bbox = diff_bbox(rec["new_png"], rec["dbg_png"])
            assert bbox is not None, f"{rec['pair_id']}: empty debug diff"
            x1, y1, x2, y2 = bbox
            assert y1 >= HEADER_H and y2 <= H and (y2 - y1) <= 90, \
                f"{rec['pair_id']}: suspicious bbox {bbox}"
            rec["bbox"] = bbox
    return pairs


# ============================================================================
# 5. Detectors
# ============================================================================

def estimate_dy(go, gn, max_shift=MAX_SHIFT):
    """Vertical offset dy s.t. new[y] ~ old[y+dy]; row-profile NCC."""
    po = go.mean(axis=1)
    pn = gn.mean(axis=1)
    n = len(pn)
    best_d, best_s = 0, -2.0
    for d in range(-max_shift, max_shift + 1):
        if d >= 0:
            a, b = pn[:n - d], po[d:]
        else:
            a, b = pn[-d:], po[:n + d]
        if len(a) < MIN_OVERLAP:
            continue
        am, bm = a - a.mean(), b - b.mean()
        den = np.sqrt((am * am).sum() * (bm * bm).sum())
        s = float((am * bm).sum() / den) if den > 1e-9 else -1.0
        if s > best_s:
            best_s, best_d = s, d
    return best_d, best_s


def run_detectors(rec):
    old = Image.open(rec["old_png"])
    new = Image.open(rec["new_png"])
    go, gn = (np.asarray(old.convert("L"), dtype=np.float64),
              np.asarray(new.convert("L"), dtype=np.float64))
    out = {}

    # --- naive (no alignment), one tile pass shared by both mask variants
    t0 = time.perf_counter()
    tiles = tile_diff(old, new)
    t_naive = 1000 * (time.perf_counter() - t0)
    out["naive_unmasked"] = {"score": max(tiles.values()), "ms": t_naive}
    masked = {k: v for k, v in tiles.items() if k[1] >= HEADER_H}
    top = max(masked, key=masked.get)
    out["naive_masked"] = {"score": max(masked.values()), "ms": t_naive,
                           "top_tile": [top[0], top[1]]}

    # --- aligned variants
    for name, y0 in (("aligned_unmasked", 0), ("aligned_masked", HEADER_H)):
        t0 = time.perf_counter()
        d, ncc = estimate_dy(go[y0:], gn[y0:])
        hh = H - y0
        ys, ye = max(0, -d), hh - max(0, d)
        new_c = new.crop((0, y0 + ys, W, y0 + ye))
        old_c = old.crop((0, y0 + ys + d, W, y0 + ye + d))
        tl = tile_diff(old_c, new_c)
        ms = 1000 * (time.perf_counter() - t0)
        top = max(tl, key=tl.get)
        out[name] = {"score": max(tl.values()), "ms": ms, "dy_hat": d,
                     "ncc": round(ncc, 4),
                     "top_tile": [top[0], top[1] + y0 + ys]}
    return out


# ============================================================================
# 6. Metrics
# ============================================================================

def best_acc(pos, neg):
    cands = sorted(set(pos + neg))
    best = 0.0
    for t in cands:
        acc = (sum(p > t for p in pos) + sum(n <= t for n in neg)) / \
              (len(pos) + len(neg))
        best = max(best, acc)
    return best


def summarize(pairs, results):
    variants = ["naive_unmasked", "naive_masked",
                "aligned_unmasked", "aligned_masked"]
    rep = {"design": {"pairs": len(pairs), "pairs_per_cell": PAIRS_PER_CELL,
                      "templates": sorted(TEMPLATES),
                      "offsets": {"zero": "0px", "small": "30-80px",
                                  "large": "200-600px"},
                      "subpixel": "half of nonzero-offset pairs carry an "
                                  "extra +0.5px (antialiased rasterization)",
                      "viewport": [W, H], "fixed_header_px": HEADER_H,
                      "tile_px": TILE, "seed": SEED},
           "per_cell": {}, "overall": {}}

    def cell_of(r):
        return r["cell"]

    tmpl_counts = {}
    for r in pairs:
        key = f"{r['template']}/{r['cell']}/{'chg' if r['changed'] else 'unc'}"
        tmpl_counts[key] = tmpl_counts.get(key, 0) + 1
    rep["design"]["counts"] = tmpl_counts

    for cell in ["zero", "small", "large"]:
        sel = [i for i, r in enumerate(pairs) if r["cell"] == cell]
        crec = {"n_changed": sum(pairs[i]["changed"] for i in sel),
                "n_unchanged": sum(not pairs[i]["changed"] for i in sel),
                "detectors": {}}
        for v in variants:
            pos = [results[i][v]["score"] for i in sel if pairs[i]["changed"]]
            neg = [results[i][v]["score"] for i in sel if not pairs[i]["changed"]]
            crec["detectors"][v] = {
                "auroc": round(auroc(pos, neg), 4),
                "best_threshold_accuracy": round(best_acc(pos, neg), 4),
                "changed_score_min": round(min(pos), 2),
                "unchanged_score_max": round(max(neg), 2)}
        for v in ("aligned_unmasked", "aligned_masked"):
            errs = [abs(results[i][v]["dy_hat"] - pairs[i]["dy_true"])
                    for i in sel]
            crec[f"offset_error_px_{v.split('_')[1]}"] = {
                "median": statistics.median(errs),
                "mean": round(statistics.mean(errs), 1),
                "max": max(errs),
                "pct_within_2px": round(sum(e <= 2 for e in errs) / len(errs), 3)}
        if cell != "zero":                      # subpixel robustness split
            sub = {}
            for tag, want in (("integer", False), ("fractional", True)):
                ss = [i for i in sel if pairs[i]["fractional"] == want]
                pos = [results[i]["aligned_masked"]["score"] for i in ss
                       if pairs[i]["changed"]]
                neg = [results[i]["aligned_masked"]["score"] for i in ss
                       if not pairs[i]["changed"]]
                sub[tag] = {"n": len(ss), "auroc": round(auroc(pos, neg), 4),
                            "best_threshold_accuracy":
                                round(best_acc(pos, neg), 4),
                            "unchanged_score_max":
                                round(max(neg), 2) if neg else None}
            crec["aligned_masked_by_subpixel"] = sub
        for v in ("naive_masked", "aligned_masked"):
            hits, tot = 0, 0
            for i in sel:
                if not pairs[i]["changed"]:
                    continue
                tot += 1
                tx, ty = results[i][v]["top_tile"]
                if in_box(tx, ty, pairs[i]["bbox"]):
                    hits += 1
            crec[f"localization_{v}"] = {"hits": f"{hits}/{tot}",
                                         "rate": round(hits / tot, 3)}
        rep["per_cell"][cell] = crec

    for v in variants:
        pos = [results[i][v]["score"] for i, r in enumerate(pairs) if r["changed"]]
        neg = [results[i][v]["score"] for i, r in enumerate(pairs) if not r["changed"]]
        ms = [results[i][v]["ms"] for i in range(len(pairs))]
        rep["overall"][v] = {
            "auroc": round(auroc(pos, neg), 4),
            "best_threshold_accuracy": round(best_acc(pos, neg), 4),
            "median_ms_per_pair": round(statistics.median(ms), 1)}

    # failure listing: aligned_masked misrankings + big offset errors
    thr_fail = []
    for i, r in enumerate(pairs):
        e = abs(results[i]["aligned_masked"]["dy_hat"] - r["dy_true"])
        if e > 8:
            thr_fail.append({"pair": r["pair_id"], "dy_true": r["dy_true"],
                             "dy_hat": results[i]["aligned_masked"]["dy_hat"],
                             "err": e})
    rep["alignment_failures_gt8px"] = thr_fail
    return rep


# ============================================================================

def main():
    for d in (OUTDIR, HTML_DIR, PNG_DIR, VERIFY_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("== calibration ==", flush=True)
    cal = calibrate()
    for t, f in cal.items():
        ys = sorted(v[0] for v in f.values())
        print(f"  {t}: {len(f)} fields, content y {ys[0]}..{ys[-1]}, "
              f"max gap {max(b - a for a, b in zip(ys, ys[1:]))}px")

    print("== generating pairs ==", flush=True)
    if MANIFEST.exists():
        pairs = json.loads(MANIFEST.read_text())
        print(f"  reusing manifest ({len(pairs)} pairs)")
    else:
        pairs = make_pairs(cal)
        MANIFEST.write_text(json.dumps(pairs, indent=1))
        print(f"  {len(pairs)} pairs rendered")

    # bbox verification crops (3 changed samples across templates)
    picked = []
    seen_t = set()
    for r in pairs:
        if r["changed"] and r["template"] not in seen_t:
            picked.append(r); seen_t.add(r["template"])
        if len(picked) == 3:
            break
    for r in picked:
        x1, y1, x2, y2 = r["bbox"]
        im = Image.open(r["new_png"]).crop(
            (max(0, x1 - 40), max(0, y1 - 25), min(W, x2 + 40), min(H, y2 + 25)))
        im.save(VERIFY_DIR / f"{r['pair_id']}_bboxcrop.png")
        dy = int(round(r["dy_true"]))
        Image.open(r["old_png"]).crop(
            (max(0, x1 - 40), max(0, y1 - 25 + dy),
             min(W, x2 + 40), min(H, y2 + 25 + dy))).save(
            VERIFY_DIR / f"{r['pair_id']}_bboxcrop_oldaligned.png")
        print(f"  verify crop: {r['pair_id']} field={r['mutated_field']} "
              f"{r.get('old_value')} -> {r.get('new_value')} bbox={r['bbox']}")

    print("== detection ==", flush=True)
    results = []
    for i, r in enumerate(pairs):
        results.append(run_detectors(r))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(pairs)}", flush=True)

    PER_PAIR.write_text(json.dumps(
        [{**{k: v for k, v in p.items() if not k.endswith("_png")}, "det": d}
         for p, d in zip(pairs, results)], indent=1))

    rep = summarize(pairs, results)
    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep["overall"], indent=2))
    print("wrote", RESULT_JSON)


if __name__ == "__main__":
    main()
