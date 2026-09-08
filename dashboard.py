"""AGY Proxy Dashboard — real-time usage stats, request log, token analytics, account rotation UI."""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
import os

router = APIRouter()

# ---------------------------------------------------------------------------
# Usage tracker — in-memory, resets on restart
# ponytail: no persistence; add JSON-file dump when needed
# ---------------------------------------------------------------------------

MAX_LOG_ENTRIES = 200


@dataclass
class _ModelStats:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: int = 0
    last_used: float = 0.0


def _estimate_message_tokens(messages: list[dict]) -> list[dict]:
    """Rough per-message token estimates (~1.3 tokens per word)."""
    breakdown = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            text = " ".join(
                b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
            )
        elif content is None:
            text = ""
        else:
            text = str(content)
        words = len(text.split()) if text.strip() else 0
        est = max(1, int(words * 1.3))
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                est += len(fn.get("arguments", "").split()) + 5
        breakdown.append({
            "role": msg.get("role", "unknown"),
            "tokens": est,
            "preview": (text[:120] + "\u2026") if len(text) > 120 else text if text else "(empty)",
        })
    return breakdown


class UsageTracker:
    def __init__(self) -> None:
        self._start = time.time()
        self._models: dict[str, _ModelStats] = {}
        self._log: deque[dict] = deque(maxlen=MAX_LOG_ENTRIES)

    def record(
        self,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        success: bool = True,
        *,
        messages: list[dict] | None = None,
        duration_ms: float = 0,
        stream: bool = False,
    ) -> None:
        s = self._models.setdefault(model, _ModelStats())
        s.requests += 1
        s.prompt_tokens += prompt_tokens
        s.completion_tokens += completion_tokens
        if not success:
            s.errors += 1
        s.last_used = time.time()

        breakdown = _estimate_message_tokens(messages) if messages else []
        self._log.append({
            "id": uuid.uuid4().hex[:10],
            "ts": time.time(),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "duration_ms": round(duration_ms, 1),
            "stream": stream,
            "success": success,
            "messages": breakdown,
        })

    def stats(self) -> dict[str, Any]:
        now = time.time()
        models = {}
        total_req = total_tok = total_err = 0
        for name, s in self._models.items():
            total = s.prompt_tokens + s.completion_tokens
            models[name] = {
                "requests": s.requests,
                "prompt_tokens": s.prompt_tokens,
                "completion_tokens": s.completion_tokens,
                "total_tokens": total,
                "errors": s.errors,
                "last_used_ago": round(now - s.last_used, 1) if s.last_used else None,
            }
            total_req += s.requests
            total_tok += total
            total_err += s.errors
        return {
            "uptime_seconds": round(now - self._start, 1),
            "total_requests": total_req,
            "total_tokens": total_tok,
            "total_errors": total_err,
            "active_models": len(self._models),
            "models": models,
        }

    def logs(self, limit: int = 50) -> list[dict]:
        entries = list(self._log)
        entries.reverse()
        return entries[:limit]

    def get_log(self, log_id: str) -> dict | None:
        for entry in self._log:
            if entry["id"] == log_id:
                return entry
        return None

    def token_series(self, limit: int = 50) -> list[dict]:
        entries = list(self._log)[-limit:]
        return [
            {"ts": e["ts"], "prompt": e["prompt_tokens"], "completion": e["completion_tokens"], "model": e["model"]}
            for e in entries
        ]


tracker = UsageTracker()

# ---------------------------------------------------------------------------
# Stub account data
# ---------------------------------------------------------------------------

_STUB_ACCOUNTS: list[dict[str, Any]] = [
    {"id": "acc_demo_google_1", "name": "primary@gmail.com", "provider": "Google", "status": "active", "requests_today": 142, "added": "2026-09-01T10:00:00Z"},
    {"id": "acc_demo_google_2", "name": "secondary@gmail.com", "provider": "Google", "status": "active", "requests_today": 87, "added": "2026-09-03T14:30:00Z"},
    {"id": "acc_demo_anthropic", "name": "team@company.com", "provider": "Anthropic", "status": "rate-limited", "requests_today": 0, "added": "2026-09-05T09:15:00Z"},
]

# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@router.get("/api/dashboard/stats")
async def dashboard_stats():
    return tracker.stats()

@router.get("/api/dashboard/logs")
async def dashboard_logs(limit: int = 50):
    return {"logs": tracker.logs(min(limit, MAX_LOG_ENTRIES))}

@router.get("/api/dashboard/logs/{log_id}")
async def dashboard_log_detail(log_id: str):
    entry = tracker.get_log(log_id)
    if not entry:
        raise HTTPException(404, "Log entry not found")
    return entry

@router.get("/api/dashboard/chart")
async def dashboard_chart(limit: int = 50):
    return {"series": tracker.token_series(min(limit, 100))}

@router.get("/api/dashboard/accounts")
async def list_accounts():
    return {"accounts": _STUB_ACCOUNTS}

@router.post("/api/dashboard/accounts")
async def add_account(body: dict[str, Any] | None = None):
    body = body or {}
    new = {
        "id": f"acc_{uuid.uuid4().hex[:8]}",
        "name": body.get("name", "new@example.com"),
        "provider": body.get("provider", "Google"),
        "status": "active",
        "requests_today": 0,
        "added": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _STUB_ACCOUNTS.append(new)
    return {"ok": True, "account": new}

@router.post("/api/dashboard/accounts/{account_id}/toggle")
async def toggle_account(account_id: str):
    for acc in _STUB_ACCOUNTS:
        if acc["id"] == account_id:
            acc["status"] = "inactive" if acc["status"] == "active" else "active"
            return {"ok": True, "account": acc}
    raise HTTPException(404, "Account not found")

@router.delete("/api/dashboard/accounts/{account_id}")
async def remove_account(account_id: str):
    for i, acc in enumerate(_STUB_ACCOUNTS):
        if acc["id"] == account_id:
            _STUB_ACCOUNTS.pop(i)
            return {"ok": True}
    raise HTTPException(404, "Account not found")

@router.get("/logo.jpg")
async def get_logo():
    if os.path.exists("logo.jpg"):
        return FileResponse("logo.jpg")
    raise HTTPException(404, "Logo not found")

@router.get("/favicon.ico")
async def get_favicon():
    if os.path.exists("logo.jpg"):
        return FileResponse("logo.jpg")
    raise HTTPException(404, "Favicon not found")


# ---------------------------------------------------------------------------
# HTML Dashboard — redesigned
# ---------------------------------------------------------------------------

def get_dashboard_html() -> HTMLResponse:
    html = r"""<!DOCTYPE html>
<html lang="en" dir="ltr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AGY Proxy</title>
<link rel="icon" href="/favicon.ico">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#06060a;--bg2:#0c0c14;--surface:rgba(16,16,28,.65);--surface-solid:#10101c;
  --glass:rgba(20,20,36,.55);--glass-border:rgba(255,255,255,.06);
  --text:#f0f0f8;--text2:#6e6e8a;--text3:#44445a;
  --mint:#34d399;--mint-dim:rgba(52,211,153,.12);
  --violet:#8b5cf6;--violet-dim:rgba(139,92,246,.12);
  --rose:#fb7185;--rose-dim:rgba(251,113,133,.12);
  --sky:#38bdf8;--sky-dim:rgba(56,189,248,.12);
  --amber:#fbbf24;--amber-dim:rgba(251,191,36,.12);
  --border:rgba(255,255,255,.05);--border-hover:rgba(255,255,255,.1);
  --radius:14px;--radius-sm:8px;
  --font:'Inter',system-ui,-apple-system,sans-serif;
  --mono:'JetBrains Mono','Fira Code',monospace;
  --shadow:0 4px 24px rgba(0,0,0,.4),0 0 0 1px rgba(255,255,255,.03);
}
html{font-size:14px;-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale}
body{background:var(--bg);color:var(--text);font-family:var(--font);min-height:100vh;overflow-x:hidden}

/* Ambient background */
.ambient{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden}
.ambient .orb{position:absolute;border-radius:50%;filter:blur(120px);opacity:.15;animation:drift 20s ease-in-out infinite alternate}
.ambient .orb1{width:600px;height:600px;background:var(--violet);top:-200px;left:-100px}
.ambient .orb2{width:500px;height:500px;background:var(--mint);bottom:-200px;right:-100px;animation-delay:-10s;animation-duration:25s}
.ambient .orb3{width:400px;height:400px;background:var(--sky);top:40%;left:50%;animation-delay:-5s;animation-duration:22s}
@keyframes drift{
  0%{transform:translate(0,0) scale(1)}
  33%{transform:translate(40px,-30px) scale(1.05)}
  66%{transform:translate(-20px,40px) scale(.95)}
  100%{transform:translate(30px,20px) scale(1.02)}
}

/* Layout shell */
.shell{position:relative;z-index:1;display:flex;min-height:100vh}

/* Sidebar */
.sidebar{
  width:240px;position:fixed;top:0;left:0;bottom:0;
  background:var(--glass);backdrop-filter:blur(24px) saturate(1.4);
  border-right:1px solid var(--glass-border);
  display:flex;flex-direction:column;padding:24px 0;z-index:10;
}
.sidebar-logo{padding:0 24px 28px;display:flex;align-items:center;gap:10px}
.sidebar-logo img.logo-mark{
  width:32px;height:32px;border-radius:10px;object-fit:cover;
  box-shadow:0 0 12px rgba(139,92,246,.4);
}
.sidebar-logo .logo-text{font-weight:700;font-size:1.1rem;letter-spacing:-.03em}
.sidebar-logo .logo-text span{color:var(--text2);font-weight:400;font-size:.75rem;margin-left:6px}

.nav{flex:1;display:flex;flex-direction:column;gap:2px;padding:0 12px}
.nav-item{
  display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:var(--radius-sm);
  font-size:.85rem;font-weight:500;color:var(--text2);cursor:pointer;
  transition:all .2s ease;position:relative;
}
.nav-item:hover{color:var(--text);background:rgba(255,255,255,.04)}
.nav-item.active{color:var(--text);background:rgba(139,92,246,.1)}
.nav-item.active::before{
  content:'';position:absolute;left:0;top:50%;transform:translateY(-50%);
  width:3px;height:20px;background:var(--violet);border-radius:0 3px 3px 0;
}
.nav-item svg{width:18px;height:18px;opacity:.6;flex-shrink:0}
.nav-item.active svg{opacity:1}

.sidebar-footer{padding:16px 24px;border-top:1px solid var(--border)}
.status-pill{
  display:flex;align-items:center;gap:8px;font-size:.78rem;color:var(--text2);
}
.status-dot{width:7px;height:7px;border-radius:50%;background:var(--mint);box-shadow:0 0 8px var(--mint);animation:glow 2s ease infinite}
@keyframes glow{0%,100%{opacity:1;box-shadow:0 0 8px var(--mint)}50%{opacity:.5;box-shadow:0 0 4px var(--mint)}}

/* Main content */
.main{margin-left:240px;flex:1;padding:28px 32px;min-height:100vh}
.tab-panel{display:none}
.tab-panel.active{display:block;animation:fadeUp .3s ease}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}

/* Page header */
.page-header{margin-bottom:24px;display:flex;align-items:center;justify-content:space-between}
.page-title{font-size:1.5rem;font-weight:700;letter-spacing:-.04em}
.page-meta{display:flex;gap:14px;align-items:center;color:var(--text2);font-size:.8rem}
.page-meta .uptime{font-family:var(--mono);font-size:.75rem}

/* Stat cards */
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px}
.stat{
  background:var(--glass);backdrop-filter:blur(20px);
  border:1px solid var(--glass-border);border-radius:var(--radius);
  padding:20px;position:relative;overflow:hidden;
  transition:border-color .2s,transform .2s;
}
.stat:hover{border-color:var(--border-hover);transform:translateY(-1px)}
.stat .stat-icon{
  width:36px;height:36px;border-radius:10px;display:grid;place-items:center;
  margin-bottom:14px;
}
.stat .stat-icon svg{width:18px;height:18px}
.stat .stat-label{font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;color:var(--text2);font-weight:600;margin-bottom:6px}
.stat .stat-value{font-size:1.8rem;font-weight:800;letter-spacing:-.04em;font-family:var(--font);line-height:1}
.stat .stat-sub{font-size:.72rem;color:var(--text2);margin-top:6px}

/* Glass cards */
.card{
  background:var(--glass);backdrop-filter:blur(20px);
  border:1px solid var(--glass-border);border-radius:var(--radius);
  overflow:hidden;transition:border-color .2s;
}
.card:hover{border-color:var(--border-hover)}
.card-header{
  padding:16px 20px;display:flex;align-items:center;justify-content:space-between;
  border-bottom:1px solid var(--border);
}
.card-title{font-size:.88rem;font-weight:600;display:flex;align-items:center;gap:8px}
.card-title svg{width:16px;height:16px;opacity:.5}
.card-body{padding:0}
.card-body.padded{padding:20px}

/* Tables */
table{width:100%;border-collapse:collapse}
th{
  text-align:left;padding:10px 16px;font-size:.68rem;text-transform:uppercase;
  letter-spacing:.08em;color:var(--text3);font-weight:600;
  background:rgba(255,255,255,.02);border-bottom:1px solid var(--border);
}
td{padding:11px 16px;border-bottom:1px solid var(--border);font-size:.84rem}
tr:last-child td{border-bottom:none}
tr:hover td{background:rgba(255,255,255,.02)}
.mono{font-family:var(--mono);font-size:.78rem}

/* Badges */
.pill{
  display:inline-flex;align-items:center;gap:4px;
  padding:3px 10px;border-radius:20px;font-size:.68rem;font-weight:600;
  letter-spacing:.03em;text-transform:uppercase;
}
.pill-mint{background:var(--mint-dim);color:var(--mint)}
.pill-rose{background:var(--rose-dim);color:var(--rose)}
.pill-violet{background:var(--violet-dim);color:var(--violet)}
.pill-sky{background:var(--sky-dim);color:var(--sky)}
.pill-amber{background:var(--amber-dim);color:var(--amber)}
.pill-muted{background:rgba(255,255,255,.05);color:var(--text2)}

/* Buttons */
.btn{
  padding:7px 14px;border-radius:var(--radius-sm);
  border:1px solid var(--border);background:rgba(255,255,255,.04);
  color:var(--text);font-size:.8rem;cursor:pointer;
  font-family:var(--font);font-weight:500;
  transition:all .15s ease;display:inline-flex;align-items:center;gap:6px;
}
.btn:hover{background:rgba(255,255,255,.08);border-color:var(--border-hover)}
.btn-primary{background:var(--violet);border-color:var(--violet);color:#fff}
.btn-primary:hover{background:#7c3aed;border-color:#7c3aed}
.btn-ghost{border-color:transparent;background:transparent}
.btn-ghost:hover{background:rgba(255,255,255,.05)}
.btn-danger{color:var(--rose)}
.btn-danger:hover{background:var(--rose-dim)}
.btn-sm{padding:5px 10px;font-size:.75rem}

/* Usage bars */
.bar-track{height:6px;background:rgba(255,255,255,.04);border-radius:3px;overflow:hidden}
.bar-fill{height:100%;border-radius:3px;transition:width .6s cubic-bezier(.4,0,.2,1)}

/* Token bar (breakdown) */
.tok-bar{height:8px;background:rgba(255,255,255,.04);border-radius:4px;overflow:hidden;flex:1}
.tok-fill{height:100%;border-radius:4px}

/* Chart */
.chart-wrap{padding:20px}
.chart-legend{display:flex;gap:16px;font-size:.72rem;color:var(--text2)}
.chart-legend i{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:5px;vertical-align:middle}
canvas{display:block;width:100%;cursor:crosshair}

/* Log rows */
tr.log-row{cursor:pointer}
tr.log-row:hover td{background:rgba(139,92,246,.04)}
.detail-row{display:none}
.detail-row.open{display:table-row}
.detail-inner{padding:14px 20px;background:rgba(0,0,0,.2);border-radius:var(--radius-sm)}

/* Message breakdown */
.msg-row{
  display:flex;align-items:center;gap:12px;padding:7px 0;
  border-bottom:1px solid rgba(255,255,255,.04);font-size:.82rem;
}
.msg-row:last-child{border-bottom:none}
.msg-role{font-weight:600;min-width:72px;font-size:.68rem;text-transform:uppercase;letter-spacing:.05em}
.msg-tokens{min-width:68px;font-family:var(--mono);font-size:.78rem;text-align:right}
.msg-pct{min-width:40px;font-size:.68rem;color:var(--text2);text-align:right}
.msg-preview{color:var(--text2);font-size:.76rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;min-width:0}

/* Forms */
.form-row{display:flex;gap:12px;align-items:end;flex-wrap:wrap}
.form-group label{display:block;font-size:.68rem;color:var(--text2);margin-bottom:4px;text-transform:uppercase;letter-spacing:.06em;font-weight:600}
.form-input{
  padding:9px 14px;background:rgba(255,255,255,.04);border:1px solid var(--border);
  border-radius:var(--radius-sm);color:var(--text);font-size:.85rem;font-family:var(--font);
  transition:border-color .15s;min-width:200px;
}
.form-input:focus{outline:none;border-color:rgba(139,92,246,.4)}
select.form-input{appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' fill='%236e6e8a' viewBox='0 0 16 16'%3E%3Cpath d='M8 11L3 6h10z'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 12px center;padding-right:32px}

/* Toggle switch */
.toggle{position:relative;display:inline-block;width:40px;height:22px}
.toggle input{opacity:0;width:0;height:0}
.toggle .track{
  position:absolute;cursor:pointer;inset:0;background:rgba(255,255,255,.08);
  border-radius:22px;transition:.2s;
}
.toggle .track::before{
  content:'';position:absolute;height:16px;width:16px;left:3px;bottom:3px;
  background:var(--text2);border-radius:50%;transition:.2s;
}
.toggle input:checked+.track{background:var(--violet)}
.toggle input:checked+.track::before{transform:translateX(18px);background:#fff}

/* Rotation settings */
.settings-grid{display:grid;gap:14px}
.setting-row{display:flex;align-items:center;gap:16px}
.setting-row label{font-size:.84rem;color:var(--text2);min-width:150px}

/* Collapsible form */
.add-form{
  display:none;padding:20px;border-bottom:1px solid var(--border);
  background:rgba(0,0,0,.15);
}
.add-form.open{display:block;animation:fadeUp .2s ease}

/* Toast */
.toast{
  position:fixed;bottom:28px;right:28px;
  background:var(--glass);backdrop-filter:blur(20px);
  border:1px solid var(--glass-border);border-radius:var(--radius);
  padding:12px 20px;font-size:.84rem;color:var(--text);
  transform:translateY(100px);opacity:0;transition:all .3s cubic-bezier(.4,0,.2,1);
  z-index:200;box-shadow:var(--shadow);
}
.toast.show{transform:translateY(0);opacity:1}

/* Empty state */
.empty-state{text-align:center;padding:48px 20px;color:var(--text2)}
.empty-state svg{width:40px;height:40px;opacity:.2;margin-bottom:12px}
.empty-state p{font-size:.88rem}

/* Scrollable */
.scroll-area{max-height:540px;overflow-y:auto}
.scroll-area::-webkit-scrollbar{width:5px}
.scroll-area::-webkit-scrollbar-track{background:transparent}
.scroll-area::-webkit-scrollbar-thumb{background:rgba(255,255,255,.08);border-radius:3px}

/* Responsive */
@media(max-width:900px){
  .sidebar{display:none}
  .main{margin-left:0}
  .stats{grid-template-columns:repeat(2,1fr)}
}
</style>
</head>
<body>

<!-- Ambient background orbs -->
<div class="ambient">
  <div class="orb orb1"></div>
  <div class="orb orb2"></div>
  <div class="orb orb3"></div>
</div>

<div class="shell">

<!-- ─── Sidebar ─── -->
<aside class="sidebar">
  <div class="sidebar-logo">
    <img src="/logo.jpg" class="logo-mark" alt="AGY Proxy">
    <div class="logo-text">AGY Proxy <span>v2.0</span></div>
  </div>
  <nav class="nav">
    <div class="nav-item active" onclick="switchTab('overview')" data-tab="overview">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
      Overview
    </div>
    <div class="nav-item" onclick="switchTab('logs')" data-tab="logs">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
      Request Log
    </div>
    <div class="nav-item" onclick="switchTab('analytics')" data-tab="analytics">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
      Token Analytics
    </div>
    <div class="nav-item" onclick="switchTab('accounts')" data-tab="accounts">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 00-3-3.87"/><path d="M16 3.13a4 4 0 010 7.75"/></svg>
      Accounts
    </div>
  </nav>
  <div class="sidebar-footer">
    <div class="status-pill">
      <span class="status-dot"></span>
      <span>Proxy running</span>
      <span style="margin-left:auto;font-family:var(--mono);font-size:.7rem" id="sb-uptime">—</span>
    </div>
  </div>
</aside>

<!-- ─── Main Content ─── -->
<main class="main">

<!-- ==================== OVERVIEW ==================== -->
<div id="tab-overview" class="tab-panel active">
  <div class="page-header">
    <h1 class="page-title">Overview</h1>
    <div class="page-meta">
      <span class="uptime" id="hdr-uptime">—</span>
    </div>
  </div>

  <div class="stats">
    <div class="stat">
      <div class="stat-icon" style="background:var(--violet-dim)">
        <svg viewBox="0 0 24 24" fill="none" stroke="var(--violet)" stroke-width="2" stroke-linecap="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
      </div>
      <div class="stat-label">Total Requests</div>
      <div class="stat-value" id="s-requests">0</div>
      <div class="stat-sub" id="s-req-rate">—</div>
    </div>
    <div class="stat">
      <div class="stat-icon" style="background:var(--mint-dim)">
        <svg viewBox="0 0 24 24" fill="none" stroke="var(--mint)" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
      </div>
      <div class="stat-label">Total Tokens</div>
      <div class="stat-value" id="s-tokens">0</div>
      <div class="stat-sub">prompt + completion</div>
    </div>
    <div class="stat">
      <div class="stat-icon" style="background:var(--sky-dim)">
        <svg viewBox="0 0 24 24" fill="none" stroke="var(--sky)" stroke-width="2" stroke-linecap="round"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"/><path d="M16 21V5a2 2 0 00-2-2h-4a2 2 0 00-2 2v16"/></svg>
      </div>
      <div class="stat-label">Active Models</div>
      <div class="stat-value" id="s-models">0</div>
      <div class="stat-sub">models used</div>
    </div>
    <div class="stat">
      <div class="stat-icon" style="background:var(--rose-dim)">
        <svg viewBox="0 0 24 24" fill="none" stroke="var(--rose)" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>
      </div>
      <div class="stat-label">Error Rate</div>
      <div class="stat-value" id="s-errors">0%</div>
      <div class="stat-sub" id="s-err-count">0 errors</div>
    </div>
  </div>

  <!-- Token Flow Chart -->
  <div class="card" style="margin-bottom:24px">
    <div class="card-header">
      <div class="card-title">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>
        Token Flow
      </div>
      <div class="chart-legend">
        <span><i style="background:var(--violet)"></i> Prompt</span>
        <span><i style="background:var(--mint)"></i> Completion</span>
      </div>
    </div>
    <div class="chart-wrap">
      <canvas id="tokenChart" height="180"></canvas>
    </div>
  </div>

  <!-- Model Usage -->
  <div class="card">
    <div class="card-header">
      <div class="card-title">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
        Model Usage
      </div>
    </div>
    <div class="card-body">
      <table>
        <thead><tr><th>Model</th><th>Requests</th><th>Prompt</th><th>Completion</th><th>Errors</th><th>Last Used</th><th>Volume</th></tr></thead>
        <tbody id="model-tbody">
          <tr><td colspan="7"><div class="empty-state"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><path d="M8 12h8M12 8v8"/></svg><p>No requests yet — send a completion to see stats</p></div></td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>

<!-- ==================== LOGS ==================== -->
<div id="tab-logs" class="tab-panel">
  <div class="page-header">
    <h1 class="page-title">Request Log</h1>
    <div class="page-meta"><span id="log-count" class="mono">—</span></div>
  </div>
  <div class="card">
    <div class="card-body">
      <div class="scroll-area">
        <table>
          <thead><tr><th style="width:24px"></th><th>Time</th><th>Model</th><th>Prompt →</th><th>← Completion</th><th>Total</th><th>Duration</th><th>Type</th><th>Status</th></tr></thead>
          <tbody id="log-tbody">
            <tr><td colspan="9"><div class="empty-state"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg><p>No requests yet</p></div></td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<!-- ==================== ANALYTICS ==================== -->
<div id="tab-analytics" class="tab-panel">
  <div class="page-header">
    <h1 class="page-title">Token Analytics</h1>
  </div>

  <div class="card" style="margin-bottom:24px">
    <div class="card-header">
      <div class="card-title">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>
        Token Usage Over Time
      </div>
      <div class="chart-legend">
        <span><i style="background:var(--violet)"></i> Prompt</span>
        <span><i style="background:var(--mint)"></i> Completion</span>
      </div>
    </div>
    <div class="chart-wrap">
      <canvas id="tokenChartLarge" height="260"></canvas>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
    <div class="card">
      <div class="card-header"><div class="card-title">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        Message Breakdown
      </div></div>
      <div class="card-body padded" id="breakdown-panel">
        <div class="empty-state" style="padding:24px"><p>Click a request in the log to inspect</p></div>
      </div>
    </div>
    <div class="card">
      <div class="card-header"><div class="card-title">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 002 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0022 16z"/></svg>
        Token Distribution
      </div></div>
      <div class="card-body padded" id="model-dist">
        <div class="empty-state" style="padding:24px"><p>No data yet</p></div>
      </div>
    </div>
  </div>
</div>

<!-- ==================== ACCOUNTS ==================== -->
<div id="tab-accounts" class="tab-panel">
  <div class="page-header">
    <h1 class="page-title">Account Rotation</h1>
    <button class="btn btn-sm" onclick="toggleAddForm()">
      <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
      Add Account
    </button>
  </div>

  <div class="card" style="margin-bottom:20px">
    <div class="add-form" id="add-form">
      <div class="form-row">
        <div class="form-group"><label>Email / Name</label><input class="form-input" type="text" id="inp-name" placeholder="user@gmail.com"></div>
        <div class="form-group"><label>Provider</label><select class="form-input" id="inp-provider"><option>Google</option><option>Anthropic</option><option>OpenAI</option></select></div>
        <div class="form-group" style="display:flex;gap:6px;align-items:end;padding-bottom:1px">
          <button class="btn btn-primary" onclick="addAccount()">Add</button>
          <button class="btn" onclick="toggleAddForm()">Cancel</button>
        </div>
      </div>
    </div>
    <div class="card-body">
      <table>
        <thead><tr><th>Account</th><th>Provider</th><th>Status</th><th>Requests Today</th><th>Added</th><th>Actions</th></tr></thead>
        <tbody id="accounts-tbody"><tr><td colspan="6"><div class="empty-state" style="padding:24px"><p>Loading…</p></div></td></tr></tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <div class="card-header"><div class="card-title">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"/></svg>
      Rotation Strategy
    </div></div>
    <div class="card-body padded">
      <div class="settings-grid">
        <div class="setting-row">
          <label>Strategy</label>
          <select class="form-input" style="min-width:160px"><option>Round Robin</option><option>Least Used</option><option>Random</option><option>Failover Only</option></select>
        </div>
        <div class="setting-row">
          <label>Rate Limit Cooldown</label>
          <input class="form-input" type="number" value="60" min="0" style="width:80px;min-width:80px">
          <span style="font-size:.8rem;color:var(--text2)">seconds</span>
        </div>
        <div class="setting-row">
          <label>Auto-rotate on error</label>
          <label class="toggle"><input type="checkbox" checked><span class="track"></span></label>
        </div>
        <div><button class="btn btn-primary btn-sm" onclick="toast('Settings saved (stub)')">Save Settings</button></div>
      </div>
    </div>
  </div>
</div>

</main>
</div><!-- /shell -->

<div class="toast" id="toast"></div>

<script>
// ─── Helpers ───
const $=s=>document.getElementById(s);
function fmt(n){if(n>=1e6)return(n/1e6).toFixed(1)+'M';if(n>=1e3)return(n/1e3).toFixed(1)+'K';return String(n)}
function ago(s){if(!s&&s!==0)return'—';if(s<60)return Math.round(s)+'s ago';if(s<3600)return Math.round(s/60)+'m ago';return Math.round(s/3600)+'h ago'}
function timeStr(ts){return new Date(ts*1000).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'})}
function toast(msg){const t=$('toast');t.textContent=msg;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),2500)}
function uptime(s){const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),sec=Math.floor(s%60);return(h?h+'h ':'')+(m?m+'m ':'')+sec+'s'}
function escHtml(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}

// ─── Tabs ───
function switchTab(name){
  document.querySelectorAll('.nav-item').forEach(n=>{n.classList.toggle('active',n.dataset.tab===name)});
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  const p=$('tab-'+name);if(p)p.classList.add('active');
  if(name==='logs')refreshLogs();
  if(name==='analytics'){refreshChart(true);refreshModelDist()}
}

// ─── Stats ───
async function refreshStats(){
  try{
    const r=await fetch('/api/dashboard/stats');const d=await r.json();
    $('hdr-uptime').textContent='Uptime: '+uptime(d.uptime_seconds);
    $('sb-uptime').textContent=uptime(d.uptime_seconds);
    $('s-requests').textContent=fmt(d.total_requests);
    $('s-tokens').textContent=fmt(d.total_tokens);
    $('s-models').textContent=d.active_models;
    const errRate=d.total_requests?((d.total_errors/d.total_requests)*100).toFixed(1)+'%':'0%';
    $('s-errors').textContent=errRate;
    $('s-err-count').textContent=d.total_errors+' error'+(d.total_errors!==1?'s':'');
    if(d.uptime_seconds>0&&d.total_requests>0)$('s-req-rate').textContent=(d.total_requests/(d.uptime_seconds/60)).toFixed(1)+' req/min';
    // Model table
    const tbody=$('model-tbody');const models=Object.entries(d.models||{});
    if(!models.length){tbody.innerHTML='<tr><td colspan="7"><div class="empty-state"><svg viewBox="0 0 24 24" width="40" height="40" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><path d="M8 12h8M12 8v8"/></svg><p>No requests yet</p></div></td></tr>';return}
    const maxReq=Math.max(...models.map(([,v])=>v.requests),1);
    const colors=['var(--violet)','var(--sky)','var(--mint)','var(--amber)','var(--rose)'];
    tbody.innerHTML=models.sort((a,b)=>b[1].requests-a[1].requests).map(([name,m],i)=>{
      const pct=Math.round((m.requests/maxReq)*100);const c=colors[i%5];
      return'<tr><td class="mono" style="font-weight:500">'+name+'</td><td>'+fmt(m.requests)+'</td><td style="color:var(--violet)">'+fmt(m.prompt_tokens)+'</td><td style="color:var(--mint)">'+fmt(m.completion_tokens)+'</td><td>'+(m.errors?'<span style="color:var(--rose)">'+m.errors+'</span>':'0')+'</td><td style="color:var(--text2)">'+ago(m.last_used_ago)+'</td><td><div class="bar-track" style="min-width:80px"><div class="bar-fill" style="width:'+pct+'%;background:'+c+'"></div></div></td></tr>';
    }).join('');
    window._statsCache=d;
  }catch(e){console.error('Stats failed',e)}
}

// ─── Logs ───
let _logCache=[];
async function refreshLogs(){
  try{
    const r=await fetch('/api/dashboard/logs?limit=100');const d=await r.json();
    _logCache=d.logs||[];
    const tbody=$('log-tbody');$('log-count').textContent=_logCache.length+' request'+(_logCache.length!==1?'s':'');
    if(!_logCache.length){tbody.innerHTML='<tr><td colspan="9"><div class="empty-state"><svg viewBox="0 0 24 24" width="40" height="40" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg><p>No requests yet</p></div></td></tr>';return}
    let html='';
    for(const e of _logCache){
      const sCls=e.success?'pill-mint':'pill-rose';const sTxt=e.success?'OK':'Error';
      const tCls=e.stream?'pill-sky':'pill-violet';const tTxt=e.stream?'Stream':'Sync';
      html+='<tr class="log-row" onclick="toggleLogDetail(\''+e.id+'\')"><td style="color:var(--text3);font-size:.7rem">›</td><td class="mono" style="color:var(--text2)">'+timeStr(e.ts)+'</td><td class="mono">'+e.model+'</td><td style="color:var(--violet)">'+fmt(e.prompt_tokens)+'</td><td style="color:var(--mint)">'+fmt(e.completion_tokens)+'</td><td style="font-weight:600">'+fmt(e.total_tokens)+'</td><td class="mono" style="color:var(--text2)">'+(e.duration_ms?Math.round(e.duration_ms)+'ms':'—')+'</td><td><span class="pill '+tCls+'">'+tTxt+'</span></td><td><span class="pill '+sCls+'">'+sTxt+'</span></td></tr>';
      html+='<tr class="detail-row" id="detail-'+e.id+'"><td colspan="9"><div class="detail-inner">'+buildBreakdown(e,true)+'</div></td></tr>';
    }
    tbody.innerHTML=html;
  }catch(e){console.error('Log failed',e)}
}

function toggleLogDetail(id){
  const row=$('detail-'+id);if(!row)return;
  const open=row.classList.toggle('open');
  if(open){const entry=_logCache.find(e=>e.id===id);if(entry)$('breakdown-panel').innerHTML=buildBreakdown(entry,true)}
}

function buildBreakdown(entry,full){
  if(!entry.messages||!entry.messages.length)return'<div style="color:var(--text2);font-size:.82rem">No message breakdown</div>';
  const totalTok=entry.messages.reduce((s,m)=>s+m.tokens,0)||1;
  const roleColors={system:'var(--amber)',user:'var(--sky)',assistant:'var(--violet)',tool:'var(--mint)'};
  let html='<div style="font-size:.68rem;color:var(--text3);margin-bottom:10px;text-transform:uppercase;letter-spacing:.06em;font-weight:600">'+entry.messages.length+' messages · ~'+fmt(totalTok)+' prompt tokens</div>';
  for(const m of entry.messages){
    const pct=Math.round((m.tokens/totalTok)*100);const c=roleColors[m.role]||'var(--text2)';
    html+='<div class="msg-row"><div class="msg-role" style="color:'+c+'">'+m.role+'</div><div class="msg-tokens">'+fmt(m.tokens)+'</div><div class="msg-pct">'+pct+'%</div><div class="tok-bar"><div class="tok-fill" style="width:'+pct+'%;background:'+c+';opacity:.5"></div></div>'+(full?'<div class="msg-preview" title="'+escHtml(m.preview)+'">'+escHtml(m.preview)+'</div>':'')+'</div>';
  }
  html+='<div class="msg-row" style="border-top:1px solid rgba(255,255,255,.06);padding-top:10px;margin-top:6px"><div class="msg-role" style="color:var(--mint)">output</div><div class="msg-tokens" style="color:var(--mint)">'+fmt(entry.completion_tokens)+'</div><div class="msg-pct"></div><div class="tok-bar"><div class="tok-fill" style="width:100%;background:var(--mint);opacity:.3"></div></div>'+(full?'<div class="msg-preview" style="color:var(--mint)">(model response)</div>':'')+'</div>';
  return html;
}

// ─── Chart ───
let _chartData=[];
function drawChart(canvasId,data,h){
  const canvas=$(canvasId);if(!canvas||!data.length)return;
  const ctx=canvas.getContext('2d');const dpr=window.devicePixelRatio||1;
  const rect=canvas.getBoundingClientRect();canvas.width=rect.width*dpr;canvas.height=h*dpr;ctx.scale(dpr,dpr);
  const W=rect.width,H=h;ctx.clearRect(0,0,W,H);
  const maxTok=Math.max(...data.map(d=>Math.max(d.prompt,d.completion)),1);
  const startX=44,chartW=W-startX-10;
  // Grid
  ctx.font='10px "JetBrains Mono",monospace';ctx.fillStyle='#44445a';
  for(let i=0;i<=4;i++){const y=H-24-(i/4)*(H-34);ctx.strokeStyle='rgba(255,255,255,.03)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(startX,y);ctx.lineTo(W-10,y);ctx.stroke();ctx.fillText(fmt(Math.round(maxTok*i/4)),2,y+3)}
  // Bars
  const bw=Math.max(4,Math.min(20,(chartW)/data.length-3));
  const gap=(chartW-data.length*(bw+2))/(data.length+1);
  data.forEach((d,i)=>{
    const x=startX+gap+(bw+2+gap)*i;const ph=(d.prompt/maxTok)*(H-38);const ch=(d.completion/maxTok)*(H-38);
    // Prompt
    ctx.fillStyle='rgba(139,92,246,.6)';rRect(ctx,x,H-24-ph,bw/2-1,ph,2);ctx.fill();
    // Completion
    ctx.fillStyle='rgba(52,211,153,.6)';rRect(ctx,x+bw/2+1,H-24-ch,bw/2-1,ch,2);ctx.fill();
  });
  ctx.strokeStyle='rgba(255,255,255,.05)';ctx.beginPath();ctx.moveTo(startX,H-24);ctx.lineTo(W-10,H-24);ctx.stroke();
}
function rRect(ctx,x,y,w,h,r){if(h<1)return;ctx.beginPath();ctx.moveTo(x+r,y);ctx.lineTo(x+w-r,y);ctx.quadraticCurveTo(x+w,y,x+w,y+r);ctx.lineTo(x+w,y+h);ctx.lineTo(x,y+h);ctx.lineTo(x,y+r);ctx.quadraticCurveTo(x,y,x+r,y);ctx.closePath()}

async function refreshChart(large){
  try{const r=await fetch('/api/dashboard/chart?limit=50');const d=await r.json();_chartData=d.series||[];drawChart('tokenChart',_chartData,180);if(large)drawChart('tokenChartLarge',_chartData,260)}catch(e){}
}

// ─── Model Distribution ───
function refreshModelDist(){
  const d=window._statsCache;if(!d||!d.models)return;
  const panel=$('model-dist');const models=Object.entries(d.models);
  if(!models.length){panel.innerHTML='<div class="empty-state" style="padding:24px"><p>No data</p></div>';return}
  const totalTok=models.reduce((s,[,m])=>s+m.total_tokens,0)||1;
  let html='';
  models.sort((a,b)=>b[1].total_tokens-a[1].total_tokens).forEach(([name,m])=>{
    const pct=Math.round((m.total_tokens/totalTok)*100);const pp=m.total_tokens?Math.round((m.prompt_tokens/m.total_tokens)*100):0;
    html+='<div style="margin-bottom:16px"><div style="display:flex;justify-content:space-between;margin-bottom:5px"><span class="mono" style="font-size:.78rem">'+name+'</span><span style="font-size:.72rem;color:var(--text2)">'+fmt(m.total_tokens)+' · '+pct+'%</span></div><div style="height:10px;background:rgba(255,255,255,.04);border-radius:5px;overflow:hidden;display:flex"><div style="width:'+pp+'%;background:var(--violet);opacity:.6;transition:width .4s"></div><div style="width:'+(100-pp)+'%;background:var(--mint);opacity:.5;transition:width .4s"></div></div><div style="display:flex;justify-content:space-between;margin-top:3px;font-size:.68rem;color:var(--text3)"><span>Prompt: '+fmt(m.prompt_tokens)+'</span><span>Completion: '+fmt(m.completion_tokens)+'</span></div></div>';
  });
  panel.innerHTML=html;
}

// ─── Accounts ───
async function refreshAccounts(){
  try{
    const r=await fetch('/api/dashboard/accounts');const d=await r.json();
    const tbody=$('accounts-tbody');
    if(!d.accounts||!d.accounts.length){tbody.innerHTML='<tr><td colspan="6"><div class="empty-state" style="padding:24px"><p>No accounts</p></div></td></tr>';return}
    tbody.innerHTML=d.accounts.map(a=>{
      const cls=a.status==='active'?'pill-mint':a.status==='rate-limited'?'pill-rose':'pill-muted';
      const lbl=a.status==='active'?'Pause':'Activate';
      const date=a.added?new Date(a.added).toLocaleDateString():'—';
      return'<tr><td style="font-weight:500">'+a.name+'</td><td>'+a.provider+'</td><td><span class="pill '+cls+'">'+a.status+'</span></td><td>'+fmt(a.requests_today||0)+'</td><td style="color:var(--text2)">'+date+'</td><td style="display:flex;gap:6px"><button class="btn btn-sm" onclick="toggleAccount(\''+a.id+'\')">'+lbl+'</button><button class="btn btn-sm btn-ghost btn-danger" onclick="removeAccount(\''+a.id+'\')">Remove</button></td></tr>';
    }).join('');
  }catch(e){console.error('Accounts failed',e)}
}

function toggleAddForm(){$('add-form').classList.toggle('open')}
async function addAccount(){
  const name=$('inp-name').value.trim();const provider=$('inp-provider').value;
  if(!name){toast('Enter an account name');return}
  try{await fetch('/api/dashboard/accounts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,provider})});$('inp-name').value='';toggleAddForm();toast('Account added');refreshAccounts()}catch(e){toast('Failed')}
}
async function toggleAccount(id){try{await fetch('/api/dashboard/accounts/'+id+'/toggle',{method:'POST'});toast('Toggled');refreshAccounts()}catch(e){toast('Failed')}}
async function removeAccount(id){try{await fetch('/api/dashboard/accounts/'+id,{method:'DELETE'});toast('Removed');refreshAccounts()}catch(e){toast('Failed')}}

// ─── Init ───
refreshStats();refreshAccounts();refreshChart();
setInterval(refreshStats,5000);
setInterval(()=>{refreshChart();if($('tab-logs').classList.contains('active'))refreshLogs()},5000);
setInterval(refreshAccounts,15000);
window.addEventListener('resize',()=>{drawChart('tokenChart',_chartData,180);drawChart('tokenChartLarge',_chartData,260)});
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


@router.get("/", response_class=HTMLResponse)
async def dashboard_page():
    return get_dashboard_html()
