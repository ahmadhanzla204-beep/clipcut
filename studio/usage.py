"""A record of what the AI was used for: one row per call, with the tokens it took.

Nothing is charged for and nothing is limited here. The AI key belongs to whoever runs this server, and their AI
service bills them directly; this is only so they (and each user) can see where the tokens went."""
import threading
import time
from collections import defaultdict

import sqlalchemy as sa

from . import ai, config, ctx, db

_where = threading.local()
_brief: dict = {}    # user -> (when, summary). Dropped the moment they make another AI call.


def set_context(creator=None, where=None, job_id=None):
    """Which creator and video the AI calls on this thread are for, until changed."""
    _where.creator, _where.where, _where.job_id = creator, where, job_id


def record(tokens_in: int, tokens_out: int, purpose: str, model: str = "", cost=None):
    """One AI call. `cost` is what the service itself reported (US dollars); when it did not, a published-price
    estimate is stored and marked as an estimate, and when even that is unknown the cost stays empty."""
    uid = ctx.uid_or_none()
    if not uid:
        return
    est = 0
    if cost is None:
        cost = ai.estimate_cost(model, int(tokens_in), int(tokens_out))
        est = 1 if cost is not None else 0
    with db.engine.begin() as c:
        c.execute(db.usage.insert().values(user_id=uid, t=time.time(), tokens_in=max(0, int(tokens_in)),
                                           tokens_out=max(0, int(tokens_out)), purpose=purpose[:60],
                                           creator=getattr(_where, "creator", None),
                                           place=(getattr(_where, "where", None) or "")[:200],
                                           model=(model or "")[:120], cost=cost, cost_est=est,
                                           job_id=(getattr(_where, "job_id", None) or None)))
    _brief.pop(uid, None)
    ctx.bump(uid)


def forget(uid: str):
    _brief.pop(uid, None)


def forget_all():
    """The AI settings changed: every page should say so on its next look."""
    _brief.clear()


def _day(t):
    return time.strftime("%Y-%m-%d", time.localtime(t))


def _tokens():
    return db.usage.c.tokens_in + db.usage.c.tokens_out


def brief() -> dict:
    """The few numbers every page shows. Counted at most every ten seconds per user."""
    uid, now = ctx.uid(), time.time()
    got = _brief.get(uid)
    if got and now - got[0] < 10:
        return dict(got[1])
    midnight = time.mktime(time.localtime(now)[:3] + (0, 0, 0, 0, 0, -1))
    u = db.usage.c
    with db.engine.connect() as c:
        today = c.execute(sa.select(sa.func.count(), sa.func.coalesce(sa.func.sum(_tokens()), 0)).where(
            u.user_id == uid, u.t >= midnight)).first()
    out = {"ready": ai.ready(), "provider": ai.label(), "model": (ai.models() or [""])[0],
           "today_calls": int(today[0]), "today_tokens": int(today[1])}
    _brief[uid] = (now, out)
    return dict(out)


def cost_total(uid: str) -> dict:
    u = db.usage.c
    with db.engine.connect() as c:
        s, est = c.execute(sa.select(sa.func.sum(u.cost), sa.func.max(u.cost_est)).where(u.user_id == uid)).first()
    return {"cost": float(s) if s is not None else None, "est": bool(est)}


def by_video(uid: str, limit: int = 300) -> list:
    """Every video's AI calls, newest video first. A call belongs to a video by its job id, or (older records
    and live streams, which have none) by the title it was made under."""
    u = db.usage.c
    with db.engine.connect() as c:
        rows = list(c.execute(sa.select(db.usage).where(u.user_id == uid).order_by(u.t.desc()).limit(5000)))
    groups: dict = {}
    for r in rows:
        t = r._mapping["t"]               # (a row's own .t is something else)
        key = r.job_id or ("title:" + (r.place or ""))
        g = groups.setdefault(key, {"job_id": r.job_id or "", "title": r.place or "Untitled", "t": t, "tokens_in": 0, "tokens_out": 0,
                                    "cost": None, "est": False, "calls": []})
        g["tokens_in"] += int(r.tokens_in)
        g["tokens_out"] += int(r.tokens_out)
        if r.cost is not None:
            g["cost"] = (g["cost"] or 0.0) + float(r.cost)
            g["est"] = g["est"] or bool(r.cost_est)
        g["calls"].append({"t": t, "purpose": r.purpose, "model": r.model or "", "tokens_in": int(r.tokens_in),
                           "tokens_out": int(r.tokens_out), "cost": r.cost, "est": bool(r.cost_est)})
    out = sorted(groups.values(), key=lambda g: -g["t"])[:limit]
    for g in out:
        g["calls"].reverse()                 # oldest call first inside a video
    return out


def delete_video(job_id: str = "", title: str = "") -> int:
    """Remove one video's usage records (the video and its clips are not touched). Returns how many rows went."""
    uid, u = ctx.uid(), db.usage.c
    if job_id:
        cond = (u.user_id == uid) & (u.job_id == job_id)
    else:
        cond = (u.user_id == uid) & u.job_id.is_(None) & (u.place == (title or "")[:200])
    with db.engine.begin() as c:
        n = c.execute(db.usage.delete().where(cond)).rowcount
    forget(uid)
    ctx.bump(uid)
    return int(n or 0)


def summary(creator_names: dict) -> dict:
    """Everything the AI usage page shows, for the signed-in user."""
    uid, now = ctx.uid(), time.time()
    u = db.usage.c
    with db.engine.connect() as c:
        totals = c.execute(sa.select(sa.func.count(), sa.func.coalesce(sa.func.sum(_tokens()), 0)).where(u.user_id == uid)).first()
        by_purpose = [[r[0], int(r[1] or 0), r[2]] for r in c.execute(
            sa.select(u.purpose, sa.func.sum(_tokens()), sa.func.count()).where(u.user_id == uid)
            .group_by(u.purpose).order_by(sa.func.sum(_tokens()).desc()))]
        by_creator = defaultdict(lambda: [0, 0])
        for r in c.execute(sa.select(u.creator, sa.func.sum(_tokens()), sa.func.count()).where(u.user_id == uid).group_by(u.creator)):
            key = creator_names.get(r[0], "No creator") if r[0] else "No creator"
            by_creator[key][0] += int(r[1] or 0)
            by_creator[key][1] += r[2]
        days = [_day(now - 86400 * i) for i in range(29, -1, -1)]
        daily = {d: [0, 0] for d in days}
        for r in c.execute(sa.select(u.t, u.tokens_in, u.tokens_out).where(u.user_id == uid, u.t >= now - 31 * 86400)):
            if _day(r[0]) in daily:
                daily[_day(r[0])][0] += int(r[1]) + int(r[2])
                daily[_day(r[0])][1] += 1
        recent = [{"t": r._mapping["t"], "tokens": int(r.tokens_in) + int(r.tokens_out), "purpose": r.purpose, "where": r.place,
                   "creator_name": creator_names.get(r.creator, "") if r.creator else ""}
                  for r in c.execute(sa.select(db.usage).where(u.user_id == uid).order_by(u.t.desc()).limit(60))]
    return {**brief(), "models": {"scan": ai.models(), "final": ai.models("final")},
            "videos": by_video(uid), "cost_total": cost_total(uid),
            "calls_total": int(totals[0]), "tokens_total": int(totals[1]),
            "daily": [{"day": d, "tokens": daily[d][0], "calls": daily[d][1]} for d in days],
            "by_purpose": by_purpose,
            "by_creator": sorted(([k, v[0], v[1]] for k, v in by_creator.items()), key=lambda x: -x[1]),
            "recent": recent}
