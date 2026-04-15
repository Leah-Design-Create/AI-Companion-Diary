# -*- coding: utf-8 -*-
"""
树洞日记 AI 基准测试脚本

用法：
  python scripts/run_benchmark.py --email xxx@xx.com --password yourpass
  python scripts/run_benchmark.py --token <token>
"""

import argparse
import json
import sys
import os
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ───────────────────────────── 认证 ─────────────────────────────

def get_token(base_url, email, password):
    r = requests.post(f"{base_url}/api/auth/login",
                      json={"email": email, "password": password}, timeout=15)
    r.raise_for_status()
    return r.json()["token"]

# ───────────────────────────── 评分（同步包装） ─────────────────────────────

def run_eval(user_message, ai_reply):
    try:
        import asyncio
        from services.evaluate import evaluate_response
        return asyncio.run(evaluate_response(user_message, ai_reply))
    except Exception as e:
        print(f"    [Eval 失败] {e}")
        return None

# ───────────────────────────── 单条测试 ─────────────────────────────

def run_case(base_url, token, case):
    headers = {"Authorization": f"Bearer {token}"}

    # 1. 拿 AI 回复
    try:
        r = requests.post(
            f"{base_url}/api/chat",
            json={"message": case["user_message"]},
            headers=headers,
            timeout=60,
        )
        r.raise_for_status()
        ai_reply = r.json().get("reply", "")
    except Exception as e:
        print(f"    [请求失败] {e}")
        return {**case, "ai_reply": "", "scores": None,
                "violations": [], "floor_failures": [], "passed": False}

    # 2. 评分
    scores = run_eval(case["user_message"], ai_reply)

    # 3. 禁用词检查
    violations = [p for p in case.get("must_not_contain", []) if p in ai_reply]

    # 4. 分数下限检查
    floor_failures = []
    if scores:
        for dim, floor in case.get("dimensions_floor", {}).items():
            actual = scores.get(dim) or 0
            if actual < floor:
                floor_failures.append(f"{dim} {actual:.1f}<{floor}")

    passed = not violations and not floor_failures
    return {
        "id": case["id"],
        "category": case["category"],
        "user_message": case["user_message"],
        "ai_reply": ai_reply,
        "scores": scores,
        "violations": violations,
        "floor_failures": floor_failures,
        "passed": passed,
    }

# ───────────────────────────── 报告 ─────────────────────────────

def generate_report(results, results_dir, ts):
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    overalls = [r["scores"]["overall"] for r in results if r.get("scores")]
    avg = sum(overalls) / len(overalls) if overalls else 0

    prev_results = None
    prev_ts = ""
    prev_files = sorted(results_dir.glob("*.json"))
    prev_files = [f for f in prev_files if f.stem != ts]
    if prev_files:
        prev_ts = prev_files[-1].stem
        try:
            prev_results = json.loads(prev_files[-1].read_text(encoding="utf-8"))
        except Exception:
            pass

    lines = [
        "# 基准测试报告",
        "",
        f"**时间：** {ts.replace('_', ' ')}",
        "",
        "## 总体结果",
        "",
    ]

    prev_pass_rate = None
    prev_avg = None
    if prev_results:
        prev_pass = sum(1 for r in prev_results if r["passed"])
        prev_overalls = [r["scores"]["overall"] for r in prev_results if r.get("scores")]
        prev_avg = sum(prev_overalls) / len(prev_overalls) if prev_overalls else 0
        prev_pass_rate = prev_pass / len(prev_results) * 100

    def diff(cur, prev, fmt=".1f"):
        if prev is None:
            return ""
        d = cur - prev
        return f" ({'+'if d>0 else ''}{d:{fmt}})"

    pass_rate = passed / total * 100
    lines += [
        f"- 通过率：**{passed}/{total}（{pass_rate:.0f}%）**{diff(pass_rate, prev_pass_rate, '.0f')}",
        f"- 平均综合分：**{avg:.2f}/5**{diff(avg, prev_avg)}",
        "",
    ]

    # 分类汇总
    cats = {}
    for r in results:
        c = r["category"]
        cats.setdefault(c, {"pass": 0, "total": 0})
        cats[c]["total"] += 1
        if r["passed"]:
            cats[c]["pass"] += 1
    lines += ["## 分类汇总", "", "| 分类 | 通过 | 总数 |", "|------|------|------|"]
    for cat, s in cats.items():
        lines.append(f"| {cat} | {s['pass']} | {s['total']} |")

    # 各维度
    dims = {"empathy": "共情", "naturalness": "自然", "helpfulness": "有用", "safety": "安全"}
    lines += ["", "## 各维度均分", "", "| 维度 | 均分 |", "|------|------|"]
    for dim, label in dims.items():
        vals = [r["scores"][dim] for r in results if r.get("scores") and r["scores"].get(dim)]
        if vals:
            lines.append(f"| {label} | {sum(vals)/len(vals):.2f} |")

    # 失败详情
    failed = [r for r in results if not r["passed"]]
    if failed:
        lines += ["", "## 失败用例", ""]
        for r in failed:
            lines.append(f"### ❌ `{r['id']}` — {r['category']}")
            lines.append(f"\n**用户：** {r['user_message']}\n")
            lines.append(f"**AI：** {(r.get('ai_reply') or '')[:200]}\n")
            if r["violations"]:
                lines.append(f"**禁用词：** {', '.join(r['violations'])}")
            if r["floor_failures"]:
                lines.append(f"**分数不足：** {', '.join(r['floor_failures'])}")
            if r.get("scores"):
                s = r["scores"]
                lines.append(f"**评分：** 共情{s.get('empathy')} 自然{s.get('naturalness')} 有用{s.get('helpfulness')} 安全{s.get('safety')} 综合{s.get('overall')}")
                if s.get("comment"):
                    lines.append(f"**评语：** {s['comment']}")
            lines.append("")

    # 退步
    if prev_results:
        prev_map = {r["id"]: r for r in prev_results}
        reg = [r["id"] for r in results if prev_map.get(r["id"], {}).get("passed") and not r["passed"]]
        if reg:
            lines += ["", "## ⚠️ 退步项", ""]
            for rid in reg:
                lines.append(f"- `{rid}`")

    lines += ["", "---", "*由 `scripts/run_benchmark.py` 自动生成*"]
    return "\n".join(lines)

# ───────────────────────────── 主流程 ─────────────────────────────

def main(base_url, token):
    benchmark_path = ROOT / "benchmark.json"
    if not benchmark_path.exists():
        print("找不到 benchmark.json")
        sys.exit(1)

    cases = json.loads(benchmark_path.read_text(encoding="utf-8"))["cases"]
    print(f"共 {len(cases)} 条用例，开始运行...\n")

    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i:02d}/{len(cases)}] {case['id']}  ({case['category']})", end="", flush=True)
        result = run_case(base_url, token, case)
        status = "  OK" if result["passed"] else "  FAIL"
        score_str = f"  overall={result['scores']['overall']}" if result.get("scores") else ""
        print(f"{status}{score_str}")
        if result.get("violations"):
            print(f"       禁用词: {result['violations']}")
        if result.get("floor_failures"):
            print(f"       分数不足: {result['floor_failures']}")
        results.append(result)

    results_dir = ROOT / "benchmark_results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    (results_dir / f"{ts}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    report = generate_report(results, results_dir, ts)
    (ROOT / "benchmark_report.md").write_text(report, encoding="utf-8")

    passed = sum(1 for r in results if r["passed"])
    overalls = [r["scores"]["overall"] for r in results if r.get("scores")]
    print(f"\n{'='*40}")
    print(f"通过率: {passed}/{len(results)} ({passed/len(results)*100:.0f}%)")
    if overalls:
        print(f"平均综合分: {sum(overalls)/len(overalls):.2f} / 5")
    print(f"{'='*40}")
    print(f"报告: benchmark_report.md")


if __name__ == "__main__":
    print("树洞日记基准测试", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default="")
    parser.add_argument("--email", default="")
    parser.add_argument("--password", default="")
    args = parser.parse_args()

    print(f"服务地址: {args.url}", flush=True)

    token = args.token
    if not token:
        if not args.email or not args.password:
            print("用法: python scripts/run_benchmark.py --email 邮箱 --password 密码")
            sys.exit(1)
        print("登录中...", flush=True)
        try:
            token = get_token(args.url, args.email, args.password)
            print("登录成功", flush=True)
        except Exception as e:
            print(f"登录失败: {e}")
            sys.exit(1)

    main(args.url, token)
