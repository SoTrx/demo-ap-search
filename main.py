"""
Today's AP search and the new one, side by side, over the 605 APs of the analytical
pattern embedder's benchmark. Each search is a real moma-management with its own Neo4j;
the questions and their expected APs are the benchmark's test set.
"""

import json
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import getenv
from pathlib import Path

import httpx
import streamlit as st

ASSETS = Path(__file__).parent / "assets"
SEARCHES = {
    "Today's search": {
        "url": getenv("MOMA_OLD_URL", "http://moma-old:5000"),
        "about": "moma-management v2.13.0 · MiniLM on the AP's own description",
        # The field of aps.jsonl this search embeds an AP from.
        "indexed": "root_description",
    },
    "New search": {
        "url": getenv("MOMA_NEW_URL", "http://moma-new:5000"),
        "about": "moma-management v2.14.0 · fine-tuned EmbeddingGemma 2 on the description built from the operators",
        "indexed": "description",
    },
}
TOP_K = 10
# Most results moma-management returns for one search.
MAX_RESULTS = 100
# moma-management's default threshold, which ap-management's searches get today.
DEFAULT_THRESHOLD = 0.75
SETS = {
    "real": "Real questions",
    "seen": "Generated, APs seen in training (new wording)",
    "unseen": "Generated, operators never seen in training",
}


@st.cache_data
def load_assets() -> tuple[dict[str, dict], list[dict]]:
    aps = {ap["id"]: ap for ap in map(json.loads, (ASSETS / "aps.jsonl").read_text().splitlines())}
    tasks = [json.loads(line) for line in (ASSETS / "tasks.jsonl").read_text().splitlines()]
    return aps, tasks


def root_id(graph: dict) -> str:
    return next(n["id"] for n in graph["nodes"] if "Analytical_Pattern" in n["labels"])


def search(url: str, query: str, threshold: float) -> list[str]:
    """The ids of the APs `url` finds for `query`, best first."""
    response = httpx.get(f"{url}/api/v1/aps/", timeout=120, params={
        "search.q": query, "search.top_k": MAX_RESULTS,
        "search.threshold": threshold, "pageSize": MAX_RESULTS})
    response.raise_for_status()
    return [root_id(graph) for graph in response.json()["aps"]]


def seed(name: str, url: str, expected: int) -> None:
    """Load the APs into `url`'s moma-management, skipping those already there."""
    total = httpx.get(f"{url}/api/v1/aps/", params={"pageSize": 1}, timeout=30).json()["total"]
    if total >= expected:
        return

    def add(path: Path) -> None:
        graph = json.loads(path.read_text())
        if httpx.get(f"{url}/api/v1/aps/{root_id(graph)}", timeout=30).status_code == 404:
            httpx.post(f"{url}/api/v1/aps/", json=graph, timeout=300).raise_for_status()

    files = sorted((ASSETS / "aps").glob("*.json"))
    bar = st.progress(0.0, text=f"{name}: loading {len(files)} APs (embedding each one)…")
    # Alone first: on an empty database, concurrent first POSTs race to create the
    # vector index, and the loser fails after storing its AP without an embedding.
    add(files[0])
    with ThreadPoolExecutor(4) as pool:
        for done, future in enumerate(as_completed(pool.submit(add, f) for f in files[1:]), 2):
            future.result()
            bar.progress(done / len(files), text=f"{name}: {done}/{len(files)} APs loaded")
    bar.empty()


def score(ranking: list[str], task: dict) -> dict[str, float]:
    """The benchmark's metrics (benchmark.py `score`) for one ranking: the task's own AP
    is worth 2, each of its building blocks 1, and its supersets are left out."""
    supersets, blocks = set(task["supersets"]), set(task["building_blocks"])
    gains = [2 if id_ == task["ap"] else 1 if id_ in blocks else 0
             for id_ in ranking if id_ not in supersets][:TOP_K]
    ideal = sorted([2] + [1] * len(blocks), reverse=True)[:TOP_K]

    def dcg(values: list[int]) -> float:
        return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(values))

    return {"recall@1": float(gains[:1] == [2]), "recall@10": float(2 in gains),
            "nDCG@10": dcg(gains) / dcg(ideal), "no result": float(not ranking)}


# The example of benchmark.py's check_score: c is left out, so d, a, b are worth 0, 2, 1.
_check = score(["c", "d", "a", "b"], {"ap": "a", "building_blocks": ["b"], "supersets": ["c"]})
assert (_check["recall@1"], _check["recall@10"]) == (0, 1)
assert abs(_check["nDCG@10"] - (3 / math.log2(3) + 1 / math.log2(4)) / (3 + 1 / math.log2(3))) < 1e-9


def rank_of(ranking: list[str], task: dict) -> int | None:
    """Where `ranking` puts the task's expected AP, from 1; None when it is not there."""
    return ranking.index(task["ap"]) + 1 if task["ap"] in ranking else None


def show_results(name: str, ranking: list[str], kept: set[str] | None, task: dict | None) -> None:
    search_ = SEARCHES[name]
    st.subheader(name)
    st.caption(search_["about"])
    if task is not None:
        rank = rank_of(ranking, task)
        st.markdown(f"Expected AP ranked **#{rank}**" if rank else f"Expected AP **not in the first {MAX_RESULTS}**")
    if kept is not None:
        st.markdown(f"Threshold {DEFAULT_THRESHOLD} keeps **{len(kept)}** of these results.")
    for rank, id_ in enumerate(ranking[:TOP_K], 1):
        ap = aps[id_]
        badges = []
        if task is not None and id_ == task["ap"]:
            badges.append(":green-badge[expected AP]")
        elif task is not None and id_ in task["building_blocks"]:
            badges.append(":green-badge[:material/extension: building block]")
        elif task is not None and id_ in task["supersets"]:
            badges.append(":gray-badge[does more than asked]")
        if ap["real"]:
            badges.append(":blue-badge[real AP]")
        if kept is not None and id_ not in kept:
            badges.append(f":red-badge[cut by threshold {DEFAULT_THRESHOLD}]")
        with st.container(border=True):
            st.markdown(f"**{rank}. {ap['name']}** " + " ".join(badges))
            st.caption(" → ".join(op.removesuffix("_Operator").replace("_", " ") for op in ap["operators"]))
            with st.expander("Text this search indexed"):
                st.write(ap[search_["indexed"]])


st.set_page_config(page_title="AP search", layout="wide")
st.title("AP search: today's and the new one")

aps, tasks = load_assets()
for name, search_ in SEARCHES.items():
    seed(name, search_["url"], len(aps))

tab_search, tab_benchmark = st.tabs(["🔎 Search", "📊 Benchmark"])

with tab_search:
    source = st.radio("Question", ["From the test set", "My own"], horizontal=True)
    task = None
    if source == "From the test set":
        set_ = st.selectbox("Set", list(SETS), format_func=SETS.get)
        in_set = [t for t in tasks if t["set"] == set_]
        task = in_set[st.selectbox("Question", range(len(in_set)), format_func=lambda i: in_set[i]["task"])]
        query = task["task"]
        st.markdown(f"Expected AP: :green-badge[{aps[task['ap']]['name']}]")
    else:
        query = st.text_input("Question", "I want to find the highest recorded temperature in Zurich in 2020")
    show_cut = st.checkbox(
        f"Mark what moma-management's default threshold ({DEFAULT_THRESHOLD}) cuts: what ap-management gets today")

    if query:
        columns = st.columns(len(SEARCHES))
        for column, (name, search_) in zip(columns, SEARCHES.items()):
            with column:
                ranking = search(search_["url"], query, 0)
                kept = set(search(search_["url"], query, DEFAULT_THRESHOLD)) if show_cut else None
                show_results(name, ranking, kept, task)

with tab_benchmark:
    st.markdown(
        "Runs test questions through both searches and scores them as the embedder's benchmark "
        "does, among all 605 APs. With the threshold, a search can return nothing at all.")
    set_ = st.selectbox("Set", list(SETS), format_func=SETS.get, key="benchmark_set")
    to_run = [t for t in tasks if t["set"] == set_]
    count = st.slider("Questions", 1, len(to_run), min(len(to_run), 100))
    if st.button("Run", type="primary"):
        chosen = random.Random(13).sample(to_run, count)
        runs = [(name, threshold) for name in SEARCHES for threshold in (0, DEFAULT_THRESHOLD)]
        bar = st.progress(0.0, text="Searching…")
        rankings: dict[tuple, list[str]] = {}
        with ThreadPoolExecutor(4) as pool:
            futures = {pool.submit(search, SEARCHES[name]["url"], task["task"], threshold): (name, threshold, i)
                       for name, threshold in runs for i, task in enumerate(chosen)}
            for done, future in enumerate(as_completed(futures), 1):
                rankings[futures[future]] = future.result()
                bar.progress(done / len(futures), text=f"{done}/{len(futures)} searches")
        bar.empty()
        st.session_state["benchmark"] = (chosen, runs, rankings)

    if "benchmark" in st.session_state:
        chosen, runs, rankings = st.session_state["benchmark"]
        rows = []
        for name, threshold in runs:
            scores = [score(rankings[name, threshold, i], task) for i, task in enumerate(chosen)]
            rows.append({"Search": name, "Threshold": threshold,
                         **{metric: round(sum(s[metric] for s in scores) / len(scores), 3) for metric in scores[0]}})
        st.markdown(f"**{len(chosen)} questions** ({SETS[chosen[0]['set']]})")
        st.dataframe(rows, hide_index=True)

        st.markdown(f"Per question, threshold 0: where each search ranks the expected AP (empty: not in the first {MAX_RESULTS})")
        old, new = SEARCHES
        st.dataframe([
            {"Question": task["task"], "Expected AP": aps[task["ap"]]["name"],
             old: rank_of(rankings[old, 0, i], task), new: rank_of(rankings[new, 0, i], task)}
            for i, task in enumerate(chosen)], hide_index=True)
