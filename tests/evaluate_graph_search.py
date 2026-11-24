"""Offline-оценка GraphRAG на вручную размеченной части корпуса."""

import argparse
import json
from pathlib import Path

import networkx as nx

from assistant.graphrag.communities import detect_communities
from assistant.infra.cli import Context
from assistant.infra.cli import main as assistant_main
from assistant.infra.config import Settings
from assistant.infra.grounding import load_vocabulary
from assistant.infra.indexing import save_kb
from assistant.infra.llm import LLMResponse, LLMStats
from assistant.infra.schemas import MAP_SCHEMA, KnowledgeBase

FACTS = (
    ("ErrorCode:E-17", "FIXED_BY", "Procedure:SP-104", "procedure_SP-104"),
    ("ErrorCode:E-17", "INDICATES", "Component:пусковое реле", "manual_AR-210"),
    ("Model:AR-210", "HAS_PART", "Part:P-1042", "manual_AR-210"),
    ("Part:P-1042", "IS_A", "Component:пусковое реле", "manual_AR-210"),
    ("Part:P-1042", "REPLACED_BY", "Part:P-1042B", "bulletin_SB-2025-03"),
    ("Model:GS-520", "HAS_PART", "Part:P-3126", "manual_GS-520"),
    ("Part:P-3126", "IS_A", "Component:реле давления", "manual_GS-520"),
    ("Part:P-3126", "REPLACED_BY", "Part:P-3126B", "bulletin_SB-2025-07"),
    ("Part:P-3126B", "SUPPLIED_BY", "Supplier:АвтоматикаСевер", "bulletin_SB-2025-07"),
    ("Model:BZ-110", "HAS_PART", "Part:P-2054", "manual_BZ-110"),
    ("Part:P-2054", "IS_A", "Component:нагреватель оттайки", "manual_BZ-110"),
    ("Part:P-2054", "REPLACED_BY", "Part:P-2054B", "bulletin_SB-2025-04"),
    ("Part:P-2054B", "REPLACED_BY", "Part:P-2054C", "bulletin_SB-2025-08"),
    ("Model:AR-320", "FAILED", "Component:пусковое реле", "report_R-2025-003"),
    ("Model:GS-420", "FAILED", "Component:циркуляционный насос", "report_R-2025-007"),
)

LOCAL_IDS = {"dev-01", "dev-06", "dev-08", "dev-09", "dev-22", "dev-23", "dev-24"}
GLOBAL_CASES = {
    "dev-13": ("AR-320", "Арктика", "пусковое реле", "report_R-2025-003.md"),
    "dev-14": ("GS-420", "Гольфстрим", "циркуляционный насос", "report_R-2025-007.md"),
}
GLOBAL_IDS = {*GLOBAL_CASES, "dev-22"}


def fixture_kb(data_dir: Path) -> KnowledgeBase:
    graph = nx.DiGraph()
    for subject, relation, target, document in FACTS:
        source = f"{document}.md#0"
        if not (data_dir / "corpus" / f"{document}.md").is_file():
            raise FileNotFoundError(document)
        for node in (subject, target):
            if node not in graph:
                kind, name = node.split(":", 1)
                graph.add_node(node, type=kind, name=name, sources=[])
            if source not in graph.nodes[node]["sources"]:
                graph.nodes[node]["sources"].append(source)
        graph.add_edge(subject, target, type=relation, weight=1, sources=[source])
    communities = detect_communities(graph)
    for community in communities:
        summaries = []
        for model, line, component, document in GLOBAL_CASES.values():
            if f"Model:{model}" in community.nodes:
                summaries.append(f"Линейка «{line}»: в документе {document} один отказ узла «{component}».")
        community.summary = " ".join(summaries) or "В сообществе нет отчётов об отказах."
    return KnowledgeBase(graph, communities, load_vocabulary(data_dir / "vocabulary.json"))


class EvidenceModel:
    """Подставная модель проверяет наличие эталонных фактов в контексте."""

    def __init__(self, questions: list[dict], mode: str) -> None:
        self.questions = questions
        self.mode = mode
        self.stats = LLMStats()

    def chat(self, messages, *, tools=None, json_schema=None, max_tokens=1024):
        self.stats.calls += 1
        prompt = "\n".join(message["content"] for message in messages if message["role"] == "user")
        question = next(row for row in self.questions if row["question"] in prompt)
        context = prompt.split(question["question"], 1)[1]
        refusal = "В базе знаний не найдено."
        if self.mode == "local":
            facts = {line.partition(". ")[2] for line in context.splitlines()}
            present = all(
                f"{subject.split(':', 1)[1]} -{relation}-> {target.split(':', 1)[1]}" in facts
                for subject, relation, target in question["evidence"]
            )
            return LLMResponse(
                ", ".join(spellings[0] for spellings in question["key_facts"])
                if present and not question["must_refuse"]
                else refusal
            )
        if question["must_refuse"]:
            return LLMResponse('{"answer": "", "score": 0}' if json_schema == MAP_SCHEMA else refusal)
        _, line, component, document = GLOBAL_CASES[question["id"]]
        found = all(term in context for term in (line, component, document))
        if json_schema == MAP_SCHEMA:
            partial = {"answer": f"{line}: {component}, {document}." if found else "", "score": 8 if found else 0}
            return LLMResponse(json.dumps(partial, ensure_ascii=False))
        return LLMResponse(component if found else refusal)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    data_dir = Path(__file__).resolve().parents[1] / "data"
    settings = Settings(data_dir=data_dir, artifacts_dir=args.out_dir)
    save_kb(fixture_kb(data_dir), settings)
    questions = [
        json.loads(line)
        for line in (data_dir / "questions_dev.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for mode, ids in (("local", LOCAL_IDS), ("global", GLOBAL_IDS)):
        cases = [row for row in questions if row["id"] in ids]
        if mode == "global":
            cases = [
                {**row, "source_groups": [[GLOBAL_CASES[row["id"]][-1]]]} if row["id"] in GLOBAL_CASES else row
                for row in cases
            ]
        subset = args.out_dir / f"questions_{mode}.jsonl"
        subset.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cases), encoding="utf-8")
        assistant_main(
            ["eval", "--mode", mode, "--questions", str(subset)],
            context=Context(settings, EvidenceModel(cases, mode), None),
        )


if __name__ == "__main__":
    main()
