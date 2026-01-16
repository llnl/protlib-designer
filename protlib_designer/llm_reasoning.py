import json
import os
import re
import statistics
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import click
import pandas as pd

from protlib_designer import logger
from protlib_designer.structure.contact_graph import compute_contact_edges

warnings.filterwarnings("ignore")


CONTACT_GRAPH_EMPTY_SENTINEL = "NO_CONTACTS"

DEFAULT_LLM_OUTPUT: Dict[str, Any] = {
    "schema_version": "1.0",
    "summary": "",
    "avoid_combinations": [],
    "suggested_mutations": [],
    "derived_scoring_function": {"objective": "minimize", "terms": []},
    "constraints": [],
    "contact_insights": [],
    "warnings": [],
}

SCHEMA_TEMPLATE: Dict[str, Any] = {
    "schema_version": "1.0",
    "summary": "string",
    "avoid_combinations": [
        {
            "mutations": ["H35Y", "L50W"],
            "severity": "hard|soft",
            "reason": "string",
        }
    ],
    "suggested_mutations": [
        {
            "mutation": "H52F",
            "reason": "string",
            "confidence": 0.0,
            "is_novel": False,
        }
    ],
    "derived_scoring_function": {
        "objective": "minimize",
        "terms": [
            {
                "term_type": "per_mutation|per_pair|global",
                "mutations": ["H52F"],
                "feature": "interface_contact_bonus",
                "weight": -0.2,
                "reason": "string",
            }
        ],
    },
    "constraints": [
        {
            "type": "forbid_mutation|forbid_combination|require_mutation|limit_count",
            "mutations": ["H35Y"],
            "limit": None,
            "reason": "string",
        }
    ],
    "contact_insights": [{"residue": "H35", "note": "string", "evidence": "string"}],
    "warnings": ["string"],
}


@dataclass(frozen=True)
class LLMReasoningConfig:
    model: str = "gpt-4o"
    temperature: float = 0.2
    max_tokens: int = 1500
    max_mutations_in_prompt: int = 50
    max_contact_edges_in_prompt: int = 200


def _chain_from_residue(residue: str) -> str:
    """Extract the chain ID from a residue string like 'H35' or 'A102B'."""
    if not residue:
        return ""
    match = re.match(r"([A-Za-z])", residue)
    return match[1] if match else ""


def _summarize_contacts(contact_edges: Sequence[Dict[str, float]]) -> Dict[str, Any]:
    """Summarize counts and distance statistics for contact edges."""
    if not contact_edges:
        return {
            "num_edges": 0,
            "ab_residue_counts": {},
            "ag_residue_counts": {},
            "ab_chain_counts": {},
            "ag_chain_counts": {},
            "min_distance": None,
            "max_distance": None,
            "mean_distance": None,
        }

    ab_counts = Counter(edge["ab_res"] for edge in contact_edges)
    ag_counts = Counter(edge["ag_res"] for edge in contact_edges)
    ab_chain_counts = Counter(
        _chain_from_residue(edge["ab_res"]) for edge in contact_edges
    )
    ag_chain_counts = Counter(
        _chain_from_residue(edge["ag_res"]) for edge in contact_edges
    )
    distances = [float(edge["distance"]) for edge in contact_edges]
    return {
        "num_edges": len(contact_edges),
        "ab_residue_counts": dict(ab_counts),
        "ag_residue_counts": dict(ag_counts),
        "ab_chain_counts": dict(ab_chain_counts),
        "ag_chain_counts": dict(ag_chain_counts),
        "min_distance": min(distances),
        "max_distance": max(distances),
        "mean_distance": statistics.mean(distances),
    }


def build_contact_graph_text(
    contact_edges: Sequence[Dict[str, float]],
    max_edges: int = 200,
) -> str:
    """Render contact edges to a compact text summary for LLM prompting."""
    if not contact_edges:
        return CONTACT_GRAPH_EMPTY_SENTINEL

    summary = _summarize_contacts(contact_edges)
    lines = ["Contact graph summary:"]
    lines.append(f"- num_edges: {summary['num_edges']}")
    lines.append(f"- ab_chain_counts: {summary['ab_chain_counts']}")
    lines.append(f"- ag_chain_counts: {summary['ag_chain_counts']}")
    lines.append(
        f"- distance_stats: min={summary['min_distance']:.2f}, "
        f"mean={summary['mean_distance']:.2f}, "
        f"max={summary['max_distance']:.2f}"
    )

    def _top_counts(counter: Dict[str, int], label: str, limit: int = 10) -> None:
        items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]
        formatted = ", ".join(f"{res}:{count}" for res, count in items)
        lines.append(f"- top_{label}: {formatted}" if formatted else f"- top_{label}:")

    _top_counts(summary["ab_residue_counts"], "ab_residues")
    _top_counts(summary["ag_residue_counts"], "ag_residues")

    lines.append("Contact edges (ab_res -> ag_res [distance]):")
    for edge in list(contact_edges)[:max_edges]:
        lines.append(
            f"{edge['ab_res']} -> {edge['ag_res']} [{float(edge['distance']):.2f}]"
        )
    if len(contact_edges) > max_edges:
        lines.append(f"... truncated {len(contact_edges) - max_edges} edges")

    return "\n".join(lines)


def build_contact_graph_text_from_pdb(
    pdb_file: str,
    heavy_chain_id: str,
    light_chain_id: str,
    antigen_chain_id: str,
    distance_threshold: float = 7.0,
    max_edges: int = 200,
) -> str:
    """Compute and render contact graph text from a PDB file."""
    edges = compute_contact_edges(
        pdb_file,
        heavy_chain_id,
        light_chain_id,
        antigen_chain_id,
        distance_threshold=distance_threshold,
    )
    return build_contact_graph_text(edges, max_edges=max_edges)


def _scores_from_dataframe(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if "Mutation" not in df.columns:
        raise ValueError("Scores CSV must include a 'Mutation' column.")
    score_columns = [col for col in df.columns if col != "Mutation"]
    scores_by_mutation: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        mutation = str(row["Mutation"])
        scores: Dict[str, float] = {}
        for column in score_columns:
            value = row[column]
            scores[column] = float("nan") if pd.isna(value) else float(value)
        scores_by_mutation[mutation] = scores
    return scores_by_mutation


def _select_mutations_for_prompt(
    scores_by_mutation: Dict[str, Dict[str, float]],
    mutation_proposals: Sequence[str],
    max_mutations: int,
) -> List[str]:
    if mutation_proposals:
        ordered = [m for m in mutation_proposals if m in scores_by_mutation]
        ordered.extend([m for m in mutation_proposals if m not in scores_by_mutation])
        return list(ordered)[:max_mutations]
    return sorted(scores_by_mutation.keys())[:max_mutations]


def _format_score_stats(scores_by_mutation: Dict[str, Dict[str, float]]) -> str:
    if not scores_by_mutation:
        return "No scores provided."
    score_columns = sorted(
        {col for scores in scores_by_mutation.values() for col in scores.keys()}
    )
    lines = []
    for column in score_columns:
        if values := [
            scores[column]
            for scores in scores_by_mutation.values()
            if column in scores and scores[column] == scores[column]
        ]:
            lines.append(
                f"- {column}: min={min(values):.4f}, "
                f"mean={statistics.mean(values):.4f}, "
                f"median={statistics.median(values):.4f}, "
                f"max={max(values):.4f}"
            )
    return "\n".join(lines) if lines else "No valid score values found."


def _format_scores_for_prompt(
    scores_by_mutation: Dict[str, Dict[str, float]],
    mutation_proposals: Sequence[str],
    max_mutations: int,
) -> str:
    if not scores_by_mutation and not mutation_proposals:
        return "No scores or mutation proposals provided."

    selected_mutations = _select_mutations_for_prompt(
        scores_by_mutation, mutation_proposals, max_mutations
    )
    score_stats = _format_score_stats(scores_by_mutation)

    lines = ["Score summary:", score_stats, "Sample scored mutations:"]
    if not selected_mutations:
        lines.append("- None")
    for mutation in selected_mutations:
        scores = scores_by_mutation.get(mutation, {})
        if scores:
            formatted_scores = ", ".join(
                f"{name}={value:.4f}" for name, value in scores.items()
            )
        else:
            formatted_scores = "no scores"
        lines.append(f"- {mutation}: {formatted_scores}")

    if mutation_proposals:
        if missing := [
            m for m in mutation_proposals if m not in scores_by_mutation
        ]:
            lines.append("Mutation proposals without scores:")
            lines.extend(f"- {mutation}" for mutation in missing)
    return "\n".join(lines)


def _build_messages(
    contact_graph_text: str,
    scores_by_mutation: Dict[str, Dict[str, float]],
    mutation_proposals: Sequence[str],
    config: LLMReasoningConfig,
) -> List[Dict[str, str]]:
    if not contact_graph_text.strip():
        contact_graph_text = CONTACT_GRAPH_EMPTY_SENTINEL

    scores_text = _format_scores_for_prompt(
        scores_by_mutation, mutation_proposals, config.max_mutations_in_prompt
    )
    proposals_text = (
        "\n".join(f"- {mutation}" for mutation in mutation_proposals)
        if mutation_proposals
        else "None provided."
    )

    schema_text = json.dumps(SCHEMA_TEMPLATE, indent=2)

    # The schema is embedded to enforce deterministic keys for downstream parsing.
    # We also cap the number of scored mutations to keep prompt size predictable.
    system_message = (
        "You are a protein design assistant. Return only valid JSON that matches "
        "the provided schema. Do not wrap the JSON in code fences or prose."
    )

    user_message = f"""Context:
We are inserting an LLM reasoning step between scoring and ILP optimization.
Scores are objective values used by a minimization ILP (lower is better).

Contact graph:
<<<
{contact_graph_text}
>>>

Scores and mutation proposals:
<<<
{scores_text}
>>>

Mutation proposals list:
{proposals_text}

Task:
- Identify mutation combinations likely to disrupt binding (avoid).
- Suggest additional mutations to improve binding (mark novel suggestions).
- Propose a derived scoring function using contact graph features.
- Provide constraints suitable for ILP integration.
- Add any extra protein-informed insights.

Return JSON only using this schema:
{schema_text}

Output rules:
- Use only mutations from the proposals/scores unless setting is_novel=true.
- For avoid_combinations, include severity: "hard" or "soft".
- For derived_scoring_function.terms, specify term_type as "per_mutation", "per_pair", or "global".
- If no contacts are available, note this in warnings and avoid overconfident claims.
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def _extract_json_from_text(text: str) -> str:
    if not text:
        raise ValueError("Empty response text from LLM.")
    trimmed = text.strip()
    if trimmed.startswith("```"):
        trimmed = re.sub(r"^```(?:json)?", "", trimmed)
        trimmed = re.sub(r"```$", "", trimmed).strip()
    if match := re.search(r"\{.*\}", trimmed, flags=re.DOTALL):
        return match[0]
    else:
        raise ValueError("No JSON object found in LLM response.")


def _normalize_llm_output(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return dict(DEFAULT_LLM_OUTPUT)

    normalized = dict(payload)
    for key, value in DEFAULT_LLM_OUTPUT.items():
        normalized.setdefault(key, value)
    if not isinstance(normalized.get("derived_scoring_function"), dict):
        normalized["derived_scoring_function"] = dict(
            DEFAULT_LLM_OUTPUT["derived_scoring_function"]
        )
    normalized["derived_scoring_function"].setdefault("objective", "minimize")
    normalized["derived_scoring_function"].setdefault("terms", [])
    if not isinstance(normalized.get("warnings"), list):
        normalized["warnings"] = []
    return normalized


def run_llm_reasoning(
    contact_graph_text: str,
    scores_by_mutation: Optional[Dict[str, Dict[str, float]]] = None,
    mutation_proposals: Optional[Sequence[str]] = None,
    config: Optional[LLMReasoningConfig] = None,
    api_base: Optional[str] = None,
    include_raw_response: bool = False,
) -> Dict[str, Any]:
    """Call the LLM with structured prompts and parse JSON guidance."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    if api_base:
        os.environ["OPENAI_API_BASE"] = api_base

    scores_by_mutation = scores_by_mutation or {}
    mutation_proposals = mutation_proposals or []
    config = config or LLMReasoningConfig()

    messages = _build_messages(
        contact_graph_text, scores_by_mutation, mutation_proposals, config
    )

    try:
        from litellm import completion
    except ImportError as exc:
        raise ImportError("litellm is required to call the LLM.") from exc

    response = completion(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    raw_text = response.choices[0].message.content

    try:
        json_text = _extract_json_from_text(raw_text)
        payload = json.loads(json_text)
    except Exception as exc:
        logger.error(f"Failed to parse LLM JSON output: {exc}")
        payload = dict(DEFAULT_LLM_OUTPUT)
        payload["warnings"] = payload.get("warnings", []) + [
            "LLM output was not valid JSON; using default empty guidance."
        ]

    normalized = _normalize_llm_output(payload)
    if include_raw_response:
        normalized["raw_response"] = raw_text
    return normalized


CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--pdb-path", type=str, help="Path to PDB file for contact graph.")
@click.option("--heavy-chain-id", type=str, default="H", help="Antibody heavy chain.")
@click.option("--light-chain-id", type=str, default="L", help="Antibody light chain.")
@click.option("--antigen-chain-id", type=str, default="A", help="Antigen chain.")
@click.option(
    "--distance-threshold",
    type=float,
    default=7.0,
    help="Distance threshold for contacts.",
)
@click.option(
    "--contact-graph-text-file",
    type=click.Path(exists=True),
    help="Path to a precomputed contact graph text file.",
)
@click.option(
    "--scores-csv",
    type=click.Path(exists=True),
    help="CSV with Mutation column and score columns.",
)
@click.option(
    "--mutation",
    "mutations",
    multiple=True,
    type=str,
    help="Mutation proposal (repeatable).",
)
@click.option("--model", type=str, default="gpt-4o", help="LLM model name.")
@click.option("--temperature", type=float, default=0.2, help="LLM temperature.")
@click.option("--max-tokens", type=int, default=1500, help="Max tokens for LLM.")
@click.option(
    "--max-mut-in-prompt",
    type=int,
    default=50,
    help="Maximum mutations to include in prompt.",
)
@click.option(
    "--max-edges-in-prompt",
    type=int,
    default=200,
    help="Maximum contact edges to include in prompt.",
)
@click.option(
    "--api-base",
    type=str,
    default=None,
    help="Override OPENAI_API_BASE for LiteLLM.",
)
@click.option(
    "--output",
    type=click.Path(exists=False),
    default=None,
    help="Write JSON output to file (defaults to stdout).",
)
@click.option(
    "--include-raw-response",
    is_flag=True,
    help="Include raw LLM response text in output JSON.",
)
def cli(
    pdb_path: Optional[str],
    heavy_chain_id: str,
    light_chain_id: str,
    antigen_chain_id: str,
    distance_threshold: float,
    contact_graph_text_file: Optional[str],
    scores_csv: Optional[str],
    mutations: Tuple[str, ...],
    model: str,
    temperature: float,
    max_tokens: int,
    max_mut_in_prompt: int,
    max_edges_in_prompt: int,
    api_base: Optional[str],
    output: Optional[str],
    include_raw_response: bool,
) -> None:
    """CLI entrypoint for LLM reasoning."""
    contact_graph_text = ""

    if contact_graph_text_file:
        with open(contact_graph_text_file, "r") as handle:
            contact_graph_text = handle.read()
    elif pdb_path:
        contact_graph_text = build_contact_graph_text_from_pdb(
            pdb_path,
            heavy_chain_id,
            light_chain_id,
            antigen_chain_id,
            distance_threshold=distance_threshold,
            max_edges=max_edges_in_prompt,
        )
    else:
        logger.warning("No contact graph source provided; proceeding without contacts.")

    scores_by_mutation: Dict[str, Dict[str, float]] = {}
    mutation_proposals: List[str] = list(mutations)

    if scores_csv:
        df = pd.read_csv(scores_csv)
        scores_by_mutation = _scores_from_dataframe(df)
        if not mutation_proposals:
            mutation_proposals = list(scores_by_mutation.keys())

    config = LLMReasoningConfig(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_mutations_in_prompt=max_mut_in_prompt,
        max_contact_edges_in_prompt=max_edges_in_prompt,
    )

    result = run_llm_reasoning(
        contact_graph_text=contact_graph_text,
        scores_by_mutation=scores_by_mutation,
        mutation_proposals=mutation_proposals,
        config=config,
        api_base=api_base,
        include_raw_response=include_raw_response,
    )

    if output:
        with open(output, "w") as handle:
            json.dump(result, handle, indent=2)
        logger.info(f"Wrote LLM guidance to {output}")
    else:
        click.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    cli()
