import json
import os
import re
import statistics
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import click
import pandas as pd

from protlib_designer import logger
from protlib_designer.structure.contact_graph import compute_contact_edges
from protlib_designer.structure.interface_profile import (
    build_interface_profile_text,
    profile_antibody_antigen_interactions,
)

warnings.filterwarnings("ignore")


CONTACT_GRAPH_EMPTY_SENTINEL = "NO_CONTACTS"
DEFAULT_LLM_GUIDANCE_FILENAME = "llm_guidance.json"
DEFAULT_LLM_PROMPT_JSON_FILENAME = "llm_prompt.json"
DEFAULT_LLM_PROMPT_TEXT_FILENAME = "llm_prompt.txt"
DEFAULT_LLM_DERIVED_SCORE_COLUMN = "llm_derived_score"

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
    max_interaction_pairs_in_prompt: int = 12
    max_interface_residues_in_prompt: int = 10
    reasoning_effort: Optional[str] = None


def _chain_from_residue(residue: str) -> str:
    """Extract the chain ID from a residue string like 'WH35' or 'H35'."""
    if not residue:
        return ""
    if len(residue) >= 2 and residue[0].isalpha() and residue[1].isalpha():
        return residue[1]
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


def build_interface_profile_text_from_pdb(
    pdb_file: str,
    heavy_chain_id: str,
    light_chain_id: str,
    antigen_chain_id: str,
    distance_threshold: float = 7.0,
    max_pairs: int = 12,
    max_contact_residues: int = 10,
) -> str:
    """Compute and render interaction profile text from a PDB file."""
    edges = compute_contact_edges(
        pdb_file,
        heavy_chain_id,
        light_chain_id,
        antigen_chain_id,
        distance_threshold=distance_threshold,
    )
    interaction_profile = profile_antibody_antigen_interactions(
        pdb_file, heavy_chain_id, light_chain_id, antigen_chain_id
    )
    return build_interface_profile_text(
        edges,
        interaction_profile,
        max_pairs=max_pairs,
        max_contact_residues=max_contact_residues,
    )


def _resolve_mutation_column(df: pd.DataFrame) -> str:
    """Resolve the mutation column name from known variants."""
    preferred_columns = ("Mutation", "MutationHL")
    for column in preferred_columns:
        if column in df.columns:
            return column

    lower_to_original = {column.lower(): column for column in df.columns}
    for column in preferred_columns:
        if column.lower() in lower_to_original:
            return lower_to_original[column.lower()]

    raise ValueError("Scores CSV must include a 'Mutation' or 'MutationHL' column.")


def _scores_from_dataframe(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    mutation_column = _resolve_mutation_column(df)
    score_columns = [col for col in df.columns if col != mutation_column]
    scores_by_mutation: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        mutation = str(row[mutation_column])
        scores: Dict[str, float] = {}
        for column in score_columns:
            value = row[column]
            scores[column] = float("nan") if pd.isna(value) else float(value)
        scores_by_mutation[mutation] = scores
    return scores_by_mutation


def _apply_llm_derived_scoring(
    df: pd.DataFrame,
    derived_scoring: Dict[str, Any],
    column_name: str,
    mutation_column: str,
) -> tuple[pd.DataFrame, bool]:
    """Apply per-mutation derived scoring terms to a score dataframe."""
    terms = derived_scoring.get("terms", []) if derived_scoring else []
    per_mutation_terms = [
        term
        for term in terms
        if isinstance(term, dict) and term.get("term_type") == "per_mutation"
    ]
    if not per_mutation_terms:
        logger.info("No per-mutation derived scoring terms; skipping derived column.")
        return df, False

    updated = df.copy()
    if column_name in updated.columns:
        logger.warning(
            f"Derived scoring column {column_name} already exists; overwriting."
        )
    updated[column_name] = 0.0
    objective = (derived_scoring or {}).get("objective", "minimize")
    multiplier = -1.0 if str(objective).lower() == "maximize" else 1.0

    for term in per_mutation_terms:
        weight = float(term.get("weight", 0.0)) * multiplier
        mutations = term.get("mutations", [])
        if not mutations:
            continue
        mask = updated[mutation_column].astype(str).isin(mutations)
        updated.loc[mask, column_name] = updated.loc[mask, column_name] + weight

    for term in terms:
        if not isinstance(term, dict):
            continue
        term_type = term.get("term_type")
        if term_type in {"per_pair", "global"}:
            logger.warning(
                f"Skipping derived scoring term_type={term_type}; not supported by linear objective."
            )
    return updated, True


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
        if missing := [m for m in mutation_proposals if m not in scores_by_mutation]:
            lines.append("Mutation proposals without scores:")
            lines.extend(f"- {mutation}" for mutation in missing)
    return "\n".join(lines)


def _resolve_output_path(
    output_path: Optional[str], output_dir: Path, default_filename: str
) -> str:
    """Resolve an output path within a directory, preserving absolute paths."""
    candidate = output_path or default_filename
    return candidate if os.path.isabs(candidate) else str(output_dir / candidate)


def _build_messages(
    contact_graph_text: str,
    interface_profile_text: str,
    scores_by_mutation: Dict[str, Dict[str, float]],
    mutation_proposals: Sequence[str],
    config: LLMReasoningConfig,
) -> List[Dict[str, str]]:
    if not contact_graph_text.strip():
        contact_graph_text = CONTACT_GRAPH_EMPTY_SENTINEL
    if not interface_profile_text.strip():
        interface_profile_text = "NO_INTERFACE_PROFILE"

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

Interface interaction profile:
<<<
{interface_profile_text}
>>>

Residue notation:
- Residue labels use WT+CHAIN+INDEX (e.g., WB99).
- Mutation labels use WT+CHAIN+INDEX+MUT (e.g., WB99A).

Scores and mutation proposals:
<<<
{scores_text}
>>>

Mutation proposals list:
{proposals_text}

Task:
- Identify mutation combinations likely to disrupt binding (avoid).
- Suggest additional mutations to improve binding (mark novel suggestions).
- Propose a derived scoring function using contact graph and interaction features.
- Provide constraints suitable for ILP integration.
- Add any extra protein-informed insights.

Return JSON only using this schema:
{schema_text}

Output rules:
- Use only mutations from the proposals/scores unless setting is_novel=true.
- For avoid_combinations, include severity: "hard" or "soft".
- For derived_scoring_function.terms, specify term_type as "per_mutation", "per_pair", or "global".
- If no contacts or interaction data are available, note this in warnings and avoid overconfident claims.
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


def _truncate_text(text: Optional[str], limit: int = 2000) -> str:
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated {len(text) - limit} chars]"


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


def _coerce_temperature(model: str, temperature: float) -> float:
    model_id = model.split("/")[-1]
    if model_id.startswith("gpt-5") and not model_id.startswith("gpt-5.1"):
        if temperature != 1.0:
            logger.warning(
                "gpt-5 models only support temperature=1; overriding requested value."
            )
        return 1.0
    return temperature


def _resolve_reasoning_effort(
    model: str, reasoning_effort: Optional[str]
) -> Optional[str]:
    model_id = model.split("/")[-1]
    if reasoning_effort is not None:
        return reasoning_effort
    if model_id.startswith("gpt-5"):
        logger.warning(
            "gpt-5 defaults to reasoning_effort='none' to ensure text output; override if needed."
        )
        return "none"
    return None


def run_llm_reasoning(
    contact_graph_text: str,
    interface_profile_text: str = "",
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
        contact_graph_text,
        interface_profile_text or "",
        scores_by_mutation,
        mutation_proposals,
        config,
    )

    try:
        from litellm import completion
    except ImportError as exc:
        raise ImportError("litellm is required to call the LLM.") from exc

    temperature = _coerce_temperature(config.model, config.temperature)
    reasoning_effort = _resolve_reasoning_effort(config.model, config.reasoning_effort)
    completion_kwargs: Dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": config.max_tokens,
    }
    if reasoning_effort is not None and config.model.split("/")[-1].startswith("gpt-5"):
        completion_kwargs["reasoning_effort"] = reasoning_effort

    response = completion(**completion_kwargs)
    raw_text = None
    finish_reason = None
    try:
        raw_text = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
    except Exception as exc:
        logger.error(f"Unable to read LLM response content: {exc}")
        logger.error(f"LLM response object: {response}")

    if not raw_text:
        logger.error(
            "LLM response text is empty. "
            f"finish_reason={finish_reason}, response={response}"
        )

    try:
        json_text = _extract_json_from_text(raw_text)
        payload = json.loads(json_text)
    except Exception as exc:
        logger.error(
            "Failed to parse LLM JSON output: "
            f"{exc}. Raw response: {_truncate_text(raw_text)}"
        )
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
    "--reasoning-effort",
    type=str,
    default=None,
    help="Override reasoning_effort for LLM (e.g., 'none', 'low', 'medium', 'high').",
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
    "--prompt-output",
    type=click.Path(exists=False),
    default=None,
    help="Write the LLM prompt messages to a JSON file.",
)
@click.option(
    "--prompt-text-output",
    type=click.Path(exists=False),
    default=None,
    help="Write the LLM prompt content to a plain-text file.",
)
@click.option(
    "--llm-output-dir",
    type=click.Path(exists=False),
    default=None,
    help=(
        "Directory to collect LLM artifacts (guidance and prompt files). "
        "Relative output paths are placed under this directory."
    ),
)
@click.option(
    "--llm-derived-score-column",
    type=str,
    default=DEFAULT_LLM_DERIVED_SCORE_COLUMN,
    help="Column name for per-mutation derived scoring values.",
)
@click.option(
    "--llm-scores-output",
    type=click.Path(exists=False),
    default=None,
    help="Optional output path for scores with the LLM-derived score column.",
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
    reasoning_effort: Optional[str],
    api_base: Optional[str],
    output: Optional[str],
    prompt_output: Optional[str],
    prompt_text_output: Optional[str],
    llm_output_dir: Optional[str],
    llm_derived_score_column: str,
    llm_scores_output: Optional[str],
    include_raw_response: bool,
) -> None:
    """CLI entrypoint for LLM reasoning."""
    contact_graph_text = ""
    interface_profile_text = ""
    llm_output_dir_path: Optional[Path] = None

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

    config = LLMReasoningConfig(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_mutations_in_prompt=max_mut_in_prompt,
        max_contact_edges_in_prompt=max_edges_in_prompt,
        reasoning_effort=reasoning_effort,
    )

    if llm_output_dir:
        llm_output_dir_path = Path(llm_output_dir)
        llm_output_dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"LLM artifacts directory: {llm_output_dir_path}")
        output = _resolve_output_path(
            output, llm_output_dir_path, DEFAULT_LLM_GUIDANCE_FILENAME
        )
        prompt_output = _resolve_output_path(
            prompt_output, llm_output_dir_path, DEFAULT_LLM_PROMPT_JSON_FILENAME
        )
        prompt_text_output = _resolve_output_path(
            prompt_text_output, llm_output_dir_path, DEFAULT_LLM_PROMPT_TEXT_FILENAME
        )

    if pdb_path:
        interface_profile_text = build_interface_profile_text_from_pdb(
            pdb_path,
            heavy_chain_id,
            light_chain_id,
            antigen_chain_id,
            distance_threshold=distance_threshold,
            max_pairs=config.max_interaction_pairs_in_prompt,
            max_contact_residues=config.max_interface_residues_in_prompt,
        )
    elif contact_graph_text_file:
        logger.warning("No PDB path provided; skipping interface interaction profile.")

    scores_by_mutation: Dict[str, Dict[str, float]] = {}
    mutation_proposals: List[str] = list(mutations)
    scores_df: Optional[pd.DataFrame] = None
    mutation_column: Optional[str] = None

    if scores_csv:
        df = pd.read_csv(scores_csv)
        scores_df = df
        mutation_column = _resolve_mutation_column(df)
        scores_by_mutation = _scores_from_dataframe(df)
        if not mutation_proposals:
            mutation_proposals = list(scores_by_mutation.keys())

    prompt_messages = None
    if prompt_output or prompt_text_output:
        prompt_messages = _build_messages(
            contact_graph_text,
            interface_profile_text,
            scores_by_mutation,
            mutation_proposals,
            config,
        )
    if prompt_output:
        with open(prompt_output, "w") as handle:
            json.dump(prompt_messages, handle, indent=2)
        logger.info(f"Wrote LLM prompt messages to {prompt_output}")
    if prompt_text_output:
        text_lines = []
        for message in prompt_messages:
            role = message.get("role", "unknown").upper()
            content = message.get("content", "")
            text_lines.append(f"{role}:\n{content}\n")
        with open(prompt_text_output, "w") as handle:
            handle.write("\n".join(text_lines))
        logger.info(f"Wrote LLM prompt text to {prompt_text_output}")

    result = run_llm_reasoning(
        contact_graph_text=contact_graph_text,
        interface_profile_text=interface_profile_text,
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

    if scores_df is not None and mutation_column:
        scores_output_path: Optional[str] = llm_scores_output
        default_scores_filename = (
            Path(scores_csv).name if scores_csv else "scores_with_llm.csv"
        )
        if llm_output_dir_path:
            scores_output_path = _resolve_output_path(
                scores_output_path, llm_output_dir_path, default_scores_filename
            )

        if scores_output_path:
            derived_scoring = result.get("derived_scoring_function", {})
            updated_df, _ = _apply_llm_derived_scoring(
                scores_df,
                derived_scoring,
                llm_derived_score_column,
                mutation_column,
            )
            updated_df.to_csv(scores_output_path, index=False)
            logger.info(f"Wrote LLM-derived scores to {scores_output_path}")


if __name__ == "__main__":
    cli()
