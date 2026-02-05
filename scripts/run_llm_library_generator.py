#!/usr/bin/env python
import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import click
import pandas as pd

from protlib_designer import logger
from protlib_designer.llm_reasoning import (
    _format_scores_for_prompt,
    _scores_from_dataframe,
    build_contact_graph_text_from_pdb,
    build_interface_profile_text_from_pdb,
    _infer_chain_label_map,
)

warnings.filterwarnings("ignore")

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
DEFAULT_LLM_LIBRARY_FILENAME = "llm_library.csv"
DEFAULT_LLM_PROMPT_JSON_FILENAME = "llm_prompt.json"
DEFAULT_LLM_PROMPT_TEXT_FILENAME = "llm_prompt.txt"
DEFAULT_LLM_RESPONSE_JSON_FILENAME = "llm_response.json"


def _coerce_temperature(model: str, temperature: float) -> float:
    model_id = model.split("/")[-1]
    if model_id.startswith("gpt-5") and not model_id.startswith("gpt-5.1"):
        if temperature != 1.0:
            logger.warning(
                "gpt-5 models only support temperature=1; overriding requested value."
            )
        return 1.0
    return temperature


def _is_azure_endpoint(api_base: Optional[str]) -> bool:
    base = api_base or os.environ.get("OPENAI_API_BASE", "")
    if not base:
        return False
    base = base.lower()
    return "azure" in base or "openai.azure.com" in base


def _resolve_reasoning_effort(
    model: str, reasoning_effort: Optional[str], api_base: Optional[str]
) -> Optional[str]:
    model_id = model.split("/")[-1]
    if reasoning_effort is not None:
        return reasoning_effort
    if model_id.startswith("gpt-5"):
        if _is_azure_endpoint(api_base):
            logger.warning(
                "Azure gpt-5 requires reasoning_effort in {'low','medium','high'}; "
                "defaulting to 'low'."
            )
            return "low"
        logger.warning(
            "gpt-5 defaults to reasoning_effort='none' to ensure text output; override if needed."
        )
        return "none"
    return None


def _sanitize_model_name(model_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")
    return sanitized or "llm_output"


def _resolve_output_dir(output_dir: Optional[str], model: str) -> str:
    return output_dir or f"{_sanitize_model_name(model)}_llm_generation"


def _build_library_prompt(
    contact_graph_text: str,
    interface_profile_text: str,
    scores_by_mutation: Dict[str, Dict[str, float]],
    mutation_proposals: Sequence[str],
    num_mutants: int,
    max_mutations_in_prompt: int,
) -> List[Dict[str, str]]:
    if not contact_graph_text.strip():
        contact_graph_text = "NO_CONTACTS"
    if not interface_profile_text.strip():
        interface_profile_text = "NO_INTERFACE_PROFILE"

    scores_text = _format_scores_for_prompt(
        scores_by_mutation, mutation_proposals, max_mutations_in_prompt
    )
    proposals_text = (
        "\n".join(f"- {mutation}" for mutation in mutation_proposals)
        if mutation_proposals
        else "None provided."
    )
    example_lines = _build_example_lines(mutation_proposals, num_mutants)

    system_message = (
        "You are a protein design assistant. Output only CSV with a single column "
        "named Mutation. Do not wrap output in code fences or prose."
    )

    user_message = f"""Context:
We are generating a diverse library of mutant antibody sequences directly from an LLM.
Use the provided structural and scoring context to propose {num_mutants} unique mutants.

Residue notation:
- Residue labels use WT+CHAIN+INDEX (e.g., WB99).
- Mutation labels use WT+CHAIN+INDEX+MUT (e.g., WB99A).

Contact graph:
<<<
{contact_graph_text}
>>>

Interface interaction profile:
<<<
{interface_profile_text}
>>>

Scores and mutation proposals:
<<<
{scores_text}
>>>

Mutation proposals list:
{proposals_text}

Task:
- Generate {num_mutants} unique mutant sequences.
- Each line should be a comma-separated list of single mutations (no spaces).
- Use only the provided mutation proposals/scores.
- Ensure diversity (vary positions and combinations; avoid duplicates).
- Ensure every mutation includes the final mutant amino acid (e.g., YH105W, not YH105).
- Every mutation token must match WT+CHAIN+INDEX+MUT (regex: ^[A-Z][A-Z][0-9]+[A-Z]$).

Output format (use only mutations from the provided list):
Mutation
{example_lines}
... (exactly {num_mutants} lines)
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def _strip_code_fences(text: str) -> str:
    trimmed = text.strip()
    if trimmed.startswith("```"):
        trimmed = re.sub(r"^```(?:csv)?", "", trimmed)
        trimmed = re.sub(r"```$", "", trimmed).strip()
    return trimmed


def _extract_mutation_lines(text: str) -> List[str]:
    if not text:
        return []
    trimmed = _strip_code_fences(text)
    # Try JSON list first
    if trimmed.startswith("["):
        try:
            payload = json.loads(trimmed)
            if isinstance(payload, list):
                return [str(item) for item in payload if str(item).strip()]
        except Exception:
            pass

    lines = []
    for raw in trimmed.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("mutation"):
            # Skip CSV header
            if line.lower() == "mutation" or line.lower().startswith("mutation,"):
                continue
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
        lines.append(line.strip())
    return lines


def _validate_mutation_line(line: str) -> bool:
    token_pattern = re.compile(r"^[A-Z][A-Z][0-9]+[A-Z]$")
    tokens = [token.strip() for token in line.split(",") if token.strip()]
    if not tokens:
        return False
    return all(token_pattern.match(token) for token in tokens)


def _filter_lines_to_allowed_mutations(
    lines: List[str], allowed: Sequence[str]
) -> List[str]:
    if not allowed:
        return lines
    allowed_set = set(allowed)
    filtered = []
    for line in lines:
        tokens = [token.strip() for token in line.split(",") if token.strip()]
        if tokens and all(token in allowed_set for token in tokens):
            filtered.append(line)
    return filtered


def _build_example_lines(
    mutation_proposals: Sequence[str], num_mutants: int
) -> str:
    if not mutation_proposals:
        return '"WH99D,DH102Y,FH104Y,YH105W,MH107F"'
    # Build 2 example lines from proposals to avoid leaking unrelated mutations.
    proposals = list(dict.fromkeys(mutation_proposals))
    line_len = min(5, len(proposals))
    first = proposals[:line_len]
    second = proposals[line_len : 2 * line_len] or first
    example_lines = [
        f"\"{','.join(first)}\"",
        f"\"{','.join(second)}\"",
    ]
    return "\n".join(example_lines)


def _extract_response_text(response: Any) -> str:
    try:
        content = response.choices[0].message.content
        if content:
            return content
    except Exception:
        pass
    try:
        message = response.choices[0].message
        if isinstance(message, dict):
            content = message.get("content")
            if content:
                return content
    except Exception:
        pass
    return ""


def _dump_response(response: Any, output_path: str) -> None:
    try:
        if hasattr(response, "model_dump"):
            payload = response.model_dump()
        elif hasattr(response, "dict"):
            payload = response.dict()
        else:
            payload = str(response)
        with open(output_path, "w") as handle:
            json.dump(payload, handle, indent=2, default=str)
        logger.info(f"Wrote raw LLM response to {output_path}")
    except Exception as exc:
        logger.warning(f"Failed to write raw LLM response: {exc}")


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--pdb-path", type=str, help="Path to PDB file for contact profiling.")
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
@click.option(
    "--num-mutants",
    type=int,
    default=1000,
    help="Number of mutant sequences to generate.",
)
@click.option("--model", type=str, default="gpt-4o", help="LLM model name.")
@click.option("--temperature", type=float, default=0.2, help="LLM temperature.")
@click.option("--max-tokens", type=int, default=2000, help="Max tokens for LLM.")
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
    "--output-dir",
    type=click.Path(exists=False),
    default=None,
    help="Directory to write outputs (defaults to a sanitized model name).",
)
@click.option(
    "--output",
    type=click.Path(exists=False),
    default=DEFAULT_LLM_LIBRARY_FILENAME,
    help="Write CSV output to file (relative to output dir unless absolute).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the prompt and exit without calling the LLM.",
)
def run_llm_library_generator(
    pdb_path: Optional[str],
    heavy_chain_id: str,
    light_chain_id: str,
    antigen_chain_id: str,
    distance_threshold: float,
    contact_graph_text_file: Optional[str],
    scores_csv: Optional[str],
    mutations: Tuple[str, ...],
    num_mutants: int,
    model: str,
    temperature: float,
    max_tokens: int,
    max_mut_in_prompt: int,
    max_edges_in_prompt: int,
    reasoning_effort: Optional[str],
    api_base: Optional[str],
    output_dir: Optional[str],
    output: str,
    dry_run: bool,
) -> None:
    scores_by_mutation: Dict[str, Dict[str, float]] = {}
    mutation_proposals: List[str] = list(mutations)

    if scores_csv:
        df = pd.read_csv(scores_csv)
        scores_by_mutation = _scores_from_dataframe(df)
        if not mutation_proposals:
            mutation_proposals = list(scores_by_mutation.keys())

    chain_id_map = _infer_chain_label_map(
        mutation_proposals,
        heavy_chain_id,
        light_chain_id,
        antigen_chain_id,
    )

    contact_graph_text = ""
    interface_profile_text = ""

    if contact_graph_text_file:
        with open(contact_graph_text_file, "r") as handle:
            contact_graph_text = handle.read()
        if chain_id_map:
            logger.warning(
                "Contact graph text file provided; chain label remapping was not applied."
            )
    elif pdb_path:
        contact_graph_text = build_contact_graph_text_from_pdb(
            pdb_path,
            heavy_chain_id,
            light_chain_id,
            antigen_chain_id,
            distance_threshold=distance_threshold,
            max_edges=max_edges_in_prompt,
            chain_id_map=chain_id_map,
        )
        interface_profile_text = build_interface_profile_text_from_pdb(
            pdb_path,
            heavy_chain_id,
            light_chain_id,
            antigen_chain_id,
            distance_threshold=distance_threshold,
            chain_id_map=chain_id_map,
        )
    else:
        logger.warning("No contact graph source provided; proceeding without contacts.")

    prompt_messages = _build_library_prompt(
        contact_graph_text,
        interface_profile_text,
        scores_by_mutation,
        mutation_proposals,
        num_mutants,
        max_mut_in_prompt,
    )

    resolved_output_dir = _resolve_output_dir(output_dir, model)
    os.makedirs(resolved_output_dir, exist_ok=True)
    output_path = (
        output if os.path.isabs(output) else os.path.join(resolved_output_dir, output)
    )
    prompt_json_path = os.path.join(
        resolved_output_dir, DEFAULT_LLM_PROMPT_JSON_FILENAME
    )
    prompt_text_path = os.path.join(
        resolved_output_dir, DEFAULT_LLM_PROMPT_TEXT_FILENAME
    )
    response_json_path = os.path.join(
        resolved_output_dir, DEFAULT_LLM_RESPONSE_JSON_FILENAME
    )

    with open(prompt_json_path, "w") as handle:
        json.dump(prompt_messages, handle, indent=2)
    text_lines = []
    for message in prompt_messages:
        role = message.get("role", "unknown").upper()
        content = message.get("content", "")
        text_lines.append(f"{role}:\n{content}\n")
    with open(prompt_text_path, "w") as handle:
        handle.write("\n".join(text_lines))
    logger.info(f"Wrote LLM prompt messages to {prompt_json_path}")
    logger.info(f"Wrote LLM prompt text to {prompt_text_path}")

    if dry_run:
        click.echo(prompt_messages[0]["content"])
        click.echo("")
        click.echo(prompt_messages[1]["content"])
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    if api_base:
        os.environ["OPENAI_API_BASE"] = api_base

    try:
        from litellm import completion
    except ImportError as exc:
        raise ImportError("litellm is required to call the LLM.") from exc

    temperature = _coerce_temperature(model, temperature)
    reasoning_effort = _resolve_reasoning_effort(model, reasoning_effort, api_base)
    completion_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": prompt_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning_effort is not None and model.split("/")[-1].startswith("gpt-5"):
        completion_kwargs["reasoning_effort"] = reasoning_effort

    response = completion(**completion_kwargs)
    _dump_response(response, response_json_path)
    raw_text = _extract_response_text(response)
    mutation_lines = _extract_mutation_lines(raw_text)

    if not mutation_lines:
        finish_reason = None
        try:
            finish_reason = response.choices[0].finish_reason
        except Exception:
            pass
        logger.error(
            "No mutation lines parsed from LLM response. finish_reason=%s",
            finish_reason,
        )
    else:
        invalid_lines = [
            line for line in mutation_lines if not _validate_mutation_line(line)
        ]
        if invalid_lines:
            logger.warning(
                "Dropping %d invalid mutation lines that do not match WT+CHAIN+INDEX+MUT.",
                len(invalid_lines),
            )
            mutation_lines = [line for line in mutation_lines if line not in invalid_lines]
        if mutation_proposals:
            filtered_lines = _filter_lines_to_allowed_mutations(
                mutation_lines, mutation_proposals
            )
            dropped = len(mutation_lines) - len(filtered_lines)
            if dropped:
                logger.warning(
                    "Dropping %d lines containing mutations not in the provided proposals.",
                    dropped,
                )
            mutation_lines = filtered_lines

    if len(mutation_lines) > num_mutants:
        mutation_lines = mutation_lines[:num_mutants]

    output_df = pd.DataFrame({"Mutation": mutation_lines})
    output_df.to_csv(output_path, index=False)
    logger.info(f"Wrote LLM-generated library to {output_path}")


if __name__ == "__main__":
    run_llm_library_generator()
