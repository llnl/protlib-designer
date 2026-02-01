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

Output format:
Mutation
"WH99D,DH102Y,FH104Y,YH105W,MH107F"
"WH99D,GH100Y,DH102Y,FH104Y,YH105W"
"WH99E,GH100Y,DH102Y,FH104Y,YH105W"
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
    "--output",
    type=click.Path(exists=False),
    default="llm_library.csv",
    help="Write CSV output to file.",
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
    reasoning_effort = _resolve_reasoning_effort(model, reasoning_effort)
    completion_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": prompt_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning_effort is not None and model.split("/")[-1].startswith("gpt-5"):
        completion_kwargs["reasoning_effort"] = reasoning_effort

    response = completion(**completion_kwargs)
    raw_text = response.choices[0].message.content
    mutation_lines = _extract_mutation_lines(raw_text)

    if not mutation_lines:
        logger.error("No mutation lines parsed from LLM response.")

    if len(mutation_lines) > num_mutants:
        mutation_lines = mutation_lines[:num_mutants]

    output_df = pd.DataFrame({"Mutation": mutation_lines})
    output_df.to_csv(output, index=False)
    logger.info(f"Wrote LLM-generated library to {output}")


if __name__ == "__main__":
    run_llm_library_generator()
