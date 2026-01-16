#!/usr/bin/env python
import json
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import pandas as pd

from protlib_designer import logger
from protlib_designer.dataloader import DataLoader
from protlib_designer.filter.no_filter import NoFilter
from protlib_designer.generator.ilp_generator import ILPGenerator
from protlib_designer.llm_reasoning import (
    LLMReasoningConfig,
    build_contact_graph_text_from_pdb,
    run_llm_reasoning,
)
from protlib_designer.scorer.ifold_scorer import IFOLDScorer
from protlib_designer.scorer.plm_scorer import PLMScorer
from protlib_designer.solution_manager import SolutionManager
from protlib_designer.solver.generate_and_remove_solver import GenerateAndRemoveSolver
from protlib_designer.utils import (
    cif_to_pdb,
    extract_sequence_from_pdb,
    format_and_validate_protlib_designer_parameters,
    write_config,
)

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


def format_and_validate_pipeline_parameters(
    sequence: str = None,
    pdb_path: str = None,
    positions: list = None,
    plm_model_names: list = None,
    plm_model_paths: list = None,
    ifold_model_name: str = None,
    ifold_model_path: str = None,
):
    """Validate the parameters for the Protlib Designer pipeline."""

    if not sequence and not pdb_path:
        raise ValueError("You must provide either a sequence or a PDB file path.")

    if pdb_path and not Path(pdb_path).exists():
        raise ValueError(f"The provided PDB path does not exist: {pdb_path}")

    if not positions or len(positions[0]) <= 1:
        raise ValueError(
            "You must provide positions correctly formatted {Wildtype}{Chain}{Position} or {*}{Chain}{*}."
        )

    if not sequence:
        if not pdb_path or not Path(pdb_path).exists():
            raise ValueError(
                f"There is no way to extract the sequence. You must provide a sequence directly or extract it from a PDB file. Provided sequence: {sequence}, PDB path: {pdb_path}."
            )

        pdb_chain = positions[0][1]
        logger.warning(
            f"No sequence provided. Extracting the chain ID {pdb_chain} from the first position {positions[0]}."
        )
        sequence = extract_sequence_from_pdb(pdb_path, pdb_chain)
        logger.warning(
            f"No sequence provided. Extracting sequence: {sequence} from PDB file {pdb_path} and chain {pdb_chain}."
        )

    if len(positions) == 1 and positions[0].startswith("*"):
        logger.warning(
            f"A placeholder position was provided {positions}, generating positions for the entire sequence."
        )
        if positions[0].endswith("*"):
            positions = [
                f"{sequence[i]}{positions[0][1]}{i+1}" for i in range(len(sequence))
            ]
        elif positions[0][2] == "{" and positions[0][-1] == "}":
            range_str = positions[0][3:-1]
            start, end = map(int, range_str.split("-"))
            positions = [
                f"{sequence[i]}{positions[0][1]}{i+1}" for i in range(start - 1, end)
            ]
        else:
            raise ValueError(
                f"Invalid position format: {positions[0]}. Expected format: {{*}}{{Chain}}{{*}} or {{*}}{{Chain}}{{start-end}}."
            )
        logger.warning(f"Inferred positions: {positions}")

    return sequence, positions


def append_plm_scores_to_dataframes(
    dataframes: list,
    sequence: str,
    positions: list,
    plm_model_names: list = None,
    plm_model_paths: list = None,
    chain_type: str = "heavy",
    score_type: str = "minus_llr",
    mask: bool = True,
    mapping: str = None,
):
    """Append PLM scores to the provided dataframes."""
    for model_name in plm_model_names:
        plm_scorer = PLMScorer(
            model_name=model_name,
            model_path=None,
            score_type=score_type,
            mask=True,
            mapping=None,
        )
        df = plm_scorer.get_scores(sequence, list(positions), chain_type)
        dataframes.append(df)

    for model_path in plm_model_paths:
        plm_scorer = PLMScorer(
            model_name=model_path,
            model_path=model_path,
            score_type=score_type,
            mask=mask,
            mapping=mapping,
        )
        df = plm_scorer.get_scores(sequence, list(positions), chain_type)
        dataframes.append(df)


def append_ifold_scores_to_dataframes(
    dataframes: list,
    pdb_path: str,
    positions: list,
    ifold_model_name: str = None,
    ifold_model_path: str = None,
    seed: int = None,
    score_type: str = "minus_llr",
):
    """Append IFold scores to the provided dataframes."""

    ifold_scorer = IFOLDScorer(
        seed=seed,
        model_name=ifold_model_name,
        model_path=ifold_model_path,
        score_type=score_type,
    )
    df = ifold_scorer.get_scores(pdb_path, list(positions))
    dataframes.append(df)


def combine_dataframes(dataframes: list):
    """Combine multiple dataframes on the 'Mutation' column."""

    for df in dataframes:
        if "Mutation" not in df.columns:
            logger.error(
                "Data file must have at minimum the Mutation column and at least one Objective/Target."
            )
            return None
        if len(df) != len(dataframes[0]):
            logger.error("Data files must have the same number of rows.")
            return None

    return reduce(
        lambda left, right: pd.merge(left, right, on="Mutation", how="left"),
        dataframes,
    )


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


def _load_mutation_list_from_csv(path: str) -> List[str]:
    df = pd.read_csv(path)
    if "Mutation" not in df.columns:
        raise ValueError("Candidates CSV must include a 'Mutation' column.")
    return [str(val) for val in df["Mutation"].dropna().values.tolist()]


def _collect_llm_constraints(llm_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    constraints: List[Dict[str, Any]] = []
    for constraint in llm_output.get("constraints", []) or []:
        if not isinstance(constraint, dict):
            continue
        ctype = constraint.get("type")
        if ctype not in {
            "forbid_mutation",
            "forbid_combination",
            "require_mutation",
            "limit_count",
        }:
            continue
        if mutations := constraint.get("mutations", []):
            constraints.append(
                {
                    "type": ctype,
                    "mutations": list(mutations),
                    "limit": constraint.get("limit"),
                    "reason": constraint.get("reason", ""),
                }
            )

    for combo in llm_output.get("avoid_combinations", []) or []:
        if not isinstance(combo, dict):
            continue
        if combo.get("severity") != "hard":
            if combo.get("severity"):
                logger.info(
                    "Skipping soft avoid_combinations entry; only hard constraints are enforced."
                )
            continue
        if mutations := combo.get("mutations", []):
            constraints.append(
                {
                    "type": "forbid_combination",
                    "mutations": list(mutations),
                    "limit": None,
                    "reason": combo.get("reason", ""),
                }
            )

    return constraints


def _apply_llm_derived_scoring(
    df: pd.DataFrame,
    derived_scoring: Dict[str, Any],
    column_name: str,
) -> tuple[pd.DataFrame, bool]:
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
        mask = updated["Mutation"].isin(mutations)
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


def _build_contact_graph_text(
    pdb_path: Optional[str],
    contact_graph_text_file: Optional[str],
    heavy_chain_id: str,
    light_chain_id: str,
    antigen_chain_id: str,
    distance_threshold: float,
    max_edges: int,
) -> str:
    if contact_graph_text_file:
        with open(contact_graph_text_file, "r") as handle:
            return handle.read()
    if pdb_path:
        return build_contact_graph_text_from_pdb(
            pdb_path,
            heavy_chain_id,
            light_chain_id,
            antigen_chain_id,
            distance_threshold=distance_threshold,
            max_edges=max_edges,
        )
    return ""


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("positions", type=str, nargs=-1)
@click.option("--sequence", type=str, help="Protein sequence for PLM scoring")
@click.option("--pdb-path", type=str, help="Path to PDB file for IFold scoring")
@click.option("--plm-model-names", type=str, multiple=True, help="PLM model names")
@click.option("--plm-model-paths", type=str, multiple=True, help="PLM model paths")
@click.option("--ifold-model-name", type=str, default=None, help="IFold model name")
@click.option("--ifold-model-path", type=str, default=None, help="IFold model path")
@click.option(
    "--score-type",
    type=click.Choice(["minus_ll", "minus_llr"]),
    default="minus_llr",
    help="Score type for scoring",
)
@click.option(
    "--intermediate-output",
    type=str,
    default="combined_scores.csv",
    help="Path to save intermediate scores",
)
@click.option(
    "--plm-chain-type", type=str, default="heavy", help="Chain type for PLM scoring"
)
@click.option(
    "--plm-mask/--no-plm-mask", default=True, help="Whether to mask wildtype amino acid"
)
@click.option(
    "--plm-mapping", type=str, default=None, help="Mapping file for PLM scoring"
)
@click.option(
    "--ifold-seed", type=int, default=None, help="Random seed for IFold scoring"
)
@click.option(
    "--nb-iterations",
    default=10,
    type=int,
    help="Number of iterations for the designer",
)
@click.option("--min-mut", default=1, type=int, help="Minimum number of mutations")
@click.option("--max-mut", default=4, type=int, help="Maximum number of mutations")
@click.option(
    "--output-folder",
    default="lp_solution",
    type=click.Path(exists=False),
    help="Output folder for the designer results",
)
@click.option(
    "--forbidden-aa", type=str, help="Comma-separated list of forbidden amino acids"
)
@click.option(
    "--max-arom-per-seq",
    type=int,
    help="Maximum number of aromatic residues per sequence",
)
@click.option(
    "--dissimilarity-tolerance",
    default=0.0,
    type=float,
    help="Dissimilarity tolerance for the designer",
)
@click.option(
    "--interleave-mutant-order",
    default=False,
    type=bool,
    help="Interleave mutant order in the designer",
)
@click.option(
    "--force-mutant-order-balance",
    default=False,
    type=bool,
    help="Force balance in mutant order in the designer",
)
@click.option(
    "--schedule",
    default=0,
    type=int,
    help="Schedule type for the designer (0: no schedule, 1: remove commonest mutation/position every p0/p1 iterations, 2: remove mutation/position if it appears more than p0/p1 times)",
)
@click.option(
    "--schedule-param", type=str, help="Parameters for the schedule (e.g., 'p0,p1')"
)
@click.option(
    "--objective-constraints",
    type=str,
    help="Objective constraints for the designer (e.g., 'constraint1,constraint2')",
)
@click.option(
    "--objective-constraints-param",
    type=str,
    help="Parameters for the objective constraints (e.g., 'param1,param2')",
)
@click.option(
    "--weighted-multi-objective",
    default=True,
    type=bool,
    help="Use weighted multi-objective optimization in the designer",
)
@click.option("--debug", default=0, type=int, help="Debug level")
@click.option(
    "--data-normalization",
    default=False,
    type=bool,
    help="Normalize data before running the designer",
)
@click.option(
    "--contact-graph-text-file",
    type=click.Path(exists=True),
    help="Path to precomputed contact graph text.",
)
@click.option(
    "--heavy-chain-id", type=str, default="H", help="Antibody heavy chain ID."
)
@click.option(
    "--light-chain-id", type=str, default="L", help="Antibody light chain ID."
)
@click.option(
    "--antigen-chain-id", type=str, default="A", help="Antigen chain ID."
)
@click.option(
    "--distance-threshold",
    type=float,
    default=7.0,
    help="Distance threshold for contacts.",
)
@click.option(
    "--candidates-csv",
    type=click.Path(exists=True),
    help="Optional CSV with a Mutation column for LLM proposals.",
)
@click.option("--llm-model", type=str, default="gpt-4o", help="LLM model name.")
@click.option("--llm-temperature", type=float, default=0.2, help="LLM temperature.")
@click.option(
    "--llm-max-tokens", type=int, default=1500, help="Max tokens for LLM."
)
@click.option(
    "--llm-max-mut-in-prompt",
    type=int,
    default=50,
    help="Maximum mutations to include in LLM prompt.",
)
@click.option(
    "--llm-max-edges-in-prompt",
    type=int,
    default=200,
    help="Maximum contact edges to include in LLM prompt.",
)
@click.option(
    "--llm-api-base",
    type=str,
    default=None,
    help="Override OPENAI_API_BASE for LiteLLM.",
)
@click.option(
    "--llm-output",
    type=click.Path(exists=False),
    default="llm_guidance.json",
    help="Path to write LLM guidance JSON.",
)
@click.option(
    "--llm-derived-score-column",
    type=str,
    default="llm_derived_score",
    help="Column name for derived scoring values.",
)
@click.option(
    "--llm-scores-output",
    type=click.Path(exists=False),
    default=None,
    help="Optional output path for scores after LLM scoring additions.",
)
def run_pipeline_with_llm(
    positions,
    sequence,
    pdb_path,
    plm_model_names,
    plm_model_paths,
    ifold_model_name,
    ifold_model_path,
    score_type,
    intermediate_output,
    plm_chain_type,
    plm_mask,
    plm_mapping,
    ifold_seed,
    nb_iterations,
    min_mut,
    max_mut,
    output_folder,
    forbidden_aa,
    max_arom_per_seq,
    dissimilarity_tolerance,
    interleave_mutant_order,
    force_mutant_order_balance,
    schedule,
    schedule_param,
    objective_constraints,
    objective_constraints_param,
    weighted_multi_objective,
    debug,
    data_normalization,
    contact_graph_text_file,
    heavy_chain_id,
    light_chain_id,
    antigen_chain_id,
    distance_threshold,
    candidates_csv,
    llm_model,
    llm_temperature,
    llm_max_tokens,
    llm_max_mut_in_prompt,
    llm_max_edges_in_prompt,
    llm_api_base,
    llm_output,
    llm_derived_score_column,
    llm_scores_output,
):
    """Run the pipeline with an LLM reasoning step between scoring and ILP."""

    logger.info("Starting the Protlib Designer pipeline with LLM reasoning...")

    sequence, positions = format_and_validate_pipeline_parameters(
        sequence,
        pdb_path,
        positions,
        plm_model_names,
        plm_model_paths,
        ifold_model_name,
        ifold_model_path,
    )

    dataframes = []

    if plm_model_names or plm_model_paths:
        logger.info("Running PLM Scorer...")
        logger.info(f"Sequence: {sequence}")
        logger.info(f"Positions: {positions}")

        append_plm_scores_to_dataframes(
            dataframes,
            sequence,
            positions,
            plm_model_names=plm_model_names,
            plm_model_paths=plm_model_paths,
            chain_type=plm_chain_type,
            score_type=score_type,
            mask=plm_mask,
            mapping=plm_mapping,
        )

        logger.info(f"PLM scoring completed with {len(dataframes)} models")

    if pdb_path:
        logger.info("Running IFOLD Scorer...")
        logger.info(f"PDB Path: {pdb_path}")
        logger.info(f"Positions: {positions}")

        if pdb_path.endswith(".cif"):
            logger.info("Converting CIF to PDB...")
            converted_pdb_path = pdb_path.replace(".cif", ".pdb")
            cif_to_pdb(pdb_path, output_pdb=converted_pdb_path)
            pdb_path = converted_pdb_path
            logger.info(f"Converted CIF to PDB: {pdb_path}")

        append_ifold_scores_to_dataframes(
            dataframes,
            pdb_path,
            positions,
            ifold_model_name=ifold_model_name,
            ifold_model_path=ifold_model_path,
            seed=ifold_seed,
            score_type=score_type,
        )
        logger.info(f"IFold scoring completed with {len(dataframes)} models")

    if not dataframes:
        logger.error("No scores were generated. Check your input parameters.")
        return

    logger.info("Combining scores from different models...")
    final_df = combine_dataframes(dataframes)
    if final_df is None:
        return
    logger.info("Scores combined successfully")

    final_df.to_csv(intermediate_output, index=False)
    logger.info(f"Combined scores saved to {intermediate_output}")

    contact_graph_text = _build_contact_graph_text(
        pdb_path,
        contact_graph_text_file,
        heavy_chain_id,
        light_chain_id,
        antigen_chain_id,
        distance_threshold,
        llm_max_edges_in_prompt,
    )

    scores_by_mutation = _scores_from_dataframe(final_df)
    mutation_proposals: List[str] = []
    if candidates_csv:
        mutation_proposals = _load_mutation_list_from_csv(candidates_csv)
    if not mutation_proposals:
        mutation_proposals = list(scores_by_mutation.keys())

    llm_config = LLMReasoningConfig(
        model=llm_model,
        temperature=llm_temperature,
        max_tokens=llm_max_tokens,
        max_mutations_in_prompt=llm_max_mut_in_prompt,
        max_contact_edges_in_prompt=llm_max_edges_in_prompt,
    )

    llm_output_data = run_llm_reasoning(
        contact_graph_text=contact_graph_text,
        scores_by_mutation=scores_by_mutation,
        mutation_proposals=mutation_proposals,
        config=llm_config,
        api_base=llm_api_base,
    )

    if llm_output:
        with open(llm_output, "w") as handle:
            json.dump(llm_output_data, handle, indent=2)
        logger.info(f"Saved LLM guidance to {llm_output}")

    constraints = _collect_llm_constraints(llm_output_data)
    derived_scoring = llm_output_data.get("derived_scoring_function", {})
    llm_scores_path = llm_scores_output or intermediate_output

    final_df, added_llm_score = _apply_llm_derived_scoring(
        final_df, derived_scoring, llm_derived_score_column
    )
    final_df.to_csv(llm_scores_path, index=False)
    if added_llm_score:
        logger.info(f"Scores with LLM-derived column saved to {llm_scores_path}")
    else:
        logger.info(f"Scores saved to {llm_scores_path} without LLM-derived column")

    logger.info("Running Protlib Designer...")

    config, _ = format_and_validate_protlib_designer_parameters(
        output_folder,
        llm_scores_path,
        min_mut,
        max_mut,
        nb_iterations,
        forbidden_aa,
        max_arom_per_seq,
        dissimilarity_tolerance,
        interleave_mutant_order,
        force_mutant_order_balance,
        schedule,
        schedule_param,
        objective_constraints,
        objective_constraints_param,
        weighted_multi_objective,
        debug,
        data_normalization,
    )
    if constraints:
        config["llm_constraints"] = constraints
    if llm_output:
        config["llm_guidance_path"] = llm_output
    if added_llm_score:
        config["llm_derived_score_column"] = llm_derived_score_column

    data_loader = DataLoader(llm_scores_path)
    data_loader.load_data()
    config = data_loader.update_config_with_data(config)

    output_path = Path(output_folder)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created directory {output_folder}")
    write_config(config, output_path)

    ilp_generator = ILPGenerator(data_loader, config)
    no_filter = NoFilter()

    generate_and_remove_solver = GenerateAndRemoveSolver(
        ilp_generator,
        no_filter,
        length_of_library=nb_iterations,
        maximum_number_of_iterations=2 * nb_iterations,
    )

    generate_and_remove_solver.run()

    solution_manager = SolutionManager(generate_and_remove_solver)
    solution_manager.process_solutions()
    solution_manager.output_results()

    logger.info("Pipeline with LLM completed successfully!")


if __name__ == "__main__":
    run_pipeline_with_llm()
