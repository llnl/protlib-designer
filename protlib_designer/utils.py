import json
import sys
from pathlib import Path

import pandas as pd

from protlib_designer import logger

amino_acids = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]

aromatic_amino_acids = [
    "F",
    "Y",
    "W",
]


def format_and_validate_protlib_designer_parameters(
    output_folder,
    data,
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
):
    """Format and validate the parameters.

    Parameters
    ----------
    data : str
        The path to the data file.
    nb_iterations : int
        The number of iterations to run.
    forbidden_aa : str
        The forbidden amino acids.
    max_arom_per_seq : int
        The maximum number of aromatic amino acids per sequence.
    schedule : int
        The schedule to use.
    schedule_param : str
        The scheduling parameters.
    objective_constraints : str
        The objective constraints.
    objective_constraints_param : str
        The objective constraints parameters.
    """
    if nb_iterations > 5000:
        logger.warning("Maximum number of iterations is 5000. Setting to 5000.")
        nb_iterations = 5000

    if forbidden_aa is not None:
        forbidden_aa = [x.strip() for x in forbidden_aa.split(",")]
        logger.info(f"Forbidden amino acids: {forbidden_aa}")
    else:
        forbidden_aa = []

    if max_arom_per_seq is not None and max_arom_per_seq < 0:
        logger.error("max_arom_per_seq must be a positive integer.")
        exit()

    if (
        schedule_param is not None
        and not str(schedule_param).strip()
        or schedule_param is None
    ):
        schedule_param = []
    else:
        schedule_param = [
            int(val) for val in schedule_param.split(",") if val.strip()
        ]
    validate_schedule_parameters(schedule, schedule_param)

    if (
        objective_constraints is not None
        and not str(objective_constraints).strip()
        or objective_constraints is None
    ):
        objective_constraints = []
    else:
        objective_constraints = [
            val for val in objective_constraints.split(",") if val.strip()
        ]
    if (
        objective_constraints_param is not None
        and not str(objective_constraints_param).strip()
        or objective_constraints_param is None
    ):
        objective_constraints_param = []
    else:
        objective_constraints_param = [
            float(val)
            for val in objective_constraints_param.split(",")
            if val.strip()
        ]
    validate_objective_constraints(objective_constraints, objective_constraints_param)

    data_df = pd.read_csv(data)
    validate_data(data_df)

    return {
        "output_folder": output_folder,
        "data": data,
        "min_mut": min_mut,
        "max_mut": max_mut,
        "nb_iterations": nb_iterations,
        "forbidden_aa": forbidden_aa,
        "max_arom_per_seq": max_arom_per_seq,
        "dissimilarity_tolerance": dissimilarity_tolerance,
        "interleave_mutant_order": interleave_mutant_order,
        "force_mutant_order_balance": force_mutant_order_balance,
        "schedule": schedule,
        "schedule_param": schedule_param,
        "objective_constraints": objective_constraints,
        "objective_constraints_param": objective_constraints_param,
        "weighted_multi_objective": weighted_multi_objective,
        "debug": debug,
        "data_normalization": data_normalization,
    }, data_df


def validate_data(df: pd.DataFrame):
    """Validate the data file. The data file must have the following columns:
    Mutation, Target1, Target2, ..., TargetN

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data.
    """
    if "Mutation" not in df.columns:
        logger.error("Data file must have a Mutation column.")
        sys.exit(2)

    if len(df.columns) < 2:
        logger.error(
            "Data file must have at minimum the Mutation column and at least one Objective/Target."
        )
        sys.exit(3)


def validate_schedule_parameters(schedule: int, schedule_param: list):
    """Validate the scheduling parameters.

    Parameters
    ----------
    schedule : int
        The schedule to use.
    schedule_param : list
        The scheduling parameters.
    """
    if schedule == 0:
        if schedule_param:
            logger.warning(
                "Scheduling = 0 ignores schedule-param values; dropping them."
            )
        return

    if schedule == 1 and len(schedule_param) != 2:
        logger.error(
            "Scheduling = 1 need 2 parameters in the form 'p0,p1'. p0 : Number of iterations to kill the commonest \
                mutation. p1 : Number of iterations to kill the commonest positions."
        )
        exit()

    if schedule == 2 and len(schedule_param) != 2:
        logger.error(
            "Scheduling = 2 need 2 parameters in the form 'p0,p1'. p0 : Number of occurrences of mutation to kill it. \
                p1 : Number of occurrences of position to kill it"
        )
        exit()


def validate_objective_constraints(
    objectives_constraints: list, objectives_constraints_param: list
):
    """Validate the objective constraints.

    Parameters
    ----------
    objectives_constraints : list
        The list of objective constraints.
    objectives_constraints_param : list
        The list of parameters for the objective constraints.
    """
    if len(objectives_constraints) != len(objectives_constraints_param):
        logger.error(
            "Number of objective constraint parameters does not match number of objective constraints"
        )
        exit()


def parse_mutation(mutation: str):
    """Parse wild-type, position, mutated amino acid information
    from compact mutation notation.

    Parameters
    ----------
    mutation : str

    Returns
    -------
    Tuple
        (WT, pos, mutation) as str values.

    """
    return mutation[0], mutation[1:-1], mutation[-1]


def extract_mutation_key(mutationbreak: str):
    """Extract the mutation key from the mutationbreak string.

    Parameters
    ----------
    mutationbreak : str
        The mutationbreak string.
    """
    chars = list(mutationbreak)  # Convert string to list of chars.
    return f"{chars[1]}_{chars[2]}"


def write_config(config: dict, output_dir: Path):
    """Write the parameters used to execute the program to disk.

    Parameters
    ----------
    config : dict
        The parameters used to execute the program.
    output_dir : str
        The output directory to write the config file to.
    """
    filepath = output_dir / "config.json"

    with open(filepath, "w") as fp:
        json.dump(config, fp, indent=4)


def cif_to_pdb(cif_file: str, save: bool = True, output_pdb: str = None):
    """Convert a CIF file to a PDB file

    Parameters
    ----------
    cif_file : str
        Path to the CIF file
    save : bool, optional
        Whether to save the PDB file, by default True
    output_pdb : str, optional
        Path to save the PDB file, by default None
        If None, the PDB file will be saved with the same name as the CIF file
        with a .pdb extension.
    """

    from Bio.PDB import MMCIFParser, PDBIO

    parser = MMCIFParser()
    structure = parser.get_structure("structure", cif_file)
    io = PDBIO()
    io.set_structure(structure)
    if save:
        pdb_file = (
            cif_file.replace(".cif", ".pdb") if output_pdb is None else output_pdb
        )
        io.save(pdb_file)
    return io


def extract_sequence_from_pdb(pdb_file: str, chain_id: str = "A") -> str:
    """Extract the sequence from a PDB file.

    Parameters
    ----------
    pdb_file : str
        Path to the PDB file.
    chain_id : str, optional
        Chain ID to extract the sequence from, by default "A".

    Returns
    -------
    str
        The sequence of the specified chain.
    """

    from Bio.PDB import PDBParser, PPBuilder

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("X", pdb_file)
    chain = struct[0][chain_id]

    builder = PPBuilder()
    # This returns a list of Polypeptide objects (fragments if there are breaks)
    peptides = builder.build_peptides(chain)

    # If you want the full chain sequence as one string, just concatenate:
    return ''.join(str(pep.get_sequence()) for pep in peptides)


def is_sequence_and_wildtype_dict_consistent(
    sequence: str, wildtype_dict: dict
) -> bool:
    """
    Check if the positions list is consistent with the sequence.
    :param sequence: The amino acid sequence of the chain.
    :param wildtype_dict: Dictionary containing positions and their corresponding wildtype amino acids, e.g. {1: 'A', 2: 'R', ...}.
    :return: True if all positions are consistent with the sequence, False otherwise.
    """
    for position_index, wildtype_aa in wildtype_dict.items():
        if position_index < 1 or position_index > len(sequence):
            raise ValueError(
                f"Position index {position_index} is out of bounds for the sequence of length {len(sequence)}."
            )
        if sequence[position_index - 1] != wildtype_aa:
            raise ValueError(
                f"Wildtype amino acid {wildtype_aa} at position {position_index} does not match the sequence at that position: {sequence[position_index - 1]}."
            )
    return True
