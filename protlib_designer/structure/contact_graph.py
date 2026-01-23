from typing import Dict, List, Tuple

from Bio.PDB import NeighborSearch, PDBParser
from Bio.PDB.Polypeptide import is_aa

from protlib_designer import logger
from protlib_designer.structure.interface_profile import map_residue_name_to_letter


def _format_residue_id(residue) -> str:
    """Format a residue as WT+Chain+Position with insertion codes when present."""
    chain_id = residue.get_parent().id
    resseq = residue.id[1]
    icode = residue.id[2].strip()
    wt = map_residue_name_to_letter(residue.get_resname())
    return f"{wt}{chain_id}{resseq}{icode}" if icode else f"{wt}{chain_id}{resseq}"


def compute_contact_edges(
    pdb_file: str,
    heavy_chain_id: str,
    light_chain_id: str,
    antigen_chain_id: str,
    distance_threshold: float = 7.0,
) -> List[Dict[str, float]]:
    """Compute antibody-antigen contact edges from a PDB file.

    Contacts are defined using the minimum atom-atom distance between residues.
    Only standard amino acids are considered.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)
    model = next(structure.get_models())

    chain_map = {chain.id: chain for chain in model}

    def _require_chain(chain_id: str, label: str):
        chain = chain_map.get(chain_id)
        if chain is None:
            raise ValueError(
                f"Chain '{chain_id}' ({label}) not found in PDB file: {pdb_file}"
            )
        return chain

    ab_chain_ids = [cid for cid in [heavy_chain_id, light_chain_id] if cid]
    if not ab_chain_ids:
        raise ValueError("At least one antibody chain ID must be provided.")

    ab_residues = []
    for chain_id in ab_chain_ids:
        chain = _require_chain(chain_id, "antibody")
        ab_residues.extend(
            [res for res in chain.get_residues() if is_aa(res, standard=True)]
        )

    antigen_chain = _require_chain(antigen_chain_id, "antigen")
    ag_residues = [
        res for res in antigen_chain.get_residues() if is_aa(res, standard=True)
    ]

    if not ab_residues:
        logger.warning("No antibody residues detected; returning empty contact list.")
        return []
    if not ag_residues:
        logger.warning("No antigen residues detected; returning empty contact list.")
        return []

    ag_atoms = [atom for res in ag_residues for atom in res.get_atoms()]
    if not ag_atoms:
        logger.warning("No antigen atoms detected; returning empty contact list.")
        return []

    neighbor_search = NeighborSearch(ag_atoms)
    contacts: Dict[Tuple[str, str], float] = {}

    for ab_res in ab_residues:
        ab_label = _format_residue_id(ab_res)
        for ab_atom in ab_res.get_atoms():
            close_atoms = neighbor_search.search(
                ab_atom.coord, distance_threshold, level="A"
            )
            for ag_atom in close_atoms:
                ag_res = ag_atom.get_parent()
                ag_label = _format_residue_id(ag_res)
                distance = float(ab_atom - ag_atom)
                key = (ab_label, ag_label)
                previous = contacts.get(key)
                if previous is None or distance < previous:
                    contacts[key] = distance

    edges = [
        {"ab_res": ab_res, "ag_res": ag_res, "distance": dist}
        for (ab_res, ag_res), dist in contacts.items()
    ]
    edges.sort(key=lambda item: (item["ab_res"], item["ag_res"]))
    return edges
