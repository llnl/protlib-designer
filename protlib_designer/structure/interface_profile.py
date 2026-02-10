from collections import defaultdict
from math import sqrt
from typing import Dict, Iterable, List, Optional, Tuple

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

from protlib_designer import logger

_RESIDUE_MAP = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
    "ASX": "B",
    "GLX": "Z",
    "UNK": "X",
}


def map_residue_name_to_letter(resname: str) -> str:
    return _RESIDUE_MAP.get(resname.upper(), "X")


def calculate_distance(atom1, atom2) -> float:
    diff = atom1.coord - atom2.coord
    return sqrt(sum(diff**2))


def _iter_residues(chain) -> List[object]:
    return [res for res in chain.get_residues() if is_aa(res, standard=True)]


def _format_residue_label(
    residue, chain_id_map: Optional[Dict[str, str]] = None
) -> str:
    chain_id = residue.get_parent().get_id()
    if chain_id_map and chain_id in chain_id_map:
        chain_id = chain_id_map[chain_id]
    res_id = residue.get_id()[1]
    insertion = residue.get_id()[2].strip()
    wt = map_residue_name_to_letter(residue.get_resname())
    if insertion:
        return f"{wt}{chain_id}{res_id}{insertion}"
    return f"{wt}{chain_id}{res_id}"


def find_hydrogen_bonds(residues1, residues2, threshold: float = 3.5):
    hydrogen_bonds = []
    for res1 in residues1:
        for atom1 in res1:
            if "H" not in atom1.get_id():
                continue
            for res2 in residues2:
                for atom2 in res2:
                    atom2_id = atom2.get_id()
                    if "O" not in atom2_id and "N" not in atom2_id:
                        continue
                    if calculate_distance(atom1, atom2) <= threshold:
                        hydrogen_bonds.append((res1, res2))
    return hydrogen_bonds


def find_salt_bridges(residues1, residues2, threshold: float = 4.0):
    charged_atoms = {
        "ARG": ["NH1", "NH2"],
        "LYS": ["NZ"],
        "ASP": ["OD1", "OD2"],
        "GLU": ["OE1", "OE2"],
    }
    salt_bridges = []
    for res1 in residues1:
        if res1.get_resname() not in charged_atoms:
            continue
        for atom1_name in charged_atoms[res1.get_resname()]:
            if atom1_name not in res1:
                continue
            for res2 in residues2:
                if res2.get_resname() not in charged_atoms:
                    continue
                for atom2_name in charged_atoms[res2.get_resname()]:
                    if atom2_name not in res2:
                        continue
                    if (
                        calculate_distance(res1[atom1_name], res2[atom2_name])
                        <= threshold
                    ):
                        salt_bridges.append((res1, res2))
    return salt_bridges


def find_hydrophobic_contacts(residues1, residues2, threshold: float = 5.0):
    hydrophobic_residues = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO"}
    hydrophobic_contacts = []
    for res1 in residues1:
        if res1.get_resname() not in hydrophobic_residues:
            continue
        for res2 in residues2:
            if res2.get_resname() not in hydrophobic_residues:
                continue
            for atom1 in res1:
                for atom2 in res2:
                    if calculate_distance(atom1, atom2) <= threshold:
                        hydrophobic_contacts.append((res1, res2))
                        break
    return hydrophobic_contacts


def _profile_pairwise_interactions(residues1, residues2, chain_id_map=None):
    h_bonds = find_hydrogen_bonds(residues1, residues2)
    s_bridges = find_salt_bridges(residues1, residues2)
    h_contacts = find_hydrophobic_contacts(residues1, residues2)

    def _format_pairs(pairs: Iterable[Tuple[object, object]]):
        formatted = [
            (
                _format_residue_label(left, chain_id_map=chain_id_map),
                _format_residue_label(right, chain_id_map=chain_id_map),
            )
            for left, right in pairs
        ]
        formatted.sort()
        return formatted

    return {
        "counts": {
            "H-bonds": len(h_bonds),
            "Salt-bridges": len(s_bridges),
            "Hydrophobic": len(h_contacts),
        },
        "pairs": {
            "H-bonds": _format_pairs(h_bonds),
            "Salt-bridges": _format_pairs(s_bridges),
            "Hydrophobic": _format_pairs(h_contacts),
        },
    }


def profile_antibody_antigen_interactions(
    pdb_file: str,
    heavy_chain_id: str,
    light_chain_id: str,
    antigen_chain_id: str,
    chain_id_map: Optional[Dict[str, str]] = None,
):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_file)
    model = next(structure.get_models())

    chain_map = {chain.id: chain for chain in model}
    for chain_id, label in (
        (heavy_chain_id, "heavy"),
        (light_chain_id, "light"),
        (antigen_chain_id, "antigen"),
    ):
        if chain_id not in chain_map:
            raise ValueError(f"Chain {chain_id} ({label}) not found in {pdb_file}")

    heavy_residues = _iter_residues(chain_map[heavy_chain_id])
    light_residues = _iter_residues(chain_map[light_chain_id])
    antigen_residues = _iter_residues(chain_map[antigen_chain_id])

    if not antigen_residues:
        logger.warning("No antigen residues detected for interaction profiling.")

    return {
        "heavy_antigen": _profile_pairwise_interactions(
            heavy_residues, antigen_residues, chain_id_map=chain_id_map
        ),
        "light_antigen": _profile_pairwise_interactions(
            light_residues, antigen_residues, chain_id_map=chain_id_map
        ),
    }


def compute_contact_metrics(edges: List[Dict[str, object]]) -> Dict[str, object]:
    degrees: Dict[str, int] = defaultdict(int)
    min_distance: Dict[str, float] = {}
    for edge in edges:
        ab_res = edge.get("ab_res")
        if not isinstance(ab_res, str):
            continue
        degrees[ab_res] += 1
        dist = edge.get("distance")
        if isinstance(dist, (int, float)) and (
            ab_res not in min_distance or dist < min_distance[ab_res]
        ):
            min_distance[ab_res] = float(dist)
    return {
        "total_edges": len(edges),
        "contact_degrees": dict(degrees),
        "contact_min_distance": min_distance,
    }


def compute_interaction_counts(profile: Dict[str, dict]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"H-bonds": 0, "Salt-bridges": 0, "Hydrophobic": 0}
    )
    for key in ("heavy_antigen", "light_antigen"):
        pairs = profile.get(key, {}).get("pairs", {})
        for interaction_type in ("H-bonds", "Salt-bridges", "Hydrophobic"):
            seen = set(pairs.get(interaction_type, []))
            for left, _ in seen:
                counts[left][interaction_type] += 1
    return dict(counts)


def build_interaction_profile_text(
    profile: Dict[str, dict], max_pairs: int = 12
) -> List[str]:
    lines = ["Interaction profile (H-bonds / salt bridges / hydrophobic):"]
    for key, label in (
        ("heavy_antigen", "Heavy-Antigen"),
        ("light_antigen", "Light-Antigen"),
    ):
        data = profile.get(key, {})
        counts = data.get("counts", {})
        pairs = data.get("pairs", {})
        lines.append(
            f"- {label}: H-bonds={counts.get('H-bonds', 0)}, "
            f"Salt-bridges={counts.get('Salt-bridges', 0)}, "
            f"Hydrophobic={counts.get('Hydrophobic', 0)}"
        )
        for interaction_type in ("H-bonds", "Salt-bridges", "Hydrophobic"):
            interaction_pairs = pairs.get(interaction_type, [])
            if not interaction_pairs:
                continue
            lines.append(f"  {interaction_type}:")
            lines.extend(
                f"    {left} -- {right}"
                for left, right in interaction_pairs[:max_pairs]
            )
            if len(interaction_pairs) > max_pairs:
                lines.append(
                    f"    ... truncated {len(interaction_pairs) - max_pairs} pairs"
                )
    return lines


def build_interface_profile_text(
    edges: List[Dict[str, object]],
    interaction_profile: Dict[str, dict],
    max_pairs: int = 12,
    max_contact_residues: int = 10,
    max_residue_interactions: int = 10,
) -> str:
    if not edges and not interaction_profile:
        return "Interface profile: NONE"

    lines = ["Interface profile summary:"]
    if edges:
        metrics = compute_contact_metrics(edges)
        lines.append(f"- total_contact_edges: {metrics['total_edges']}")
        if degrees := metrics.get("contact_degrees", {}):
            top = sorted(degrees.items(), key=lambda item: (-item[1], item[0]))[
                :max_contact_residues
            ]
            formatted = []
            for residue, count in top:
                min_dist = metrics.get("contact_min_distance", {}).get(residue)
                if min_dist is None:
                    formatted.append(f"{residue}(deg={count})")
                else:
                    formatted.append(f"{residue}(deg={count},min_dist={min_dist:.2f})")
            lines.append("- top_contact_residues: " + ", ".join(formatted))

    if interaction_profile:
        lines.extend(build_interaction_profile_text(interaction_profile, max_pairs))
        if interaction_counts := compute_interaction_counts(interaction_profile):
            lines.append("Residue interaction counts (ab side):")
            sorted_residues = sorted(
                interaction_counts.items(),
                key=lambda item: (-sum(item[1].values()), item[0]),
            )[:max_residue_interactions]
            lines.extend(
                f"- {residue}: H-bonds={counts.get('H-bonds', 0)}, Salt-bridges={counts.get('Salt-bridges', 0)}, Hydrophobic={counts.get('Hydrophobic', 0)}"
                for residue, counts in sorted_residues
            )
    return "\n".join(lines)
