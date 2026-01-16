import sys
import json
import numpy as np
import torch
import random
import os.path
from pathlib import Path

from protlib_designer.scorer.pmpnn.utils import parse_PDB
from protlib_designer.scorer.pmpnn.model import ProteinMPNN
from protlib_designer.scorer.pmpnn.protein import Protein
from protlib_designer.scorer.pmpnn.dataset import (
    StructureDataset,
    StructureDatasetPDB,
)


class ProteinMPNNRunner:
    def __init__(
        self,
        seed: int | None = None,
        model_weights_path: str | None = None,
        use_soluble_model: bool = False,
        ca_only: bool = False,
        model_name: str = "v_48_020",
        num_seq_per_target: int = 1,
        batch_size: int = 1,
        max_sequence_length: int = 20000,
        sampling_temp: float = 0.1,
        omit_AAs_list: list | None = None,
        backbone_noise: float = 0.0,
        pssm_threshold: float = 0.0,
        chain_id_jsonl: str | None | dict = None,
        fixed_positions_jsonl: str | None | dict = None,
        pssm_jsonl: str | None | dict = None,
        omit_AA_jsonl: str | None | dict = None,
        bias_AA_jsonl: str | None | dict = None,
        tied_positions_jsonl: str | None | dict = None,
        bias_by_res_jsonl: str | None | dict = None,
        pdb_path: str | None | dict = None,
        pdb_path_chains: str | None | dict = None,
    ):
        if seed is None:
            seed = int(np.random.randint(0, high=999, size=1, dtype=int)[0])
        else:
            seed = seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed

        self.hidden_dim = 128
        self.num_layers = 3
        if model_weights_path:
            model_folder_path = model_weights_path
            if model_folder_path[-1] != "/":
                model_folder_path = f"{model_folder_path}/"
        else:
            file_path = os.path.realpath(__file__)
            k = file_path.rfind("/")
            if ca_only:
                print("Using CA-ProteinMPNN!")
                model_folder_path = f"{file_path[:k]}/weights/ca_model_weights/"
                if use_soluble_model:
                    print("WARNING: CA-SolubleMPNN is not available yet")
                    sys.exit()
            elif use_soluble_model:
                print("Using ProteinMPNN trained on soluble proteins only!")
                model_folder_path = f"{file_path[:k]}/weights/soluble_model_weights/"
            else:
                model_folder_path = f"{file_path[:k]}/weights/vanilla_model_weights/"
        checkpoint_path = f"{model_folder_path}{model_name}.pt"

        self.num_batches = num_seq_per_target // batch_size
        self.batch_copies = batch_size
        self.alphabet = "ACDEFGHIKLMNPQRSTVWYX"
        self.alphabet_dict = dict(zip(self.alphabet, range(21)))
        # print_all = suppress_print == 0
        if omit_AAs_list is None:
            omit_AAs_list = []
        self.omit_AAs_np = np.array(
            [AA in omit_AAs_list for AA in self.alphabet]
        ).astype(np.float32)
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.pssm_threshold = pssm_threshold

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = ProteinMPNN(
            num_letters=21,
            node_features=self.hidden_dim,
            edge_features=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_encoder_layers=self.num_layers,
            num_decoder_layers=self.num_layers,
            augment_eps=backbone_noise,
            ca_only=ca_only,
        )
        self.model.to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def load_data(
        self,
        chain_id_jsonl: str | None | dict = None,
        fixed_positions_jsonl: str | None | dict = None,
        pssm_jsonl: str | None | dict = None,
        omit_AA_jsonl: str | None | dict = None,
        bias_AA_jsonl: str | None | dict = None,
        tied_positions_jsonl: str | None | dict = None,
        bias_by_res_jsonl: str | None | dict = None,
        pdb_path: str | None | dict = None,
        pdb_path_chains: str | None | dict = None,
        jsonl_path: str | None = None,
    ):
        if os.path.isfile(chain_id_jsonl):
            with open(chain_id_jsonl, "r") as json_file:
                json_list = list(json_file)
            for json_str in json_list:
                chain_id_dict = json.loads(json_str)
        elif type(chain_id_jsonl) is dict:
            chain_id_dict = chain_id_jsonl
        else:
            chain_id_dict = None
        self.chain_id_dict = chain_id_dict

        if os.path.isfile(fixed_positions_jsonl):
            with open(fixed_positions_jsonl, "r") as json_file:
                json_list = list(json_file)
            for json_str in json_list:
                fixed_positions_dict = json.loads(json_str)
        elif type(fixed_positions_jsonl) is dict:
            fixed_positions_dict = fixed_positions_jsonl
        else:
            fixed_positions_dict = None
        self.fixed_positions_dict = fixed_positions_dict

        if os.path.isfile(pssm_jsonl):
            with open(pssm_jsonl, "r") as json_file:
                json_list = list(json_file)
            pssm_dict = {}
            for json_str in json_list:
                pssm_dict |= json.loads(json_str)
        elif type(pssm_jsonl) is dict:
            pssm_dict = pssm_jsonl
        else:
            pssm_dict = None
        self.pssm_dict = pssm_dict

        if os.path.isfile(omit_AA_jsonl):
            with open(omit_AA_jsonl, "r") as json_file:
                json_list = list(json_file)
            for json_str in json_list:
                omit_AA_dict = json.loads(json_str)
        else:
            omit_AA_dict = None
        self.omit_AA_dict = omit_AA_dict

        if os.path.isfile(bias_AA_jsonl):
            with open(bias_AA_jsonl, "r") as json_file:
                json_list = list(json_file)
            for json_str in json_list:
                bias_AA_dict = json.loads(json_str)
        elif type(bias_AA_jsonl) is dict:
            bias_AA_dict = bias_AA_jsonl
        else:
            bias_AA_dict = None
        self.bias_AA_dict = bias_AA_dict

        if os.path.isfile(tied_positions_jsonl):
            with open(tied_positions_jsonl, "r") as json_file:
                json_list = list(json_file)
            for json_str in json_list:
                tied_positions_dict = json.loads(json_str)
        elif type(tied_positions_jsonl) is dict:
            tied_positions_dict = tied_positions_jsonl
        else:
            tied_positions_dict = None
        self.tied_positions_dict = tied_positions_dict

        if os.path.isfile(bias_by_res_jsonl):
            with open(bias_by_res_jsonl, "r") as json_file:
                json_list = list(json_file)

            for json_str in json_list:
                bias_by_res_dict = json.loads(json_str)
        elif type(bias_by_res_jsonl) is dict:
            bias_by_res_dict = bias_by_res_jsonl
        else:
            bias_by_res_dict = None
        self.bias_by_res_dict = bias_by_res_dict

        bias_AAs_np = np.zeros(len(self.alphabet))
        if bias_AA_dict:
            for n, AA in enumerate(self.alphabet):
                if AA in list(bias_AA_dict.keys()):
                    bias_AAs_np[n] = bias_AA_dict[AA]
        return (
            self._extracted_from_load_data_97(pdb_path, pdb_path_chains)
            if pdb_path
            else StructureDataset(
                jsonl_path,
                truncate=None,
                max_length=self.max_length,
            )
        )

    # TODO Rename this here and in `load_data`
    def _extracted_from_load_data_97(self, pdb_path, pdb_path_chains):
        pdb_dict_list = parse_PDB(pdb_path, ca_only=self.ca_only)
        return StructureDatasetPDB(
            pdb_dict_list, truncate=None, max_length=self.max_length
        )

    def llr_score(
        self,
        wildtype_protein: str | Path,
        mutant_protein: str | Path,
        mut_position: int = None,
        aa_index: int = None,
        mask_chains: list[str] | None = None,
        visible_chains: list[str] | None = None,
        name: str | None = None,
    ):
        if mut_position is not None:
            mut_position = torch.tensor(mut_position, dtype=torch.int32)
        if aa_index is not None:
            aa_index = torch.tensor(aa_index, dtype=torch.int32)

        if mask_chains is not None and visible_chains is not None:
            name = wildtype_protein[-8:-4] if name is None else name
            chain_id_dict = {name: [mask_chains, visible_chains]}
        else:
            chain_id_dict = None

        wildtype = Protein.from_pdb(wildtype_protein, self.device, chain_id_dict)
        randn = torch.ones(wildtype.chain_M.shape, device=wildtype.device)
        wildtype_logps, wt_mask = self.forward(wildtype, randn=randn)
        wildtype_logps *= wt_mask
        if mut_position:
            wildtype_logps = wildtype_logps[:, mut_position, :].unsqueeze(1)
        if aa_index:
            wildtype_logps = wildtype_logps[:, :, aa_index]
        wildtype_logps = wildtype_logps.squeeze()

        # for ix, protein in enumerate(designed_proteins):
        llrs = []
        for _ in [mutant_protein] * 10:
            prot = Protein.from_pdb(mutant_protein, self.device, chain_id_dict)
            prot_logps, prot_mask = self.forward(prot, randn=randn)
            prot_logps *= prot_mask
            if mut_position:
                prot_logps = prot_logps[:, mut_position, :].unsqueeze(1)
            if aa_index:
                prot_logps = prot_logps[:, :, aa_index]
            prot_logps = prot_logps.squeeze()
            llr = prot_logps - wildtype_logps
            llrs.append(llr.detach().cpu().numpy())
        return np.mean(llrs)

    def forward(self, protein: Protein, randn: torch.Tensor | None = None):
        """Perform a forward pass through the model."""
        with torch.no_grad():
            mask = protein.mask * protein.chain_M * protein.chain_M_pos
            mask = mask.unsqueeze(-1)
            mask = mask.expand(-1, -1, 21)
            if randn is not None:
                randn = torch.randn(protein.chain_M.shape, device=protein.device)
            logps = self.model(
                protein.structure_feats,
                protein.sequence_feats,
                protein.mask,
                protein.chain_M * protein.chain_M_pos,
                protein.residue_idx,
                protein.chain_encoding_all,
                randn,
            )
        return logps, mask

    def conditional_probs(self, protein: Protein, randn_1: torch.Tensor | None = None):
        if randn_1 is None:
            randn_1 = torch.randn(
                protein.chain_M.shape, device=protein.structure_feats.device
            )
        log_probs = self.model.conditional_probs(
            protein.structure_feats,
            protein.sequence_feats,
            protein.mask,
            protein.chain_M * protein.chain_M_pos,
            protein.residue_idx,
            protein.chain_encoding_all,
            randn_1,
            backbone_only=False,
        )
        S = protein.sequence_feats[0,]
        return log_probs, S

    def unconditional_probs(self, protein: Protein):
        log_probs = self.model.unconditional_probs(
            protein.structure_feats,
            protein.mask,
            protein.residue_idx,
            protein.chain_encoding_all,
        )
        S = protein.sequence_feats[0,]
        return log_probs, S

    def get_probabilities(
        self,
        pdb: str | Protein,
        fixed_chain_dict: dict | None = None,
        fixed_positions_dict: dict | None = None,
        device: torch.device | str = "cuda",
        conditional_probs: bool = True,
        randn_1: torch.Tensor | None = None,
    ):
        if type(pdb) is str:
            protein = Protein.from_pdb(
                pdb,
                device=device,
                chain_id_dict=fixed_chain_dict,
                fixed_positions_dict=fixed_positions_dict,
            )
        else:
            protein = pdb
        with torch.no_grad():
            if conditional_probs:
                log_probs, S = self.conditional_probs(protein, randn_1)
            else:
                log_probs, S = self.unconditional_probs(protein)
            design_mask = (protein.chain_M * protein.chain_M_pos * protein.mask)[0,]
            chain_order = protein.chain_list_list

        return log_probs, S, protein.mask, design_mask, chain_order
