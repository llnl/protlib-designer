import torch
import pandas as pd
from typing import List

from transformers import AutoModelForMaskedLM, AutoTokenizer

from protlib_designer.utils import (
    amino_acids,
    is_sequence_and_wildtype_dict_consistent,
    validate_canonical_sequence_and_wildtypes,
)
from protlib_designer.scorer.scorer import (
    score_function,
    from_user_input_to_scorer_input,
    from_scorer_output_to_user_output,
    Scorer,
)


def load_huggingface_model(model_reference: str, device: torch.device):
    """
    Load the Hugging Face model and tokenizer.

    Parameters
    ----------
    model_reference : str
        Reference to the model, it can be a model name (e.g., 'Rostlab/prot_bert') or a path to an existing model.
    device : torch.device
        Device to load the model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_reference)
    model = AutoModelForMaskedLM.from_pretrained(
        model_reference, output_hidden_states=True
    )
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    model = model.eval()
    return model, tokenizer


class PLMScorer(Scorer):
    def __init__(
        self,
        model_name: str = "Rostlab/prot_bert",
        model_path: str = None,
        score_type: str = "minus_llr",
        mask: bool = True,
        mapping: dict = None,
    ):
        """Initialize the PLM Scorer.

        Parameters
        ----------
        model_name : str
            Name of the model to use. Defaults to 'Rostlab/prot_bert'.
        model_path : str
            Path to the model to use. If None, the model is loaded from the Hugging Face model hub using the model name.
        score_type : str
            Type of score to use. Options are 'll', 'llr', 'minus_llr', 'probs'.
        mask : bool
            If True: mask positions before passing to PLM, if False: do not mask positions before passing to PLM.
        mapping : dict
            A dictionary to map the positions from the user input to the numbering used by the scorer. Example:
            {'H': {'H1': 1, 'H5': 2, 'H6': 3}, 'L': {'L1': 1, 'L2': 2, 'L3': 3}}
        """

        if model_name is None and model_path is None:
            raise ValueError("Please provide a model name or a model path.")

        self.model_name = model_name
        self.model_path = model_path
        self.score_type = score_type
        self.mask = mask
        self.mapping = mapping

        # Set the device to GPU if available, otherwise use CPU.
        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Load the PLM model and tokenizer.
        self.model, self.tokenizer = self.load_model()

        # If the model is not ESM, use token type ids.
        self.use_token_type_ids = "esm" not in model_name.lower()

        # Get indices of the PLM token library from the tokenizer with respect to all amino acids.
        self.aa_token_indices = self.tokenizer.convert_tokens_to_ids(amino_acids)

    def load_model(self):
        """Load the PLM model and tokenizer.

        Parameters
        ----------
        model_name : str
            Name of the model.
        model_path : str
            Path to the model.
        device : torch.device
            Device to load the model.

        Returns
        -------
        model : torch.nn.Module
            The loaded model.
        tokenizer : transformers.AutoTokenizer
            The loaded tokenizer.
        """
        if self.model_path is not None:
            return load_huggingface_model(self.model_path, self.device)
        try:
            return load_huggingface_model(self.model_name, self.device)
        except Exception as e:
            raise ValueError(
                f"Model {self.model_name} not found in Hugging Face model hub. Please provide a valid model name."
            ) from e

    def prepare_input(self, sequence: str, positions: List[str], chain_type: str):
        """Prepare the input for the model.

        Parameters
        ----------
        sequence : str
            Sequence used to generated the scores. This a string of amino acids.
        positions : list
            Positions on the sequence to be used to generate the score.
            Positions must be in the following format: {WT}{CHAIN}{STRINGINDEX}.
            Note: STRINGINDEX is 1-indexed, that is, the first position is 1. For example, the first positions in
            the list of positions are [EH1, VH2, QH3, ...].
        chain_type : str
            Optional parameter to specify the chain type. For example, heavy or light chain for antibodies.

        Returns
        -------
        batch : list
            If mask is True, the batch is a list of copies of the sequence with the required position masked.
            If mask is False, the batch is a single copy of the sequence.
        chain_token : int
            Chain token to pass to the PLM model.
        wildtype : list
            List of wildtype amino acids for each position.
        """

        if self.mapping is not None:
            positions = from_user_input_to_scorer_input(positions, self.mapping)

        # Check that all positions have the same chain letter (2nd character).
        chain_letter = {position[1] for position in positions}
        if len(chain_letter) > 1:
            raise ValueError(
                "All positions must have the same chain letter. Please provide positions with the same chain type."
            )
        chain_letter = chain_letter.pop()

        # Get the positions indices.
        position_indices = [int(position[2:]) for position in positions]

        # Get wildtype dict: {position: wildtype}
        wildtype_dict = {int(position[2:]): position[0] for position in positions}

        validate_canonical_sequence_and_wildtypes(sequence, wildtype_dict)

        # Check if sequence and positions are consistent.
        if not is_sequence_and_wildtype_dict_consistent(sequence, wildtype_dict):
            raise ValueError(
                "The sequence and the wildtype dictionary are not consistent. "
                "Please check the sequence and the positions provided."
            )

        # Create batch to generate score in one pass.
        batch = []
        # Iterate over positions to apply mask token at each required position.
        if self.mask:
            # Create mask token.
            mask_token = self.tokenizer.mask_token
            for position_index in position_indices:
                # Convert sequence to list.
                sequence_list = list(sequence)
                # Mask the required position.
                # Subtract 1 from the position index to convert to 0-indexed.
                position_index = position_index - 1
                sequence_list[position_index] = mask_token
                # Append sequence to batch, and add whitespace to sequence for protbert tokenizer.
                batch.append(" ".join(sequence_list))
        else:
            sequence_list = list(sequence)
            batch.append(" ".join(sequence_list))

        chain_token = 1 if chain_type == "light" else 0

        return batch, wildtype_dict, position_indices, chain_letter, chain_token

    def forward_pass(self, batch, chain_token):
        """Perform the forward pass.

        Parameters
        ----------
        batch : list
            List of sequences to pass to the model.
        chain_token : int
            Chain token to pass to the PLM model.
        """
        input_ids = self.tokenizer(batch, padding=True)["input_ids"]
        input_ids = torch.tensor(input_ids, device=self.device)
        bz, seq_length = input_ids.shape
        token_type_ids = (
            torch.zeros(bz, seq_length).fill_(chain_token).to(self.device).long()
        )

        with torch.no_grad():
            if self.use_token_type_ids:
                logits = self.model(input_ids=input_ids, token_type_ids=token_type_ids)[
                    "logits"
                ]
            else:
                logits = self.model(input_ids=input_ids)["logits"]

        return logits

    def get_scores(self, sequence: str, positions: List[str], chain_type: str):
        """Compute the scores (in silico deep mutational scanning) for a given sequence and positions.

        Parameters
        ----------
        sequence : str
            Sequence used to generated the scores. This a string of amino acids.
        positions : list
            Positions on the sequence to be used to generate the score.
            Positions must be in the following format: {WT}{CHAIN}{STRINGINDEX}.
            Note: STRINGINDEX is 1-indexed, that is, the first position is 1. For example, the first positions in
            the list of positions are [EH1, VH2, QH3, ...].
        chain_type : str
            Type of antibody chain (heavy or light). This is used to determine the chain token to pass to the LLM model.
        """
        # Prepare the input for the model.
        (
            batch,
            wildtype_dict,
            position_indices,
            chain_letter,
            chain_token,
        ) = self.prepare_input(sequence, positions, chain_type)
        # Get the logits from the forward pass (shape: (batch_size, sequence_length, num_tokens)).
        logits = self.forward_pass(batch, chain_token)
        # Get AA tokens (shape: (batch_size, sequence_length, num_amino_acids)).
        logits = logits[:, :, self.aa_token_indices]
        # Apply softmax and take log over the logits (shape: (batch_size, sequence_length, num_amino_acids)).
        logps = torch.log_softmax(logits, dim=-1)
        # Create list to store elements for dataframe output
        mutation2score = {}
        # Iterate over each position and corresponding wildtype to get the scores.
        for batch_idx, position_index in enumerate(position_indices):
            batch_idx = batch_idx if self.mask else 0
            # Get the wildtype amino acid at the current position.
            wildtype_aa = wildtype_dict[position_index]
            # Index 0 of logits corresponds to CLS token, index 1 is where the sequence begins.
            sequence_index = position_index
            # Per position logp and probs (shape: (num_amino_acids)).
            position_logps = logps[batch_idx][sequence_index].cpu().numpy()
            # Get the wildtype logp.
            wildtype_aa_id = self.tokenizer.convert_tokens_to_ids(wildtype_aa)
            try:
                wildtype_aa_index = self.aa_token_indices.index(wildtype_aa_id)
            except ValueError as exc:
                raise ValueError(
                    f"Wildtype amino acid {wildtype_aa!r} at position {position_index} is not one of "
                    f"the {len(amino_acids)} canonical residues this scorer supports "
                    f"({''.join(amino_acids)}). Non-canonical residues (e.g. 'X') cannot be scored."
                ) from exc
            wildtype_aa_logp = position_logps[wildtype_aa_index]
            # Compute the scores.
            position_scores = list(
                score_function(
                    position_logps, wildtype_aa_logp, score_type=self.score_type
                )
            )
            # Update the mutation2score dictionary.
            for i, amino_acid in enumerate(amino_acids):
                mutation = f"{wildtype_aa}{chain_letter}{position_index}{amino_acid}"
                if self.mapping is not None:
                    mutation = from_scorer_output_to_user_output(mutation, self.mapping)
                mutation2score[mutation] = position_scores[i]

        # create a dataframe from the mutation2score dictionary. columns: Mutation, score
        return pd.DataFrame(
            mutation2score.items(),
            columns=["Mutation", f"{self.model_name}_{self.score_type}"],
        )

    def __str__(self):
        return super().__str__() + ": PLM Scorer"
