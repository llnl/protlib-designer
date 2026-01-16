import time
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pulp
from numpy.linalg import matrix_rank, svd

from protlib_designer import logger
from protlib_designer.generator.generator import Generator
from protlib_designer.utils import amino_acids, aromatic_amino_acids, parse_mutation

# Ignore UserWarnings from pulp
warnings.filterwarnings("ignore", category=UserWarning, module="pulp")


class ILPGenerator(Generator):
    def __init__(self, data_loader, config):
        self.data_loader = data_loader
        self.config = config

        self.x_vars = []  # All the x variables.
        self.x_vars_dict = {}
        self.aromatic_vars = []  # The aromatic variables.
        self.aromatic_vars_dict = {}
        self.forbidden_vars = []  # The forbidden variables.
        self.forbidden_vars_dict = {}
        self.missing_vars = []  # The missing variables.
        self.missing_vars_dict = {}
        self.zero_enforced_mutations = []

        self.A = None
        self.A_original = None
        self.targets = data_loader.targets

        self.output_folder = self.config.get("output_folder", ".")
        self.min_mut = self.config.get("min_mut", 0)
        self.max_mut = self.config.get("max_mut", 1000)
        self.forbidden_aa = self.config.get("forbidden_aa", [])
        self.max_arom_per_seq = self.config.get("max_arom_per_seq", None)
        self.dissimilarity_tolerance = self.config.get("dissimilarity_tolerance", 0)
        self.interleave_mutant_order = self.config.get("interleave_mutant_order", False)
        self.force_mutant_order_balance = self.config.get(
            "force_mutant_order_balance", False
        )
        self.schedule = self.config.get("schedule", [])
        self.schedule_param = self.config.get("schedule_param", [])
        self.objective_constraints = self.config.get("objective_constraints", [])
        self.objective_constraints_param = self.config.get(
            "objective_constraints_param", []
        )
        self.debug = self.config.get("debug", 0)
        self.weighted_multi_objective = self.config.get(
            "weighted_multi_objective", False
        )
        self.debug = self.config.get("debug", 0)
        self.data_normalization = self.config.get("data_normalization", False)
        self.llm_constraints = self.config.get("llm_constraints", [])

        self.many_hot_encoded_solutions = []
        self.list_of_solution_dicts = []
        self.cpu_times = []
        self.positions_in_solution_counts = {}
        self.mutations_in_solution_counts = {}

        self._prepare_problem_and_solver()
        self._prepare_variables_and_zero_pad_matrix()
        self._prepare_objective_matrix()
        self._add_base_constraints()
        self.set_objective()

    def _prepare_problem_and_solver(self):
        """Initialize the linear programming problem and the solver."""
        self.problem = pulp.LpProblem("Protein_Library_Optimization", pulp.LpMinimize)
        logger.info("Linear programming problem initialized")
        solver_list = pulp.listSolvers(onlyAvailable=True)
        logger.info(f"Available solvers: {solver_list}")
        self.solver = pulp.PULP_CBC_CMD(msg=self.debug > 2, keepFiles=False)

    def _prepare_variables_and_zero_pad_matrix(self):
        """Build the symbolic variable vector X."""
        data_df = self.data_loader.data_df
        zero_enforced_mutations = []
        data_df_padded = []

        # Loop over the positions to create the variables and store information
        # for the constraints.
        for position in self.data_loader.positions:
            wt = self.data_loader.wildtype_position_amino[position]
            for aa in amino_acids:
                if aa == wt:
                    continue
                mutation_name = f"{wt}{position}{aa}"
                # Create the x variable.
                x_var = pulp.LpVariable(f"X_{mutation_name}", cat="Binary")
                self.x_vars.append(x_var)
                self.x_vars_dict[mutation_name] = x_var
                # Save the aromatic variables.
                if aa in aromatic_amino_acids:
                    self.aromatic_vars.append(x_var)
                    self.aromatic_vars_dict[mutation_name] = x_var
                # Save the forbidden variables.
                if aa in self.forbidden_aa:
                    self.forbidden_vars.append(x_var)
                    self.forbidden_vars_dict[mutation_name] = x_var
                # Check if row exists in the input dataframe.
                if mutation_name in data_df["Mutation"].values:
                    # Extract the row from the dataframe in a dictionary format.
                    row = data_df[data_df["Mutation"] == mutation_name].to_dict(
                        "records"
                    )[0]
                    data_df_padded.append(row)
                else:  # The row does not exist in the input dataframe.
                    # Add 0-vector row for the new mutation.
                    new_row = {"Mutation": mutation_name}
                    # Save the position and aa to add X_pos_a = 0 constraint later in the script.
                    zero_enforced_mutations.append((wt, position, aa))
                    self.missing_vars.append(x_var)
                    self.missing_vars_dict[mutation_name] = x_var
                    for target in self.targets:
                        new_row[target] = 0
                    # Append the row to the padded dataframe.
                    data_df_padded.append(new_row)
        # Save the zero enforced mutations.
        self.zero_enforced_mutations = zero_enforced_mutations
        # Update the data_df with the padded data.
        self.data_df = pd.DataFrame(data_df_padded)
        del data_df_padded

        logger.info(f"Number of x-variables: {len(self.x_vars)}")

        self._check_data_and_variables_consistency()

        # Save the data.
        self.data_df.to_csv(Path(self.output_folder) / "padded_data.csv", index=False)

    def _check_data_and_variables_consistency(self):
        # Check the dimensions of the dataframe after adding the missing position-amino acid pairs.
        if self.data_df.shape[0] != len(self.data_loader.positions) * (
            len(amino_acids) - 1
        ):
            logger.error(
                f"Error adding missing position-amino acid pairs. Expected {len(self.data_loader.positions) * (len(amino_acids) - 1)} rows. \
                Got {self.data_df.shape[0]} rows."
            )
            exit()

        # Check that data_df["Mutation"].values is equivalent (ordered in the same way) as x_vars.
        for index, x_var in enumerate(self.x_vars):
            mutation_name = x_var.getName().split("_")[1]
            if mutation_name != self.data_df["Mutation"].values[index]:
                logger.error(
                    f"Error adding missing position-amino acid pairs. Expected {mutation_name}. \
                    Got {self.data_df['Mutation'].values[index]}"
                )
                exit()

    def _prepare_objective_matrix(self):
        self.A_original = self.data_df[self.data_df.columns[1:]].values.transpose()

        self._detect_objective_constraints_rows()

        if self.objective_constraint_row_indices is not None:
            self.constraint_data_df = self.data_df[
                self.data_df.columns[self.objective_constraint_row_indices]
            ]
            self.data_df = self.data_df.drop(
                self.data_df.columns[self.objective_constraint_row_indices], axis=1
            )

        # Tranpose the data to get the matrix A.
        A_auxiliar = self.data_df[self.data_df.columns[1:]].values.transpose()
        if self.data_normalization:
            min_vals = np.min(A_auxiliar, axis=1)
            max_vals = np.max(A_auxiliar, axis=1)
            # normalize the data to be between 0 and 1.
            self.A = (A_auxiliar - min_vals[:, np.newaxis]) / (max_vals - min_vals)[
                :, np.newaxis
            ]
            # Project to be between -1 and 1.
            self.A = 2 * self.A - 1
            logger.info("Data was normalized to be between -1 and 1")
        else:
            self.A = A_auxiliar
        del A_auxiliar

        logger.info(f"Dimensions of raw data after transposing: {self.A.shape}")

        self.single_objective = len(self.targets) == 1
        if self.single_objective:
            assert self.A.shape[0] == 1
            if self.weighted_multi_objective:
                logger.warning(
                    "Only one target was found. Setting weighted_multi_objective = False."
                )
                self.weighted_multi_objective = False

        # Get the low rank decomposition.
        rank_of_matrix = matrix_rank(self.A)
        logger.info(f"The rank of the matrix is {rank_of_matrix}")

        # If we do not use the weighted multi-objective formulation, then we need to compute the SVD
        # to get the rank-1 approximation of the objective matrix.
        if not self.single_objective and not self.weighted_multi_objective:
            logger.info(
                "We have a multi-objective problem and not using the weighted multi-objective formulation.\n\
    Computing the SVD to get the rank-1 approximation of the objective matrix."
            )
            try:
                u, sigma, vt = svd(self.A, full_matrices=False)
            except Exception:
                logger.error("Unable to compute the SVD")
                exit(1)
            self.rr = np.mean(u[:, 0]) * sigma[0] * vt[0, :]

    def _detect_objective_constraints_rows(self):
        self.objective_constraint_row_indices = []
        if len(self.objective_constraints) > 0:
            if self.objective_constraints[0].isdigit():
                self.objective_constraint_row_indices = [
                    int(val) for val in self.objective_constraints
                ]
            else:
                cols = list(self.data_df.columns)
                for constraint_name in self.objective_constraints:
                    # Subtract 1 because index 0 is the string "Mutation" not a target and will not be in the A matrix.
                    try:
                        row_number = cols.index(constraint_name)
                        self.objective_constraint_row_indices.append(row_number)
                    except ValueError:
                        logger.error(
                            f"Constraint '{constraint_name}' was not found in the input file."
                        )
                        exit()

    def _add_constraint(self, constraint, name, debug_level=0):
        self.problem += constraint, name
        if self.debug > debug_level:
            logger.debug(f"Adding constraint ({name}): {constraint}")

    def _add_base_constraints(self):
        # Force forbidden amino acids to be zero.
        for x_var in self.forbidden_vars:
            self._add_constraint(
                x_var == 0, f"Forbidden_{x_var.getName()}", debug_level=2
            )

        # Force the missing position-amino acid pairs to be zero.
        for x_var in self.missing_vars:
            self._add_constraint(
                x_var == 0, f"Missing_{x_var.getName()}", debug_level=2
            )

        # The number of mutations per position is at most 1
        # We have 20 amino acids minus one for the wild type.
        NUM_AVAILABLE_AA = 19
        self.position_to_x_vars_dict = {
            parse_mutation(self.x_vars[i].getName().split("_")[1])[1]: self.x_vars[
                i : i + NUM_AVAILABLE_AA
            ]
            for i in range(0, len(self.x_vars), NUM_AVAILABLE_AA)
        }
        for position, x_vars_in_pos in self.position_to_x_vars_dict.items():
            assert len(x_vars_in_pos) == 19
            self._add_constraint(
                pulp.lpSum(x_vars_in_pos) <= 1, f"tot_mut_at_{position}"
            )

        # The number of total mutations is at least min_mut and at most max_mut.
        self._add_constraint(
            pulp.lpSum(self.x_vars) >= self.min_mut, "min_mut", debug_level=0
        )
        self._add_constraint(
            pulp.lpSum(self.x_vars) <= self.max_mut, "max_mut", debug_level=0
        )

        # Aromatic var count is less than user set value max_arom_per_seq.
        if self.max_arom_per_seq is not None:
            self._add_constraint(
                pulp.lpSum(self.aromatic_vars) <= self.max_arom_per_seq,
                "max_arom_per_seq",
                debug_level=0,
            )

        # The objective constraints.
        if self.objective_constraint_row_indices is not None:
            for index, constraint in enumerate(self.objective_constraint_row_indices):
                A_column = self.constraint_data_df.iloc[:, index].values
                self._add_constraint(
                    pulp.lpSum(A_column * self.x_vars)
                    <= self.objective_constraints_param[index],
                    f"objective_constraint_{constraint}",
                )
        self._add_llm_constraints()

    def _add_llm_constraints(self):
        if not self.llm_constraints:
            return
        for index, constraint in enumerate(self.llm_constraints):
            if not isinstance(constraint, dict):
                continue
            ctype = constraint.get("type")
            mutations = constraint.get("mutations", [])
            if not mutations:
                continue
            vars_for_mutations = []
            missing = []
            for mutation in mutations:
                x_var = self.x_vars_dict.get(mutation)
                if x_var is None:
                    missing.append(mutation)
                else:
                    vars_for_mutations.append(x_var)
            if missing and self.debug > 0:
                logger.warning(
                    f"LLM constraint references unknown mutations: {missing}"
                )
            if not vars_for_mutations:
                continue
            name_prefix = f"llm_{ctype}_{index}"
            if ctype == "forbid_mutation":
                for x_var in vars_for_mutations:
                    self._add_constraint(
                        x_var == 0, f"{name_prefix}_{x_var.getName()}", debug_level=1
                    )
            elif ctype == "require_mutation":
                for x_var in vars_for_mutations:
                    self._add_constraint(
                        x_var == 1, f"{name_prefix}_{x_var.getName()}", debug_level=1
                    )
            elif ctype == "forbid_combination":
                self._add_constraint(
                    pulp.lpSum(vars_for_mutations) <= len(vars_for_mutations) - 1,
                    name_prefix,
                    debug_level=1,
                )
            elif ctype == "limit_count":
                limit = constraint.get("limit")
                if limit is None:
                    logger.warning(
                        f"LLM constraint {name_prefix} missing limit; skipping."
                    )
                    continue
                self._add_constraint(
                    pulp.lpSum(vars_for_mutations) <= int(limit),
                    name_prefix,
                    debug_level=1,
                )
            else:
                logger.warning(f"Unsupported LLM constraint type: {ctype}")

    def set_objective(self):

        # Static objective problem.
        if self.single_objective:
            self.problem += pulp.lpSum(np.dot(self.A, self.x_vars))
        elif not self.weighted_multi_objective:
            self.problem += pulp.lpSum(self.rr * self.x_vars)

    def update_generator_before_generation(self, iteration: int):
        if self.interleave_mutant_order:
            curr_max_mut = self.min_mut + (
                iteration % (self.max_mut - self.min_mut + 1)
            )
            self.problem.constraints["max_mut"].changeRHS(curr_max_mut)
            if self.debug > 0:
                logger.debug(
                    f"Changing constraint max_mut constraint to {curr_max_mut}"
                )
            if self.force_mutant_order_balance:
                self.problem.constraints["min_mut"].changeRHS(curr_max_mut)
                if self.debug > 0:
                    logger.debug(
                        f"Changing constraint min_mut constraint to {curr_max_mut}"
                    )

        if self.weighted_multi_objective:
            self.weights = np.random.random(size=self.A.shape[0])
            # Normalize to add up to 1.
            self.weights = self.weights / np.sum(self.weights)
            dict_weights = dict(zip(self.targets, self.weights))
            if self.debug > 0:
                logger.debug(f"Target weights: {dict_weights}")
            self.problem += pulp.lpSum(self.weights * np.dot(self.A, self.x_vars))

        if self.debug > 1:
            if iteration == 0:
                dir_problems = Path(self.output_folder) / "ilp_problems"
                # Create the directory to save the problem files to
                if not dir_problems.exists():
                    dir_problems.mkdir(parents=True, exist_ok=True)
            # Save the problem to a file
            self.problem.writeLP(str(dir_problems / f"problem_{iteration}.lp"))

    def generate_one_solution(self, iteration: int):

        cpu_time_start = time.time()

        # Call the solver.
        status = self.problem.solve(self.solver)

        if status != 1:
            logger.error(
                f"Error in ILPGenerator when solving the problem. Status: {pulp.LpStatus[status]}"
            )
            return None

        cpu_time = time.time() - cpu_time_start
        self.cpu_times.append(cpu_time)

        # Post processing the solution.
        self.many_hot_encoded_solution = [x.value() for x in self.x_vars]
        x_mutations_in_solution = [x for x in self.x_vars if x.value() == 1]
        self.parsed_mutations_in_solution = [
            parse_mutation(x.getName().split("_")[1]) for x in x_mutations_in_solution
        ]
        formatted_mutations_in_solution = ",".join(
            [f"{x[0]}{x[1]}{x[2]}" for x in self.parsed_mutations_in_solution]
        )
        # Check if the solution is valid.
        for mutation in self.parsed_mutations_in_solution:
            if mutation in self.zero_enforced_mutations:
                logger.error(
                    f"Solution {formatted_mutations_in_solution} is not valid. Skipping."
                )
                continue
        logger.info(f"Solution: {formatted_mutations_in_solution}")

        return self.create_solution_dict(
            formatted_mutations_in_solution,
            iteration,
            x_mutations_in_solution,
            cpu_time,
        )

    def create_solution_dict(
        self,
        formatted_mutations_in_solution,
        iteration,
        x_mutations_in_solution,
        cpu_time,
    ):

        # Logging.
        self.many_hot_encoded_solutions.append(self.many_hot_encoded_solution)

        objective_values_all = np.dot(self.A, self.many_hot_encoded_solution)
        solution_dict = {
            "solution": formatted_mutations_in_solution,
            "objective_value": self.problem.objective.value(),
            "reason_selected": f"ilp_solve_iteration_{iteration}",
        }
        if self.weighted_multi_objective:
            solution_dict["weights"] = self.weights
        if self.data_normalization or self.objective_constraint_row_indices is not None:
            objective_values_all_original = np.dot(
                self.A_original, self.many_hot_encoded_solution
            )
        for target, obj_value in zip(self.targets, objective_values_all):
            solution_dict[target] = obj_value
        if self.data_normalization or self.objective_constraint_row_indices is not None:
            for target, obj_value in zip(self.targets, objective_values_all_original):
                solution_dict[f"{target}_original"] = obj_value

        self.list_of_solution_dicts.append(solution_dict)

        logger.info(
            f"Runtime: {cpu_time}, Solution: {x_mutations_in_solution}, Objective: {self.problem.objective.value()}"
        )

        return solution_dict

    def update_generator(self, iteration: int):

        # Prepare constraints for next iteration.
        for wt, pos, mutation_aa in self.parsed_mutations_in_solution:
            if pos in self.positions_in_solution_counts:
                self.positions_in_solution_counts[pos] += 1
            else:
                self.positions_in_solution_counts[pos] = 1

            mutation = f"{wt}{pos}{mutation_aa}"
            if mutation in self.mutations_in_solution_counts:
                self.mutations_in_solution_counts[mutation] += 1
            else:
                self.mutations_in_solution_counts[mutation] = 1

        # Remove from search space the ball of radius dissimilarity_tolerance around the solution
        # (x_vars - previous_sol)^2 >= 1 + dissimilarity_tolerance
        # x_vars^2 - 2 * x_vars.previous_sol + previous_sol^2 >= 1 + dissimilarity_tolerance
        # Sum(x_vars) - 2 * x_vars.previous_sol  >= 1 + dissimilarity_tolerance - Sum(previous_sol) (binary variables)
        dissimilarity_rhs = (
            1
            + self.dissimilarity_tolerance
            - pulp.lpSum(self.many_hot_encoded_solution)
        )
        dissimilarity_lhs = pulp.lpSum(self.x_vars) - 2 * np.dot(
            self.x_vars, self.many_hot_encoded_solution
        )
        self._add_constraint(
            dissimilarity_lhs >= dissimilarity_rhs,
            f"dissimilarity_from_iter_{iteration}",
            debug_level=1,
        )

        if self.schedule == 1:

            if iteration % self.schedule_param[0] == 0:
                sorted_mutation_counts = sorted(
                    self.mutations_in_solution_counts.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                mutation_to_eliminate = sorted_mutation_counts[0][0]

                # Remove the commonest mutation every p0 iterations.
                self._add_constraint(
                    self.x_vars_dict[mutation_to_eliminate] == 0,
                    f"rm_{mutation_to_eliminate}_from_iter_{iteration}",
                    debug_level=1,
                )
                self.mutations_in_solution_counts[mutation_to_eliminate] = -1

            if iteration % self.schedule_param[1] == 0:
                sorted_position_counts = sorted(
                    self.positions_in_solution_counts.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
                position_to_eliminate = sorted_position_counts[0][0]

                # Remove the commonest position every p1 iterations.
                self._add_constraint(
                    pulp.lpSum(self.position_to_x_vars_dict[position_to_eliminate])
                    == 0,
                    f"rm_{position_to_eliminate}_from_iter_{iteration}",
                    debug_level=1,
                )
                self.positions_in_solution_counts[position_to_eliminate] = -1

        elif self.schedule == 2:

            for mutation, count in self.mutations_in_solution_counts.items():
                if count > self.schedule_param[0]:
                    # Remove the mutation if it appears more than p0 times.
                    self._add_constraint(
                        self.x_vars_dict[mutation] == 0,
                        f"rm_{mutation}_from_iter_{iteration}",
                        debug_level=1,
                    )
                    self.mutations_in_solution_counts[mutation] = -1

            for position, count in self.positions_in_solution_counts.items():
                if count > self.schedule_param[1]:
                    # Remove the position if it appears more than p1 times.
                    self._add_constraint(
                        pulp.lpSum(self.position_to_x_vars_dict[position]) == 0,
                        f"rm_{position}_from_iter_{iteration}",
                        debug_level=1,
                    )
                    self.positions_in_solution_counts[position] = -1

    def __str__(self):
        return super().__str__() + ": ILP Generator"
