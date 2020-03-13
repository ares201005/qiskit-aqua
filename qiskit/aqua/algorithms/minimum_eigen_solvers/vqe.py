# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
The Variational Quantum Eigensolver algorithm.

See https://arxiv.org/abs/1304.3061
"""

from typing import Optional, List, Callable
import logging
import functools
from time import time

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import ParameterVector

from qiskit.aqua.algorithms import VQAlgorithm
from qiskit.aqua import AquaError
from qiskit.aqua.operators import (TPBGroupedWeightedPauliOperator, WeightedPauliOperator,
                                   MatrixOperator, op_converter)
from qiskit.aqua.utils.backend_utils import (is_statevector_backend,
                                             is_aer_provider)
from qiskit.aqua.operators import BaseOperator
from qiskit.aqua.components.optimizers import Optimizer
from qiskit.aqua.components.variational_forms import VariationalForm

logger = logging.getLogger(__name__)


class VQE(VQAlgorithm):
    r"""
    The Variational Quantum Eigensolver algorithm.

    `VQE <https://arxiv.org/abs/1304.3061>`__ is a hybrid algorithm that uses a
    variational technique and interleaves quantum and classical computations in order to find
    the minimum eigenvalue of the Hamiltonian :math:`H` of a given system.

    An instance of VQE requires defining two algorithmic sub-components:
    a trial state (ansatz) from Aqua's :mod:`~qiskit.aqua.components.variational_forms`, and one
    of the classical :mod:`~qiskit.aqua.components.optimizers`. The ansatz is varied, via its set
    of parameters, by the optimizer, such that it works towards a state, as determined by the
    parameters applied to the variational form, that will result in the minimum expectation value
    being measured of the input operator (Hamiltonian).

    An optional array of parameter values, via the *initial_point*, may be provided as the
    starting point for the search of the minimum eigenvalue. This feature is particularly useful
    such as when there are reasons to believe that the solution point is close to a particular
    point.  As an example, when building the dissociation profile of a molecule,
    it is likely that using the previous computed optimal solution as the starting
    initial point for the next interatomic distance is going to reduce the number of iterations
    necessary for the variational algorithm to converge.  Aqua provides an
    `initial point tutorial <https://github.com/Qiskit/qiskit-tutorials-community/blob/master
    /chemistry/h2_vqe_initial_point.ipynb>`__ detailing this use case.

    The length of the *initial_point* list value must match the number of the parameters
    expected by the variational form being used. If the *initial_point* is left at the default
    of ``None``, then VQE will look to the variational form for a preferred value, based on its
    given initial state. If the variational form returns ``None``,
    then a random point will be generated within the parameter bounds set, as per above.
    If the variational form provides ``None`` as the lower bound, then VQE
    will default it to :math:`-2\pi`; similarly, if the variational form returns ``None``
    as the upper bound, the default value will be :math:`2\pi`.
    """

    def __init__(self, operator: BaseOperator, var_form: VariationalForm, optimizer: Optimizer,
                 initial_point: Optional[np.ndarray] = None, max_evals_grouped: int = 1,
                 aux_operators: Optional[List[BaseOperator]] = None,
                 callback: Optional[Callable[[int, np.ndarray, float, float], None]] = None,
                 auto_conversion: bool = True) -> None:
        """

        Args:
            operator: Qubit operator of the Hamiltonian
            var_form: A parameterized variational form (ansatz).
            optimizer: A classical optimizer.
            initial_point: An optional initial point (i.e. initial parameter values)
                for the optimizer. If ``None`` then VQE will look to the variational form for a
                preferred point and if not will simply compute a random one.
            max_evals_grouped: Max number of evaluations performed simultaneously. Signals the
                given optimizer that more than one set of parameters can be supplied so that
                potentially the expectation values can be computed in parallel. Typically this is
                possible when a finite difference gradient is used by the optimizer such that
                multiple points to compute the gradient can be passed and if computed in parallel
                improve overall execution time.
            aux_operators: Optional list of auxiliary operators to be evaluated with the eigenstate
                of the minimum eigenvalue main result and their expectation values returned.
                For instance in chemistry these can be dipole operators, total particle count
                operators so we can get values for these at the ground state.
            callback: a callback that can access the intermediate data during the optimization.
                Four parameter values are passed to the callback as follows during each evaluation
                by the optimizer for its current set of parameters as it works towards the minimum.
                These are: the evaluation count, the optimizer parameters for the
                variational form, the evaluated mean and the evaluated standard deviation.
            auto_conversion: When ``True`` allows an automatic conversion for operator and
                aux_operators into the type which is most suitable for the backend on which the
                algorithm is run.

                - for *non-Aer statevector simulator:*
                  :class:`~qiskit.aqua.operators.MatrixOperator`
                - for *Aer statevector simulator:*
                  :class:`~qiskit.aqua.operators.WeightedPauliOperator`
                - for *qasm simulator or real backend:*
                  :class:`~qiskit.aqua.operators.TPBGroupedWeightedPauliOperator`
        """
        super().__init__(var_form=var_form,
                         optimizer=optimizer,
                         cost_fn=self._energy_evaluation,
                         initial_point=initial_point)
        self._use_simulator_snapshot_mode = None
        self._ret = None
        self._eval_time = None
        self._optimizer.set_max_evals_grouped(max_evals_grouped)
        self._callback = callback
        if initial_point is None:
            #print('test-zy (VQE): initial_point is none! before preferred_init_points')
            self._initial_point = var_form.preferred_init_points
            print('test-zy(VQE): initial_point=', self._initial_point)
        self._operator = operator
        self._eval_count = 0
        self._aux_operators = []
        if aux_operators is not None:
            #print('test-zy(VQE): aux_operators is not none')
            aux_operators = \
                [aux_operators] if not isinstance(aux_operators, list) else aux_operators
            for aux_op in aux_operators:
                self._aux_operators.append(aux_op)
        self._auto_conversion = auto_conversion
        logger.info(self.print_settings())
        self._var_form_params = ParameterVector('θ', self._var_form.num_parameters)

        #print('test-zy(VQE): var_form_params', self._var_form_params)

        self._parameterized_circuits = None

    @property
    def setting(self):
        """Prepare the setting of VQE as a string."""
        ret = "Algorithm: {}\n".format(self.__class__.__name__)
        params = ""
        for key, value in self.__dict__.items():
            if key[0] == "_":
                if "initial_point" in key and value is None:
                    params += "-- {}: {}\n".format(key[1:], "Random seed")
                else:
                    params += "-- {}: {}\n".format(key[1:], value)
        ret += "{}".format(params)
        return ret

    def print_settings(self):
        """
        Preparing the setting of VQE into a string.

        Returns:
            str: the formatted setting of VQE
        """
        ret = "\n"
        ret += "==================== Setting of {} ============================\n".format(
            self.__class__.__name__)
        ret += "{}".format(self.setting)
        ret += "===============================================================\n"
        ret += "{}".format(self._var_form.setting)
        ret += "===============================================================\n"
        ret += "{}".format(self._optimizer.setting)
        ret += "===============================================================\n"
        return ret

    def _config_the_best_mode(self, operator, backend):

        if not isinstance(operator, (WeightedPauliOperator, MatrixOperator,
                                     TPBGroupedWeightedPauliOperator)):
            logger.debug("Unrecognized operator type, skip auto conversion.")
            return operator

        ret_op = operator
        if not is_statevector_backend(backend) and not (
                is_aer_provider(backend)
                and self._quantum_instance.run_config.shots == 1):
            if isinstance(operator, (WeightedPauliOperator, MatrixOperator)):
                logger.debug("When running with Qasm simulator, grouped pauli can "
                             "save number of measurements. "
                             "We convert the operator into grouped ones.")
                ret_op = op_converter.to_tpb_grouped_weighted_pauli_operator(
                    operator, TPBGroupedWeightedPauliOperator.sorted_grouping)
        else:
            if not is_aer_provider(backend):
                if not isinstance(operator, MatrixOperator):
                    logger.info("When running with non-Aer statevector simulator, "
                                "represent operator as a matrix could "
                                "achieve the better performance. We convert "
                                "the operator to matrix.")
                    ret_op = op_converter.to_matrix_operator(operator)
            else:
                if not isinstance(operator, WeightedPauliOperator):
                    logger.info("When running with Aer simulator, "
                                "represent operator as weighted paulis could "
                                "achieve the better performance. We convert "
                                "the operator to weighted paulis.")
                    ret_op = op_converter.to_weighted_pauli_operator(operator)
        return ret_op

    def construct_circuit(self, parameter, statevector_mode=False,
                          use_simulator_snapshot_mode=False, circuit_name_prefix=''):
        """Generate the circuits.

        Args:
            parameter (numpy.ndarray): parameters for variational form.
            statevector_mode (bool, optional): indicate which type of simulator are going to use.
            use_simulator_snapshot_mode (bool, optional): is backend from AerProvider,
                            if True and mode is paulis, single circuit is generated.
            circuit_name_prefix (str, optional): a prefix of circuit name

        Returns:
            list[QuantumCircuit]: the generated circuits with Hamiltonian.
        """
        wave_function = self._var_form.construct_circuit(parameter)
        circuits = self._operator.construct_evaluation_circuit(
            wave_function, statevector_mode,
            use_simulator_snapshot_mode=use_simulator_snapshot_mode,
            circuit_name_prefix=circuit_name_prefix)
        return circuits

    def _eval_aux_ops(self, threshold=1e-12, params=None):
        if params is None:
            params = self.optimal_params
        wavefn_circuit = self._var_form.construct_circuit(params)
        circuits = []
        values = []
        params = []
        for idx, operator in enumerate(self._aux_operators):
            if not operator.is_empty():
                temp_circuit = QuantumCircuit() + wavefn_circuit
                circuit = operator.construct_evaluation_circuit(
                    wave_function=temp_circuit,
                    statevector_mode=self._quantum_instance.is_statevector,
                    use_simulator_snapshot_mode=self._use_simulator_snapshot_mode,
                    circuit_name_prefix=str(idx))
            else:
                circuit = None
            circuits.append(circuit)

        if circuits:
            to_be_simulated_circuits = \
                functools.reduce(lambda x, y: x + y, [c for c in circuits if c is not None])
            result = self._quantum_instance.execute(to_be_simulated_circuits)

            for idx, operator in enumerate(self._aux_operators):
                if operator.is_empty():
                    mean, std = 0.0, 0.0
                else:
                    mean, std = operator.evaluate_with_result(
                        result=result, statevector_mode=self._quantum_instance.is_statevector,
                        use_simulator_snapshot_mode=self._use_simulator_snapshot_mode,
                        circuit_name_prefix=str(idx))

                mean = mean.real if abs(mean.real) > threshold else 0.0
                std = std.real if abs(std.real) > threshold else 0.0
                values.append((mean, std))

        if values:
            aux_op_vals = np.empty([1, len(self._aux_operators), 2])
            aux_op_vals[0, :] = np.asarray(values)
            self._ret['aux_ops'] = aux_op_vals

    def _run(self):
        """
        Run the algorithm to compute the minimum eigenvalue.

        Returns:
            dict: Dictionary of results

        Raises:
            AquaError: wrong setting of operator and backend.
        """
        if self._auto_conversion:
            self._operator = \
                self._config_the_best_mode(self._operator, self._quantum_instance.backend)
            for i in range(len(self._aux_operators)):
                if not self._aux_operators[i].is_empty():
                    self._aux_operators[i] = \
                        self._config_the_best_mode(self._aux_operators[i],
                                                   self._quantum_instance.backend)

        # sanity check
        if isinstance(self._operator, MatrixOperator) and not self._quantum_instance.is_statevector:
            raise AquaError("Non-statevector simulator can not work "
                            "with `MatrixOperator`, either turn ON "
                            "auto_conversion or use the proper "
                            "combination between operator and backend.")

        self._use_simulator_snapshot_mode = (
            is_aer_provider(self._quantum_instance.backend)
            and self._quantum_instance.run_config.shots == 1
            and not self._quantum_instance.noise_config
            and isinstance(self._operator,
                           (WeightedPauliOperator, TPBGroupedWeightedPauliOperator)))

        print("use_simulator_snapshot_mode    = ",self._use_simulator_snapshot_mode)
        print("quantum_instance.is_statevector=", self._quantum_instance.is_statevector)

        self._quantum_instance.circuit_summary = True

        self._eval_count = 0
        self._ret = self.find_minimum(initial_point=self.initial_point,
                                      var_form=self.var_form,
                                      cost_fn=self._energy_evaluation,
                                      optimizer=self.optimizer)
        if self._ret['num_optimizer_evals'] is not None and \
                self._eval_count >= self._ret['num_optimizer_evals']:
            self._eval_count = self._ret['num_optimizer_evals']
        self._eval_time = self._ret['eval_time']
        logger.info('Optimization complete in %s seconds.\nFound opt_params %s in %s evals',
                    self._eval_time, self._ret['opt_params'], self._eval_count)

        #ZY
        print('')
        theta = self._ret['opt_params'].tolist()
        print('Optimization complete in %d evals' % self._eval_count)
        print('optimization parameters=', theta)

        self._ret['eval_count'] = self._eval_count

        self._ret['energy'] = self.get_optimal_cost()
        self._ret['eigvals'] = np.asarray([self.get_optimal_cost()])
        self._ret['eigvecs'] = np.asarray([self.get_optimal_vector()])
        self._eval_aux_ops()

        self.cleanup_parameterized_circuits()
        return self._ret

    # This is the objective function to be passed to the optimizer that is used for evaluation
    def _energy_evaluation(self, parameters):
        """
        Evaluate energy at given parameters for the variational form.

        Args:
            parameters (numpy.ndarray): parameters for variational form.

        Returns:
            Union(float, list[float]): energy of the hamiltonian of each parameter.
        """
        num_parameter_sets = len(parameters) // self._var_form.num_parameters
        parameter_sets = np.split(parameters, num_parameter_sets)
        mean_energy = []
        std_energy = []

        def _build_parameterized_circuits():
            if self._var_form.support_parameterized_circuit and \
                    self._parameterized_circuits is None:
                parameterized_circuits = self.construct_circuit(
                    self._var_form_params,
                    statevector_mode=self._quantum_instance.is_statevector,
                    use_simulator_snapshot_mode=self._use_simulator_snapshot_mode)

                self._parameterized_circuits = \
                    self._quantum_instance.transpile(parameterized_circuits)
        _build_parameterized_circuits()
        circuits = []
        # binding parameters here since the circuits had been transpiled
        if self._parameterized_circuits is not None:
            for idx, parameter in enumerate(parameter_sets):
                curr_param = {self._var_form_params: parameter}
                for qc in self._parameterized_circuits:
                    tmp = qc.bind_parameters(curr_param)
                    tmp.name = str(idx) + tmp.name
                    circuits.append(tmp)
            to_be_simulated_circuits = circuits
        else:
            for idx, parameter in enumerate(parameter_sets):
                circuit = self.construct_circuit(
                    parameter,
                    statevector_mode=self._quantum_instance.is_statevector,
                    use_simulator_snapshot_mode=self._use_simulator_snapshot_mode,
                    circuit_name_prefix=str(idx))
                circuits.append(circuit)
            to_be_simulated_circuits = functools.reduce(lambda x, y: x + y, circuits)

        start_time = time()

        print('test-zy: call quantum_instance.execulte')
        result = self._quantum_instance.execute(to_be_simulated_circuits,
                                                self._parameterized_circuits is not None)

        for idx, _ in enumerate(parameter_sets):
            mean, std = self._operator.evaluate_with_result(
                result=result, statevector_mode=self._quantum_instance.is_statevector,
                use_simulator_snapshot_mode=self._use_simulator_snapshot_mode,
                circuit_name_prefix=str(idx))
            end_time = time()
            mean_energy.append(np.real(mean))
            std_energy.append(np.real(std))
            self._eval_count += 1
            if self._callback is not None:
                self._callback(self._eval_count, parameter_sets[idx], np.real(mean), np.real(std))

            # If there is more than one parameter set then the calculation of the
            # evaluation time has to be done more carefully,
            # therefore we do not calculate it
            if len(parameter_sets) == 1:
                logger.info('Energy evaluation %s returned %s - %.5f (ms)',
                            self._eval_count, np.real(mean), (end_time - start_time) * 1000)
            else:
                logger.info('Energy evaluation %s returned %s',
                            self._eval_count, np.real(mean))

        return mean_energy if len(mean_energy) > 1 else mean_energy[0]

    def get_optimal_cost(self):
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot return optimal cost before running the "
                            "algorithm to find optimal params.")
        return self._ret['min_val']

    def get_optimal_circuit(self):
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal circuit before running the "
                            "algorithm to find optimal params.")
        return self._var_form.construct_circuit(self._ret['opt_params'])

    def get_optimal_vector(self):
        # pylint: disable=import-outside-toplevel
        from qiskit.aqua.utils.run_circuits import find_regs_by_name

        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal vector before running the "
                            "algorithm to find optimal params.")
        qc = self.get_optimal_circuit()
        if self._quantum_instance.is_statevector:
            ret = self._quantum_instance.execute(qc)
            self._ret['min_vector'] = ret.get_statevector(qc)
        else:
            c = ClassicalRegister(qc.width(), name='c')
            q = find_regs_by_name(qc, 'q')
            qc.add_register(c)
            qc.barrier(q)
            qc.measure(q, c)
            ret = self._quantum_instance.execute(qc)
            self._ret['min_vector'] = ret.get_counts(qc)
        return self._ret['min_vector']

    @property
    def optimal_params(self):
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal params before running the algorithm.")
        return self._ret['opt_params']
