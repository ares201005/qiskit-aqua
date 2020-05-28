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

""" sub system """

from collections import defaultdict
import numpy as np
from scipy.linalg import sqrtm

from qiskit.quantum_info.states import partial_trace


def get_subsystem_density_matrix(statevector, trace_systems):
    """
    Compute the reduced density matrix of a quantum subsystem.

    Args:
        statevector (list|array): The state vector of the complete system
        trace_systems (list|range): The indices of the qubits to be traced out.

    Returns:
        numpy.ndarray: The reduced density matrix for the desired subsystem
    """
    rho = np.outer(statevector, np.conj(statevector))
    rho_sub = partial_trace(rho, trace_systems).data
    return rho_sub


def get_subsystem_fidelity(statevector, trace_systems, subsystem_state):
    """
    Compute the fidelity of the quantum subsystem.

    Args:
        statevector (list|array): The state vector of the complete system
        trace_systems (list|range): The indices of the qubits to be traced.
            to trace qubits 0 and 4 trace_systems = [0,4]
        subsystem_state (list|array): The ground-truth state vector of the subsystem

    Returns:
        numpy.ndarray: The subsystem fidelity
    """
    rho = np.outer(np.conj(statevector), statevector)
    rho_sub = partial_trace(rho, trace_systems).data
    rho_sub_in = np.outer(np.conj(subsystem_state), subsystem_state)
    fidelity = np.trace(
        sqrtm(
            np.dot(
                np.dot(sqrtm(rho_sub), rho_sub_in),
                sqrtm(rho_sub)
            )
        )
    ) ** 2
    return fidelity


def get_subsystems_counts(complete_system_counts, post_select_index=None, post_select_flag=None):
    """
    Extract all subsystems' counts from the single complete system count dictionary.

    If multiple classical registers are used to measure various parts of a quantum system,
    Each of the measurement dictionary's keys would contain spaces as delimiters to separate
    the various parts being measured. For example, you might have three keys
    '11 010', '01 011' and '11 011', among many other, in the count dictionary of the
    5-qubit complete system, and would like to get the two subsystems' counts
    (one 2-qubit, and the other 3-qubit) in order to get the counts for the 2-qubit
    partial measurement '11' or the 3-qubit partial measurement '011'.

    If the post_select_index and post_select_flag parameter are specified, the counts are
    returned subject to that specific post selection, that is, the counts for all subsystems where
    the subsystem at index post_select_index is equal to post_select_flag.


    Args:
        complete_system_counts (dict): The measurement count dictionary of a complete system
            that contains multiple classical registers for measurements s.t. the dictionary's
            keys have space delimiters.
        post_select_index (int): Optional, the index of the subsystem to apply the post selection
            to.
        post_select_flag (str): Optional, the post selection value to apply to the subsystem
            at index post_select_index.

    Returns:
        list: A list of measurement count dictionaries corresponding to
                each of the subsystems measured.
    """
    mixed_measurements = list(complete_system_counts)
    subsystems_counts = [defaultdict(int) for _ in mixed_measurements[0].split()]
    for mixed_measurement in mixed_measurements:
        count = complete_system_counts[mixed_measurement]
        subsystem_measurements = mixed_measurement.split()
        for k, d_l in zip(subsystem_measurements, subsystems_counts):
            if (post_select_index is None
                    or subsystem_measurements[post_select_index] == post_select_flag):
                d_l[k] += count
    return [dict(d) for d in subsystems_counts]

def get_entropy(rho):

    w, v = np.linalg.eig(rho)

    #print('rho=',rho)
    #print('w=', w)

    s = 0.0
    for omega in w:
       g = np.real(omega) 
       if g > 1.E-12:
          s = s - g * np.log(g)

    return s

def get_mutual_info(statevector,num_qubits):

    """
      calculate the mutual informaiton
      input: statevector, number of qubits

    """
    mutual = np.zeros((num_qubits,num_qubits)) # [ [0] * num_qubits ] * num_qubits
    #mutual = []
    qubits = range(num_qubits)

    cost = 0.0
    for i in qubits:
      tracei = [x for x in qubits if x != i]
      rhoi = get_subsystem_density_matrix(statevector, tracei)

      si = get_entropy(rhoi)

      #mutuali = []
      for j in qubits:
         Iij = 0.0
         print("test--", i,j)
         if j != i:
           print("true", i,j)
           tracej = [x for x in qubits if x != j]
           rhoj = get_subsystem_density_matrix(statevector, tracej)
           sj = get_entropy(rhoj)
           
           traceij = [x for x in tracej if x != i]
           print('tracei=',tracei)
           print('tracej=',tracej)
           print('traceij=',traceij)
           rhoij = get_subsystem_density_matrix(statevector, traceij)
           
           sij = get_entropy(rhoij)
           Iij = si + sj - sij
           
         mutual[i][j] = Iij
         cost = cost + Iij * (i-j) * (i-j) 
         print('test-i,j,Iij=', i, j, mutual[i][j])
         #mutuali.append(Iij)

      #mutual.append(mutuali)
    print('total cost=', cost)
    return mutual

