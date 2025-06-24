from typing import List, Literal, Optional, Tuple

from qualibrate import QualibrationNode
from qualibrate.parameters import RunnableParameters
from qualibration_libs.core import BatchableList
from quam_builder.architecture.superconducting.qpu import AnyQuam
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from quam_builder.architecture.superconducting.qubit_pair import AnyTransmonPair


class BaseExperimentNodeParameters(RunnableParameters):
    multiplexed: bool = False
    """Whether to play control pulses, readout pulses and active/thermal reset at the same time for all qubits (True)
    or to play the experiment sequentially for each qubit (False). Default is False."""
    use_state_discrimination: bool = False
    """Whether to use on-the-fly state discrimination and return the qubit 'state', or simply return the demodulated
    quadratures 'I' and 'Q'. Default is False."""
    reset_type: Literal["thermal", "active", "active_gef"] = "thermal"
    """The qubit reset method to use. Must be implemented as a method of Quam.qubit. Can be "thermal", "active", or
    "active_gef". Default is "thermal"."""


class QubitsExperimentNodeParameters(BaseExperimentNodeParameters):
    qubits: Optional[List[str]] = None
    """A list of qubit names which should participate in the execution of the node. Default is None."""


class TwoQubitExperimentNodeParameters(BaseExperimentNodeParameters):
    qubit_pairs: Optional[List[str]] = None
    """A list of qubit names which should participate in the execution of the node. Default is None."""


def get_qubits(node: QualibrationNode) -> BatchableList[AnyTransmon]:
    qubits, batch_groups = _get_qubits(node.machine, node.parameters)

    if isinstance(node.parameters, QubitsExperimentNodeParameters):
        multiplexed = node.parameters.multiplexed
    else:
        multiplexed = False

    qubits_batchable_list = _make_batchable_list_from_multiplexed(qubits, multiplexed)

    return qubits_batchable_list


def _get_qubits(
    machine: AnyQuam, node_parameters: QubitsExperimentNodeParameters
) -> Tuple[List[AnyTransmon], Optional[List[List[int]]]]:
    if node_parameters.qubits is None or node_parameters.qubits == "":
        qubits = machine.active_qubits
        batch_groups = [[i] for i in range(len(qubits))]
    else:
        flat_index = 0
        qubits: List[AnyTransmon] = []
        batch_groups: List[List[int]] = []
        for group in node_parameters.qubits:
            if isinstance(group, str):
                group = [group]
            batch = []
            for q in group:
                qubits.append(machine.qubits[q])
                batch.append(flat_index)
                flat_index += 1
            batch_groups.append(batch)

    return qubits, batch_groups


def get_qubit_pairs(node: QualibrationNode) -> BatchableList[AnyTransmonPair]:
    qubit_pairs, batch_groups = _get_qubit_pairs(node.machine, node.parameters)

    if isinstance(node.parameters, TwoQubitExperimentNodeParameters):
        multiplexed = node.parameters.multiplexed
    else:
        multiplexed = False

    qubit_pairs_batchable_list = _make_batchable_list_from_multiplexed(
        qubit_pairs, multiplexed
    )

    return qubit_pairs_batchable_list


def _get_qubit_pairs(
    machine: AnyQuam, node_parameters: TwoQubitExperimentNodeParameters
) -> Tuple[List[AnyTransmonPair], Optional[List[List[int]]]]:
    if node_parameters.qubit_pairs is None or node_parameters.qubit_pairs == "":
        qubit_pairs = machine.active_qubit_pairs
        batch_groups = [[i] for i in range(len(qubit_pairs))]
    else:
        flat_index = 0
        qubit_pairs: List[AnyTransmonPair] = []
        batch_groups: List[List[int]] = []
        for group in node_parameters.qubit_pairs:
            if isinstance(group, str):
                group = [group]
            batch = []
            for qp in group:
                qubit_pairs.append(machine.qubit_pairs[qp])
                batch.append(flat_index)
                flat_index += 1
            batch_groups.append(batch)

    return qubit_pairs, batch_groups


def _make_batchable_list_from_multiplexed(
    items: List,
    multiplexed: bool,
    batch_groups: Optional[List[List[int]]] = None,
) -> BatchableList:
    if batch_groups is not None:
        return BatchableList(items, batch_groups)

    if multiplexed:
        batch_groups = [[i for i in range(len(items))]]
    else:
        batch_groups = [[i] for i in range(len(items))]

    return BatchableList(items, batch_groups)
