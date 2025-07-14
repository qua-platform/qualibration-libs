from typing import List, Literal, Optional

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
    qubits = _get_qubits(node.machine, node.parameters)

    if isinstance(node.parameters, QubitsExperimentNodeParameters):
        multiplexed = node.parameters.multiplexed
    else:
        multiplexed = False

    qubits_batchable_list = _make_batchable_list_from_multiplexed(qubits, multiplexed)

    return qubits_batchable_list


def _get_qubits(machine: AnyQuam, node_parameters: QubitsExperimentNodeParameters) -> List[AnyTransmon]:
    if node_parameters.qubits is None or node_parameters.qubits == "":
        qubits = machine.active_qubits
    else:
        qubits = [machine.qubits[q] for q in node_parameters.qubits]

    return qubits


def get_qubit_pairs(node: QualibrationNode) -> BatchableList[AnyTransmonPair]:
    qubit_pairs = _get_qubit_pairs(node.machine, node.parameters)

    if isinstance(node.parameters, TwoQubitExperimentNodeParameters):
        multiplexed = node.parameters.multiplexed
    else:
        multiplexed = False

    qubit_pairs_batchable_list = _make_batchable_list_from_multiplexed(qubit_pairs, multiplexed)

    return qubit_pairs_batchable_list


def _get_qubit_pairs(machine: AnyQuam, node_parameters: TwoQubitExperimentNodeParameters) -> List[AnyTransmonPair]:
    if node_parameters.qubit_pairs is None or node_parameters.qubit_pairs == "":
        qubit_pairs = machine.active_qubit_pairs
    else:
        qubit_pairs = [machine.qubit_pairs[q] for q in node_parameters.qubit_pairs]

    return qubit_pairs


def _make_batchable_list_from_multiplexed(items: List, multiplexed: bool) -> BatchableList:
    if multiplexed:
        batched_groups = [[i for i in range(len(items))]]
    else:
        batched_groups = [[i] for i in range(len(items))]

    return BatchableList(items, batched_groups)
