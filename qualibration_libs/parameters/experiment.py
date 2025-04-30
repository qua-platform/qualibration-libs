from typing import List, Optional, Literal
from qualibrate import QualibrationNode
from qualibrate.parameters import RunnableParameters
from qualibration_libs.core import BatchableList
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from quam_builder.architecture.superconducting.qpu import AnyQuam


class QubitsExperimentNodeParameters(RunnableParameters):
    qubits: Optional[List[str]] = None
    """A list of qubit names which should participate in the execution of the node. Default is None."""
    multiplexed: bool = False
    """Whether to play control pulses, readout pulses and active/thermal reset at the same time for all qubits (True)
    or to play the experiment sequentially for each qubit (False). Default is False."""
    use_state_discrimination: bool = False
    """Whether to use on-the-fly state discrimination and return the qubit 'state', or simply return the demodulated
    quadratures 'I' and 'Q'. Default is False."""
    reset_type: Literal["thermal", "active", "active_gef"] = "thermal"
    """The qubit reset method to use. Must be implemented as a method of Quam.qubit. Can be "thermal", "active", or
    "active_gef". Default is "thermal"."""

    def get_qubits(self, node: QualibrationNode) -> BatchableList[AnyTransmon]:
        """Get the batchable list of active qubits according to the multiplexed parameter.
        For instance [[machine.qubits["q1"]], [machine.qubits["q2"]], [machine.qubits["q3"]], [machine.qubits["q4"]]] for multiplexed==False
        and [machine.qubits["q1"], machine.qubits["q2"], machine.qubits["q3"], machine.qubits["q4"]] for multiplexed==True.
        """
        qubits = self._get_qubits(node.machine)

        if isinstance(node.parameters, QubitsExperimentNodeParameters):
            multiplexed = node.parameters.multiplexed
        else:
            multiplexed = False

        qubits_batchable_list = self._make_batchable_list_from_multiplexed(
            qubits, multiplexed
        )

        return qubits_batchable_list

    def _get_qubits(self, machine: AnyQuam) -> List[AnyTransmon]:
        if self.qubits is None or self.qubits == "":
            qubits = machine.active_qubits
        else:
            qubits = [machine.qubits[q] for q in self.qubits]

        return qubits

    @staticmethod
    def _make_batchable_list_from_multiplexed(
        items: List, multiplexed: bool
    ) -> BatchableList:
        if multiplexed:
            batched_groups = [[i for i in range(len(items))]]
        else:
            batched_groups = [[i] for i in range(len(items))]

        return BatchableList(items, batched_groups)
