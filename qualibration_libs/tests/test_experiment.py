"""Tests for exception handling in parameters/experiment.py."""

import pytest
from unittest.mock import Mock
from qualibration_libs.parameters.experiment import (
    get_qubits,
    get_qubit_pairs,
    QubitsExperimentNodeParameters,
    QubitPairExperimentNodeParameters,
)


class TestGetQubits:
    """Test exception handling in get_qubits function."""

    def test_invalid_qubit_name(self):
        mock_qubit1 = Mock(name="q1")
        mock_qubit2 = Mock(name="q2")
        mock_machine = Mock()
        mock_machine.qubits = {"q1": mock_qubit1, "q2": mock_qubit2}

        mock_node = Mock()
        mock_node.machine = mock_machine
        mock_node.parameters = QubitsExperimentNodeParameters(qubits=["q1", "invalid_qubit"])

        with pytest.raises(KeyError) as exc_info:
            get_qubits(mock_node)

        error_msg = str(exc_info.value)
        assert "Qubit 'invalid_qubit' not found" in error_msg
        assert "Available qubits:" in error_msg
        assert "'q1'" in error_msg
        assert "'q2'" in error_msg
        assert exc_info.value.__cause__ is not None

    def test_valid_qubit_names_work(self):
        mock_qubit1 = Mock(name="q1")
        mock_qubit2 = Mock(name="q2")
        mock_machine = Mock()
        mock_machine.qubits = {"q1": mock_qubit1, "q2": mock_qubit2}

        mock_node = Mock()
        mock_node.machine = mock_machine
        mock_node.parameters = QubitsExperimentNodeParameters(qubits=["q1", "q2"])

        result = get_qubits(mock_node)
        assert len(result) == 2

    def test_none_qubits_uses_active(self):
        mock_qubit1 = Mock(name="q1")
        mock_qubit2 = Mock(name="q2")
        mock_machine = Mock()
        mock_machine.qubits = {"q1": mock_qubit1, "q2": mock_qubit2}
        mock_machine.active_qubits = [mock_qubit1, mock_qubit2]

        mock_node = Mock()
        mock_node.machine = mock_machine
        mock_node.parameters = QubitsExperimentNodeParameters(qubits=None)

        result = get_qubits(mock_node)
        assert len(result) == 2


class TestGetQubitPairs:
    """Test exception handling in get_qubit_pairs function."""

    def test_invalid_pair_name(self):
        mock_pair1 = Mock(name="pair1")
        mock_pair2 = Mock(name="pair2")
        mock_machine = Mock()
        mock_machine.qubit_pairs = {"pair1": mock_pair1, "pair2": mock_pair2}

        mock_node = Mock()
        mock_node.machine = mock_machine
        mock_node.parameters = QubitPairExperimentNodeParameters(qubit_pairs=["pair1", "invalid_pair"])

        with pytest.raises(KeyError) as exc_info:
            get_qubit_pairs(mock_node)

        error_msg = str(exc_info.value)
        assert "Qubit pair 'invalid_pair' not found" in error_msg
        assert "Available qubit pairs:" in error_msg
        assert "'pair1'" in error_msg
        assert "'pair2'" in error_msg
        assert exc_info.value.__cause__ is not None

    def test_valid_pair_names_work(self):
        mock_pair1 = Mock(name="pair1")
        mock_pair2 = Mock(name="pair2")
        mock_machine = Mock()
        mock_machine.qubit_pairs = {"pair1": mock_pair1, "pair2": mock_pair2}

        mock_node = Mock()
        mock_node.machine = mock_machine
        mock_node.parameters = QubitPairExperimentNodeParameters(qubit_pairs=["pair1"])

        result = get_qubit_pairs(mock_node)
        assert len(result) == 1

    def test_none_pairs_uses_active(self):
        mock_pair1 = Mock(name="pair1")
        mock_machine = Mock()
        mock_machine.qubit_pairs = {"pair1": mock_pair1}
        mock_machine.active_qubit_pairs = [mock_pair1]

        mock_node = Mock()
        mock_node.machine = mock_machine
        mock_node.parameters = QubitPairExperimentNodeParameters(qubit_pairs=None)

        result = get_qubit_pairs(mock_node)
        assert len(result) == 1
