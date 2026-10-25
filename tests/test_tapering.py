"""Tests for o2_tapering.py. Run with:  pytest -v tests/"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp  # noqa: E402
import pennylane as qml  # noqa: E402

import o2_tapering as ot  # noqa: E402


@pytest.fixture(scope="module")
def system():
    H, qubits = ot.build_o2_hamiltonian(r=2.30, active_electrons=2, active_orbitals=3)
    generators, paulixops = ot.get_symmetries(H, qubits)
    return H, qubits, generators, paulixops


def test_hamiltonian_shape(system):
    H, qubits, generators, _ = system
    assert qubits == 6
    assert len(H.terms()[0]) == 34
    assert len(generators) == 4


# Test 1: tapering is an exact block diagonalisation
def test_sector_union_equals_full_spectrum(system):
    H, qubits, _, _ = system
    sectors = ot.scan_all_sectors(H, qubits)
    assert len(sectors) == 16
    union = np.sort(np.concatenate([d["eigenvalues"] for d in sectors]))
    full = ot.exact_spectrum(H, qubits)
    assert union.shape == full.shape == (64,)
    assert np.allclose(union, full, atol=1e-8)


# Test 2: known sector energies and excitation energy
def test_sector_ground_energies(system):
    H, qubits, g, px = system
    e_t0 = ot.operator_eigenvalues(ot.taper_hamiltonian(H, qubits, ot.SECTOR_T0, g, px), 2)[0]
    e_t0_alt = ot.operator_eigenvalues(ot.taper_hamiltonian(H, qubits, ot.SECTOR_T0_ALT, g, px), 2)[0]
    e_s1 = ot.operator_eigenvalues(ot.taper_hamiltonian(H, qubits, ot.SECTOR_S1, g, px), 2)[0]

    assert e_t0 == pytest.approx(-147.632415, abs=1e-5)
    assert e_t0_alt == pytest.approx(-147.632415, abs=1e-5)   # T0 appears in both
    assert e_s1 == pytest.approx(-147.578487, abs=1e-5)
    assert (e_s1 - e_t0) * ot.HA_TO_EV == pytest.approx(1.4675, abs=1e-3)

    # global ground state of the full Hamiltonian is T0
    assert ot.exact_spectrum(H, qubits)[0] == pytest.approx(e_t0, abs=1e-8)


# Test 3: 2-qubit VQE reaches the exact sector ground energy
@pytest.mark.parametrize("sector", [ot.SECTOR_T0, ot.SECTOR_S1], ids=["T0", "S1"])
def test_2q_vqe_converges(system, sector):
    H, qubits, g, px = system
    H_2q = ot.taper_hamiltonian(H, qubits, sector, g, px)
    exact = ot.operator_eigenvalues(H_2q, 2)[0]
    res = ot.run_2q_vqe(H_2q, n_steps=400, lr=0.05, n_restarts=4, seed=0)
    assert abs(res["energy"] - exact) < 1e-4, (
        f"VQE {res['energy']:.8f} vs exact {exact:.8f}; restarts {res['restart_energies']}"
    )


# Test 4: destructive SWAP test returns |<psi|phi>|^2
def _state_2q(params):
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circ():
        ot.ansatz_2q(params)
        return qml.state()

    return np.asarray(circ())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_destructive_swap_matches_overlap_2q(seed):
    rng = np.random.default_rng(seed)
    p1 = jnp.array(rng.uniform(0, 2 * np.pi, ot.N_PARAMS_2Q))
    p2 = jnp.array(rng.uniform(0, 2 * np.pi, ot.N_PARAMS_2Q))
    expected = abs(np.vdot(_state_2q(p1), _state_2q(p2))) ** 2
    got = float(ot.destructive_swap_overlap(p1, p2))
    assert got == pytest.approx(expected, abs=1e-8)


def test_destructive_swap_limits():
    p = jnp.array(np.linspace(0.3, 2.0, ot.N_PARAMS_2Q))
    assert float(ot.destructive_swap_overlap(p, p)) == pytest.approx(1.0, abs=1e-8)

    def basis(bits, wires):
        qml.BasisState(np.array(bits), wires=wires)

    # |00> vs |11> are orthogonal
    assert float(ot.destructive_swap_overlap([0, 0], [1, 1], ansatz=basis, n_qubits=2)) \
        == pytest.approx(0.0, abs=1e-10)


def test_destructive_swap_matches_overlap_3q():
    rng = np.random.default_rng(7)

    def rand_state():
        v = rng.normal(size=8) + 1j * rng.normal(size=8)
        return v / np.linalg.norm(v)

    def prep(v, wires):
        qml.StatePrep(v, wires=wires)

    a, b = rand_state(), rand_state()
    expected = abs(np.vdot(a, b)) ** 2
    got = float(ot.destructive_swap_overlap(a, b, ansatz=prep, n_qubits=3))
    assert got == pytest.approx(expected, abs=1e-8)


def test_gate_counts_tapered_ansatz():
    counts = ot.circuit_gate_counts(2, 6)
    assert counts["(d) 2q tapered VQE"]["cnot"] == 1
    # destructive SWAP adds exactly 6 CNOTs on top of two copies of the ansatz
    a = counts["(a) 6q AllSinglesDoubles VQE"]["cnot"]
    c = counts["(c) 12q destructive SWAP"]["cnot"]
    assert c == 2 * a + 6
