"""
o2_tapering.py - Z2 qubit tapering, sector scan, destructive SWAP test and
2-qubit VQE for molecular oxygen (O2, STO-3G, CAS(2,3)).

Usage:
    python o2_tapering.py            # full report: sectors, 2q VQE, gate counts

Main functions:
    build_o2_hamiltonian(r, active_electrons, active_orbitals)
    taper_hamiltonian(H, qubits, sector)
    scan_all_sectors(H, qubits)
    destructive_swap_overlap(state1_params, state2_params, ansatz, n_qubits)
    run_2q_vqe(H_2q, n_steps, lr)
    circuit_gate_counts(active_electrons, qubits)
"""

import itertools
from collections import Counter

import jax

# float64 everywhere: without this, Hamiltonian coefficients are cast to
# complex64 and the VQE stalls ~1e-5 Ha above the exact energy.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax
import pennylane as qml
from pennylane import qchem

HA_TO_EV = 27.211386

# Generator order as returned by qml.symmetry_generators for this Hamiltonian:
#   Z0 Z1, Z2 Z3, Z0 Z2 Z4, Z0 Z2 Z5
SECTOR_T0 = (-1, -1, 1, 1)       # triplet ground state (also in (-1,-1,-1,-1))
SECTOR_T0_ALT = (-1, -1, -1, -1)
SECTOR_S1 = (1, 1, -1, -1)       # lowest singlet (= HF optimal_sector)

E_T0_EXACT = -147.632415   # Ha, exact CASCI
E_S1_EXACT = -147.578487   # Ha, exact CASCI


# ---------------------------------------------------------------------------
# Hamiltonian and tapering
# ---------------------------------------------------------------------------
def build_o2_hamiltonian(r=2.30, active_electrons=2, active_orbitals=3,
                         method="openfermion"):
    """Qubit Hamiltonian of O2 at bond length r (Bohr). Returns (H, n_qubits).

    Coefficients are cast to real floats so tapering and JAX work cleanly.
    """
    symbols = ["O", "O"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]])
    molecule = qchem.Molecule(symbols, coordinates)
    H, qubits = qchem.molecular_hamiltonian(
        molecule,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        method=method,
    )
    coeffs, ops = H.terms()
    H = qml.Hamiltonian(np.real(np.asarray(coeffs, dtype=complex)), ops)
    return H, qubits


def get_symmetries(H, qubits):
    """Z2 generators and the single-qubit PauliX ops used to taper them."""
    generators = qml.symmetry_generators(H)
    paulixops = qml.paulix_ops(generators, qubits)
    return generators, paulixops


def _to_consecutive_wires(H_t, qubits, paulixops):
    """Relabel the surviving wires of a tapered operator to 0..k-1."""
    removed = {px.wires[0] for px in paulixops}
    kept = [w for w in range(qubits) if w not in removed]
    n_kept = len(kept)
    if set(H_t.wires).issubset(set(range(n_kept))):
        return H_t, n_kept          # qml.taper already relabelled
    wire_map = {w: i for i, w in enumerate(kept)}
    return qml.map_wires(H_t, wire_map), n_kept


def taper_hamiltonian(H, qubits, sector, generators=None, paulixops=None):
    """Taper H into a symmetry sector (list of +-1, one per generator).

    Returns the tapered Hamiltonian acting on wires 0..k-1, with
    k = qubits - len(generators) (2 for O2 CAS(2,3)).
    """
    if generators is None or paulixops is None:
        generators, paulixops = get_symmetries(H, qubits)
    if len(sector) != len(generators):
        raise ValueError(f"sector needs {len(generators)} entries, got {len(sector)}")
    H_t = qml.taper(H, generators, paulixops, [int(s) for s in sector])
    H_t, _ = _to_consecutive_wires(H_t, qubits, paulixops)
    return H_t


def operator_eigenvalues(op, n_wires):
    """Sorted eigenvalues of op as a 2^n x 2^n matrix on wires 0..n-1."""
    mat = qml.matrix(op, wire_order=list(range(n_wires)))
    return np.sort(np.linalg.eigvalsh(np.asarray(mat)))


def exact_spectrum(H, qubits):
    return operator_eigenvalues(H, qubits)


def scan_all_sectors(H, qubits):
    """Exactly diagonalise H in every one of the 2^k symmetry sectors.

    Returns a list of dicts (sector, ground_energy, eigenvalues, H_tapered),
    sorted from lowest to highest ground energy.
    """
    generators, paulixops = get_symmetries(H, qubits)
    n_kept = qubits - len(generators)
    results = []
    for sector in itertools.product([1, -1], repeat=len(generators)):
        H_t = taper_hamiltonian(H, qubits, sector, generators, paulixops)
        eigs = operator_eigenvalues(H_t, n_kept)
        results.append({
            "sector": tuple(sector),
            "ground_energy": float(eigs[0]),
            "eigenvalues": eigs,
            "H_tapered": H_t,
        })
    results.sort(key=lambda d: d["ground_energy"])
    return results


# ---------------------------------------------------------------------------
# 2-qubit ansatz and VQE
# ---------------------------------------------------------------------------
N_PARAMS_2Q = 8


def ansatz_2q(params, wires=(0, 1)):
    """General 2-qubit state preparation with a single CNOT.

    Any 2-qubit pure state (complex amplitudes included) is reachable: by the
    Schmidt decomposition one CNOT between local rotations is enough. The old
    RY-only version was limited to real states and had no RZ phases.
    """
    w0, w1 = wires
    qml.RY(params[0], wires=w0)
    qml.RZ(params[1], wires=w0)
    qml.RY(params[2], wires=w1)
    qml.RZ(params[3], wires=w1)
    qml.CNOT(wires=[w0, w1])
    qml.RY(params[4], wires=w0)
    qml.RZ(params[5], wires=w0)
    qml.RY(params[6], wires=w1)
    qml.RZ(params[7], wires=w1)




def make_2q_cost(H_2q):
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, interface="jax")
    def cost(params):
        ansatz_2q(params)
        return qml.expval(H_2q)

    return cost


def run_2q_vqe(H_2q, n_steps=400, lr=0.05, n_restarts=4, seed=0, tol=1e-12,
               verbose=False):
    """VQE on a tapered 2-qubit Hamiltonian with Adam + random restarts.

    Returns dict(energy, params, history, restart_energies). `history` is the
    energy trace of the best restart.
    """
    cost = make_2q_cost(H_2q)
    opt = optax.adam(lr)

    @jax.jit
    def step(params, opt_state):
        energy, grads = jax.value_and_grad(cost)(params)
        updates, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, energy

    rng = np.random.default_rng(seed)
    best = None
    restart_energies = []
    for restart in range(n_restarts):
        params = jnp.array(rng.uniform(0, 2 * np.pi, N_PARAMS_2Q))
        opt_state = opt.init(params)
        history = []
        for it in range(n_steps):
            params, opt_state, energy = step(params, opt_state)
            history.append(float(energy))
            if it > 20 and abs(history[-1] - history[-2]) < tol:
                break
        final = float(cost(params))
        history.append(final)
        restart_energies.append(final)
        if verbose:
            print(f"  restart {restart}: E = {final:.10f} Ha after {len(history)-1} steps")
        if best is None or final < best["energy"]:
            best = {"energy": final, "params": params, "history": history}
    best["restart_energies"] = restart_energies
    return best


# ---------------------------------------------------------------------------
# Destructive SWAP test
# ---------------------------------------------------------------------------
def swap_parity_vector(n_qubits):
    """(-1)^(sum_i a_i b_i) for each basis state |a>|b> of 2n qubits.

    qml.probs is big-endian: wire i is bit (2n-1-i) of the index.
    """
    n = 2 * n_qubits
    k = np.arange(2 ** n)
    parity = np.ones(2 ** n)
    for i in range(n_qubits):
        a = (k >> (n - 1 - i)) & 1
        b = (k >> (n_qubits - 1 - i)) & 1
        parity *= np.where(a & b, -1.0, 1.0)
    return parity


def make_destructive_swap(ansatz=ansatz_2q, n_qubits=2):
    """Return overlap(p1, p2) = |<psi(p1)|psi(p2)>|^2 via a destructive SWAP test.

    `ansatz(params, wires)` prepares a state on `wires`. Uses 2n wires, n CNOTs
    and n Hadamards, no ancilla and no CSWAP.
    """
    dev = qml.device("default.qubit", wires=2 * n_qubits)
    parity = jnp.array(swap_parity_vector(n_qubits))

    @qml.qnode(dev, interface="jax")
    def circuit(p1, p2):
        ansatz(p1, list(range(n_qubits)))
        ansatz(p2, list(range(n_qubits, 2 * n_qubits)))
        for i in range(n_qubits):
            qml.CNOT(wires=[i, i + n_qubits])
            qml.Hadamard(wires=i)
        return qml.probs(wires=range(2 * n_qubits))

    def overlap(p1, p2):
        return jnp.dot(parity, circuit(p1, p2))

    return overlap


def destructive_swap_overlap(state1_params, state2_params, ansatz=ansatz_2q,
                             n_qubits=2):
    """|<psi(state1_params)|psi(state2_params)>|^2 from a destructive SWAP test."""
    return make_destructive_swap(ansatz, n_qubits)(state1_params, state2_params)


# ---------------------------------------------------------------------------
# Gate counts
# ---------------------------------------------------------------------------
GATE_SET = {
    "CNOT", "RX", "RY", "RZ", "Hadamard", "PauliX", "PauliY", "PauliZ",
    "S", "Adjoint(S)", "T", "Adjoint(T)", "SX", "PhaseShift", "GlobalPhase",
}


def count_gates(qfunc, *args):
    """Decompose qfunc into {CNOT + 1-qubit gates} and count gates by name."""
    tape = qml.tape.make_qscript(qfunc)(*args)
    try:
        (tape,), _ = qml.transforms.decompose(tape, gate_set=GATE_SET)
    except TypeError:  # very old PennyLane without gate_set
        tape = tape.expand(depth=10, stop_at=lambda op: op.name in GATE_SET)
    counts = Counter(op.name for op in tape.operations)
    counts.pop("GlobalPhase", None)
    n_two_qubit = sum(1 for op in tape.operations if len(op.wires) == 2)
    return {
        "total": sum(counts.values()),
        "cnot": counts.get("CNOT", 0),
        "two_qubit": n_two_qubit,
        "by_type": dict(counts),
    }


def uccsd_ansatz_factory(active_electrons, qubits):
    """AllSinglesDoubles ansatz on arbitrary wires, as in the notebooks."""
    singles, doubles = qchem.excitations(active_electrons, qubits)
    hf = qchem.hf_state(active_electrons, qubits)

    def ansatz(theta, wires):
        wires = list(wires)
        s = [[wires[i] for i in ex] for ex in singles]
        d = [[wires[i] for i in ex] for ex in doubles]
        qml.AllSinglesDoubles(theta, wires, hf, s, d)

    return ansatz, len(singles) + len(doubles)


def circuit_gate_counts(active_electrons=2, qubits=6):
    """Gate counts, fully decomposed to CNOT + single-qubit gates, of:

    (a) baseline 6-qubit AllSinglesDoubles VQE state preparation
    (b) 13-qubit VQD overlap circuit with ancilla + 6 CSWAPs
    (c) 12-qubit destructive SWAP overlap circuit
    (d) 2-qubit tapered VQE ansatz
    """
    ansatz, n_params = uccsd_ansatz_factory(active_electrons, qubits)
    theta = np.linspace(0.1, 0.8, n_params)
    q = qubits

    def vqe_6q():
        ansatz(theta, range(q))

    def cswap_13q():
        ansatz(theta, range(1, q + 1))
        ansatz(theta, range(q + 1, 2 * q + 1))
        qml.Hadamard(wires=0)
        for i in range(q):
            qml.CSWAP(wires=[0, 1 + i + q, 1 + i])
        qml.Hadamard(wires=0)

    def destructive_12q():
        ansatz(theta, range(q))
        ansatz(theta, range(q, 2 * q))
        for i in range(q):
            qml.CNOT(wires=[i, i + q])
            qml.Hadamard(wires=i)

    def tapered_2q():
        ansatz_2q(np.linspace(0.1, 0.8, N_PARAMS_2Q))

    return {
        "(a) 6q AllSinglesDoubles VQE": dict(wires=q, **count_gates(vqe_6q)),
        "(b) 13q CSWAP overlap": dict(wires=2 * q + 1, **count_gates(cswap_13q)),
        "(c) 12q destructive SWAP": dict(wires=2 * q, **count_gates(destructive_12q)),
        "(d) 2q tapered VQE": dict(wires=2, **count_gates(tapered_2q)),
    }


def print_gate_table(counts):
    print(f"{'Circuit':34s} {'wires':>5s} {'CNOT':>6s} {'2q gates':>9s} {'total':>7s}")
    for name, c in counts.items():
        print(f"{name:34s} {c['wires']:5d} {c['cnot']:6d} {c['two_qubit']:9d} {c['total']:7d}")


# ---------------------------------------------------------------------------
def main():
    H, qubits = build_o2_hamiltonian()
    print(f"O2 STO-3G CAS(2,3): {qubits} qubits, {len(H.terms()[0])} Pauli terms")

    generators, paulixops = get_symmetries(H, qubits)
    print("Generators:", [str(g) for g in generators])
    print("Tapered wires:", [px.wires[0] for px in paulixops])

    full = exact_spectrum(H, qubits)
    print(f"\nExact 6-qubit lowest eigenvalues: {np.round(full[:6], 8)}")

    print("\nSector scan (ground energy per sector):")
    for d in scan_all_sectors(H, qubits):
        print(f"  {str(list(d['sector'])):18s} {d['ground_energy']:.6f} Ha")

    print("\n2-qubit VQE (Adam, x64, 4 restarts):")
    results = {}
    for name, sector, exact in [("T0", SECTOR_T0, E_T0_EXACT),
                                ("S1", SECTOR_S1, E_S1_EXACT)]:
        H_2q = taper_hamiltonian(H, qubits, sector, generators, paulixops)
        res = run_2q_vqe(H_2q)
        exact_sector = operator_eigenvalues(H_2q, 2)[0]
        results[name] = res["energy"]
        print(f"  {name} {list(sector)}: VQE {res['energy']:.8f}  exact {exact_sector:.8f}  "
              f"error {abs(res['energy'] - exact_sector):.2e} Ha")
    dE = results["S1"] - results["T0"]
    print(f"  Delta E = {dE:.6f} Ha = {dE * HA_TO_EV:.4f} eV "
          f"(exact {(E_S1_EXACT - E_T0_EXACT) * HA_TO_EV:.4f} eV)")

    print("\nGate counts (decomposed to CNOT + 1-qubit gates):")
    print_gate_table(circuit_gate_counts(2, qubits))


if __name__ == "__main__":
    main()
