"""
NQCC Hackathon 2025 - Part 1 (Tutorial Mode)
Variational Quantum Deflation (VQD) for Molecular Oxygen (O2)
Ground State (T0 triplet) and First Excited State (S1 singlet)
"""

import sys
import numpy as np
import pennylane as qml
from pennylane import qchem
import jax
import jax.numpy as jnp
import optax
from functools import partial

# Configure JAX
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

def main():
    print("=" * 65)
    print("  NQCC Hackathon 2025: O2 VQE/VQD Ground & Excited State Simulation")
    print("=" * 65)

    # 1. System Setup
    symbols = ['O', 'O']
    r = 2.30  # Bond length in Bohr (~1.217 Angstroms, near equilibrium)
    coordinates = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, r]])
    active_electrons = 2
    active_orbitals = 3

    print(f"\n[1] Constructing Molecule and Hamiltonian...")
    print(f"    - Bond length r = {r} Bohr")
    print(f"    - Active electrons = {active_electrons}, Active orbitals = {active_orbitals}")
    
    molecule = qchem.Molecule(symbols, coordinates)
    H, qubits = qchem.molecular_hamiltonian(
        molecule,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        method='openfermion'
    )
    print(f"    - Qubits: {qubits}, Hamiltonian Pauli terms: {len(H.ops)}")

    # 2. Excitations and Hartree-Fock state
    singles, doubles = qchem.excitations(active_electrons, qubits)
    hf = qchem.hf_state(active_electrons, qubits)
    init_state = hf
    n_params = len(singles) + len(doubles)
    print(f"    - HF State: {hf}")
    print(f"    - Singles excitations ({len(singles)}): {singles}")
    print(f"    - Doubles excitations ({len(doubles)}): {doubles}")
    print(f"    - Total ansatz parameters: {n_params}")

    # Select backend device
    try:
        dev_vqe = qml.device("lightning.qubit", wires=qubits)
    except Exception:
        dev_vqe = qml.device("default.qubit", wires=qubits)

    @qml.qnode(dev_vqe, interface='jax')
    def vqe_circuit(parameters):
        qml.AllSinglesDoubles(parameters, range(qubits), init_state, singles, doubles)
        return qml.expval(H)

    # 3. Ground State VQE Optimization
    print(f"\n[2] Running VQE for Ground State (T0 Triplet)...")
    np.random.seed(42)
    init_params = jnp.array(np.random.rand(n_params))
    opt_vqe = optax.sgd(learning_rate=0.8)
    opt_state = opt_vqe.init(init_params)
    params = init_params
    prev_energy = 0.0

    def update_step(params, opt_state):
        grads = jax.grad(vqe_circuit)(params)
        updates, opt_state = opt_vqe.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    # Try JIT-compiling update step
    try:
        update_step_jit = jax.jit(update_step)
    except Exception:
        update_step_jit = update_step

    for i in range(101):
        energy = float(vqe_circuit(params))
        if i % 20 == 0:
            print(f"    Step {i:3d}: Energy = {energy:.8f} Ha")
        if abs(energy - prev_energy) < 1e-6 and i > 0:
            print(f"    --> Converged at step {i}!")
            break
        prev_energy = energy
        params, opt_state = update_step_jit(params, opt_state)

    ground_state_energy = energy
    ground_state_params = params
    print(f"    Final Ground State (T0) Energy: {ground_state_energy:.8f} Ha")

    # Verify ground state spin <S^2>
    S2 = qchem.spin2(active_electrons, qubits)
    @qml.qnode(dev_vqe, interface="jax")
    def s2_vqe(p):
        qml.AllSinglesDoubles(p, range(qubits), init_state, singles, doubles)
        return qml.expval(S2)

    s2_val_gs = float(s2_vqe(ground_state_params))
    print(f"    <S^2> expectation value: {s2_val_gs:.4f} (Expected ~2.0 for Triplet, S=1)")

    # 4. First Excited State (S1 Singlet) via VQD (SWAP Test)
    print(f"\n[3] Setting up VQD with SWAP-test overlap penalty for First Excited State (S1)...")

    def ansatz(theta, wires):
        mapped_singles = [[wires[idx] for idx in s] for s in singles]
        mapped_doubles = [[wires[idx] for idx in d] for d in doubles]
        qml.AllSinglesDoubles(theta, wires, hf, mapped_singles, mapped_doubles)

    try:
        dev_swap = qml.device("lightning.qubit", wires=2 * qubits + 1)
    except Exception:
        dev_swap = qml.device("default.qubit", wires=2 * qubits + 1)

    @qml.qnode(dev_swap, interface="jax")
    def swap_test(theta):
        ansatz(ground_state_params, range(1, qubits + 1))
        ansatz(theta, range(qubits + 1, 2 * qubits + 1))
        qml.Hadamard(wires=0)
        for i in range(qubits):
            qml.CSWAP(wires=[0, 1 + i + qubits, 1 + i])
        qml.Hadamard(wires=0)
        return qml.expval(qml.Z(0))

    @qml.qnode(dev_swap, interface="jax")
    def expected_value(theta):
        ansatz(theta, range(qubits))
        return qml.expval(H)

    beta = 2.0  # Orthogonality penalty coefficient

    def loss_f(theta, beta_val):
        return expected_value(theta) + beta_val * swap_test(theta)

    theta = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    opt_vqd = optax.sgd(learning_rate=0.4)
    opt_state_vqd = opt_vqd.init(theta)

    print(f"    Initial Overlap with Ground State: {float(swap_test(theta)):.4f}")
    print(f"    Running VQD optimization (beta = {beta})...")

    prev_loss = float(loss_f(theta, beta))
    for n in range(101):
        grad = jax.grad(loss_f)(theta, beta)
        updates, opt_state_vqd = opt_vqd.update(grad, opt_state_vqd)
        theta = optax.apply_updates(theta, updates)
        current_loss = float(loss_f(theta, beta))

        if n % 20 == 0:
            e_val = float(expected_value(theta))
            overlap = float(swap_test(theta))
            print(f"    Step {n:3d}: Loss = {current_loss:.8f} Ha | Energy = {e_val:.8f} Ha | Overlap = {overlap:.5f}")

        if abs(current_loss - prev_loss) < 1e-6 and n > 0:
            print(f"    --> Converged at step {n}!")
            break
        prev_loss = current_loss

    excited_energy = float(expected_value(theta))
    final_overlap = float(swap_test(theta))
    print(f"    Final Excited State (S1) Energy: {excited_energy:.8f} Ha")
    print(f"    Final Overlap with Ground State: {final_overlap:.6f} (~ 0 => orthogonal)")

    # Verify excited state spin <S^2>
    @qml.qnode(dev_swap, interface="jax")
    def s2_vqd(theta_p):
        ansatz(theta_p, range(qubits))
        return qml.expval(S2)

    s2_val_es = float(s2_vqd(theta))
    print(f"    <S^2> expectation value: {s2_val_es:.4f} (Expected ~0.0 for Singlet, S=0)")

    # 5. Exact Diagonalization Comparison
    print(f"\n[4] Exact Diagonalization Comparison:")
    H_mat = qml.matrix(H)
    exact_eigenvalues = np.sort(np.linalg.eigvalsh(H_mat))
    exact_gs = exact_eigenvalues[0]
    # Find lowest singlet (first excited state with distinct eigenvalue)
    exact_es = exact_eigenvalues[3]  # Triplet has 3 degenerate states in Sz
    exact_delta_hartree = exact_es - exact_gs
    exact_delta_ev = exact_delta_hartree * 27.2114

    print(f"    Exact Ground State Energy (T0): {exact_gs:.8f} Ha")
    print(f"    Exact First Excited Energy (S1): {exact_es:.8f} Ha")
    print(f"    Exact Vertical Excitation Energy: {exact_delta_hartree:.6f} Ha = {exact_delta_ev:.4f} eV")

    # 6. Final Results Summary
    delta_hartree = excited_energy - ground_state_energy
    delta_ev = delta_hartree * 27.2114

    print("\n" + "=" * 65)
    print("                     RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Ground State Energy (T0):    {ground_state_energy:12.6f} Ha  (<S^2> = {s2_val_gs:.3f}, Triplet)")
    print(f"  Excited State Energy (S1):   {excited_energy:12.6f} Ha  (<S^2> = {s2_val_es:.3f}, Singlet)")
    print(f"  Energy Difference (ΔE):      {delta_hartree:12.6f} Ha")
    print(f"  Vertical Excitation (VQD):   {delta_ev:12.4f} eV")
    print(f"  Vertical Excitation (Exact): {exact_delta_ev:12.4f} eV")
    print("-" * 65)
    print("  Benchmark Comparison:")
    print("    - VQD result (this run):     {:.3f} eV".format(delta_ev))
    print("    - README expected value:    ~1.467 eV")
    print("    - CCSD(T)/STO-3G:            1.334 eV")
    print("    - Experimental value:        0.980 eV")
    print("=" * 65)

if __name__ == "__main__":
    main()
