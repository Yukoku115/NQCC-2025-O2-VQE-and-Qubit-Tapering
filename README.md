# NQCC-2025-O2-VQE-and-Qubit-Tapering

## Qubit Tapering and Circuit Resource Reduction for Molecular Oxygen (O₂)

A self-study project built on the **NQCC 2025 Hackathon** challenge (OpenBioSim / Alice & Bob), forked from [OpenBioSim/nqcc-hackathon-2025](https://github.com/OpenBioSim/nqcc-hackathon-2025).

**Goal:** compute the T₀ → S₁ vertical excitation energy of O₂ and study how symmetry-based Hamiltonian reduction (ℤ₂ qubit tapering) and overlap-test simplifications reduce the qubit count and entangling-gate cost. The focus is on **resources**, not on improving chemical accuracy beyond the given active-space model.

All PennyLane VQE/VQD and tapering work is in [`VQD-O2-colab.ipynb`](VQD-O2-colab.ipynb), with standalone tapering utilities in [`o2_tapering.py`](o2_tapering.py) and unit tests in [`tests/test_tapering.py`](tests/test_tapering.py). Hardware emulation using Alice & Bob's Qiskit provider (`EMU:15Q:LOGICAL_EARLY`) is in [`alice_bob_emulation.ipynb`](alice_bob_emulation.ipynb). A derivation of the tapering pipeline is in [`QUBIT_TAPERING_GUIDE.md`](QUBIT_TAPERING_GUIDE.md).

---

### Model
- O₂ at r = 2.30 Bohr, STO-3G basis, active space of 2 electrons in 3 spatial orbitals → **6 qubits** (Jordan–Wigner), 34 Pauli terms.

### Results

| | Baseline VQD (CSWAP overlap test) | Task 1: Destructive SWAP VQD | Tasks 2–3: Tapered, sector-resolved |
| :--- | :---: | :---: | :---: |
| Qubits per state | 6 | 6 | **2** |
| Total wires for excited-state run | 13 (12 + ancilla) | 12 | **2** |
| Overlap-test gates | 6 CSWAPs | 6 CNOTs + 6 H | **None needed** (states lie in different symmetry sectors) |
| Ansatz | `AllSinglesDoubles`, 8 parameters | same | 2-qubit `RY-RZ-CNOT-RY-RZ`, 8 parameters, **1 CNOT** |
| E(T₀) / Ha | −147.63229 (VQE) | — | −147.632415 (exact) / **−147.632390** (2Q VQE) |
| E(S₁) / Ha | −147.57663 (VQD) | — | −147.578487 (exact) / **−147.578487** (2Q VQE) |
| ΔE(T₀→S₁) / eV | 1.515 | — | **1.4675** (exact & 2Q VQE) |

Reference values: 1.467 eV is the exact result for this active space (the challenge's stated value); experiment is 0.980 eV.

---

### What was done
- **Destructive SWAP test (Task 1):** replaced the ancilla-based CSWAP overlap test with a Bell-basis destructive SWAP test (transversal CNOT + H, followed by classical parity post-processing). This removes the ancilla and all controlled-SWAP gates.
- **ℤ₂ symmetry tapering (Task 2):** used PennyLane's `qml.symmetry_generators`, `qml.paulix_ops` and `qml.taper` to find **4 independent ℤ₂ symmetry generators** and taper the Hamiltonian from 6 qubits to 2. These routines find the generators via binary (𝔽₂) linear algebra and apply the Clifford rotations U = ∏ (τᵢ + X_kᵢ)/√2.
- **Exhaustive sector scan:** PennyLane's `optimal_sector` picks the sector from a closed-shell Hartree–Fock reference, which doesn't suit the open-shell triplet ground state of O₂. Instead, all 2⁴ = 16 sectors were tapered and diagonalized exactly:
  - The lowest energy, −147.632415 Ha (T₀, 4 Pauli terms), appears in sectors `[-1,-1,1,1]` and `[-1,-1,-1,-1]` and is doubly degenerate within each sector.
  - −147.578487 Ha (S₁, 10 Pauli terms) appears in sector `[1,1,-1,-1]`. This is identified as S₁ by comparison with the baseline VQD singlet energy.
- **Sector-resolved VQE (Task 3):** because T₀ and S₁ lie in different symmetry sectors, they are orthogonal. Each can be found by an independent 2-qubit VQE using an 8-parameter 2-qubit circuit (`RY-RZ-CNOT-RY-RZ`, 1 CNOT) with no overlap penalty or SWAP test.
- **Alice & Bob Emulator Testing ([`alice_bob_emulation.ipynb`](alice_bob_emulation.ipynb), WIP):**
  - Rebuilt the 2-qubit circuit in Qiskit and transpiled it for Alice & Bob's logical emulator (`EMU:15Q:LOGICAL_EARLY`).
  - **Gate count expansion:** Because logical qubits only run a discrete set of gates (`H, S, T, CX`), Qiskit has to approximate the 8 continuous rotation angles (`RY, RZ`) using long sequences of discrete gates. This turns a 9-gate circuit into **2,462 gates** (including **1,061 $T/T^\dagger$ gates** and **1,098 $H$ gates**).
  - **Effect of hardware noise:** In an ideal noiseless simulation, the VQE angles for T₀ give only `'00'` (~49%) and `'11'` (~51%), so $\langle Z_0 Z_1 \rangle = +1.0$ and the energy is **−147.632 Ha**. On `EMU:15Q:LOGICAL_EARLY` with its default noise settings, errors build up across the 2,462 gates and randomize the output (~25% for each state), pushing $\langle Z_0 Z_1 \rangle$ to ~0 and the measured energy to **−146.53 Ha**.
  - **Next steps:** Test the noiseless logical backend (`EMU:40Q:LOGICAL_NOISELESS`), adjust the emulator's noise parameters (`average_nb_photons`, `distance`, `kappa_1/kappa_2`) to see when the clean `'00'`/`'11'` counts return, and test a simpler circuit without $T$-gates.

---

### Limitations & Ongoing Work
- **Model accuracy vs. experiment:** 1.4675 eV is the exact answer *for this active space*, not the experimental value (0.980 eV). The gap comes from the minimal basis and small active space, not from the quantum algorithm. Improving it was outside the scope of this project.
- **Sectors were chosen by classical diagonalization.** That is only feasible because the system is tiny. For larger systems, sectors must be fixed from physical quantum numbers (particle number, S_z, spatial symmetry) or from the reference state.
- **S₁ identification relies on matching energies.** ⟨S²⟩ has not yet been evaluated in the tapered space; the S² operator should be tapered too to confirm the spin of each state.
- **Same-sector excited states:** if an excited state shares a sector with the ground state, sector separation can't isolate it, and an overlap method (e.g. destructive SWAP VQD) is still needed inside the tapered space.
- **$T$-gate overhead on the logical emulator (in progress):** transpiling continuous VQE angles into discrete gates creates over 1,000 $T$-gates, which is too deep for the default noise level of `EMU:15Q:LOGICAL_EARLY`. Reducing this gate count or tuning the hardware noise parameters is the next step.
- **Tapering is a Hamiltonian-level reduction.** It isn't a general circuit-compilation pass and relies on symmetries specific to the problem.

---

# Original Challenge Description (from OpenBioSim)
Code for OpenBioSim's use case for the NQCC 2025 Hackathon

This repository contains a Pennylane script to compute the energies of the ground state and the first excited state of molecular oxygen using variational quantum algorithms. 

# Installation instructions

We encourage you to run your code inside the Google Colab environment that will be provided for you. If you wanted to run the notebooks on your laptop, you can follow the instructions below, that have been tested on Linux/Ubunu 20.04.6 LTS.

```
mamba create -n hackathon python==3.12
mamba activate hackhaton
pip install custatevec_cu12 pennylane-lightning-gpu pennylane-catalyst jax==0.4.28 jaxlib==0.4.28 jax-cuda12-plugin==0.4.28 optax notebook matplotlib jax[cuda12_pip]==0.4.28 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install openfermionpyscf
git clone https://github.com/OpenBioSim/nqcc-hackathon-2025
cd nqcc-hackathon-2025/
jupyter-notebook VQD-O2-hackathon.ipynb
```


# Scientific Background

Quantum computers offer the potential to achieve highly accurate models of the electronic structure of molecules, addressing the challenges posed by the steep computational cost of post-Hartree–Fock wavefunction methods. However, in the current era of noisy intermediate-scale quantum (NISQ) hardware, it is essential to carefully optimize quantum circuits to minimize the number of gate operations, enabling reliable execution on existing devices.

In this hackathon, you will take on the challenge of computing the energies of the ground state and low-lying excited states of molecular dioxygen (O₂) using quantum algorithms. O2 is the second most abundant molecule in Earth’s atmosphere. In its electronic ground state, O₂ exists as a triplet state, meaning two electrons are unpaired.  O₂ can also adopt  two distinct reactive singlet states. These singlet states are short-lived and highly reactive, playing crucial roles in atmospheric chemistry, healthcare technologies (e.g., photodynamic cancer therapies), and materials science (e.g., battery corrosion).

![Figure 1. MO diagrams for O2 in its ground state and first two-excited states -source Wikipedia-](images/figure1.png)
**Figure 1.** MO diagrams for O2 in its ground state and first two-excited states -source Wikipedia-

We can predict the optimal wavelength of a photon required to excite O₂ from its electronic ground state to an excited state by computing vertical excitation energies—the energy differences between the ground and excited states at the same nuclear geometry (regardless of whether the process would occur with high efficiency). These energies are obtained by solving the time-independent Schrödinger equation for each state at a fixed geometry. At room temperature, O₂ in its ground state predominantly occupies geometries near its equilibrium bond length. Therefore, calculating vertical excitation energies at this equilibrium geometry provides a good estimate of the energy gap between the ground and excited states.

![Figure 2. Schematic potential energy surfaces for three electronic states of O2, the ground state T0, the first excited state S1, the second excited state S2. -source Wikipedia-](images/figure2.png)

**Figure 2.** Schematic potential energy surfaces for three electronic states of O2, the ground state T0, the first excited state S1, the second excited state S2. -source Wikipedia-

We can gain deeper insights into the properties of O₂ by computing the energies of different electronic states as a function of bond length. As the O₂ bond is stretched, its electronic structure becomes increasingly multireference in character, and classical electronic structure methods like CCSD(T) begin to struggle with wavefunction convergence.
Below, you can see potential energy surface plots for various O₂ states, generated using different classical computational chemistry methods. These calculations were performed using the ORCA 6 software package. The Hartree–Fock solutions are poor, exhibiting significant spin contamination as the geometry deviates from the equilibrium bond length. The CCSD(T) calculations were carried out until the SCF solver failed to converge. As the basis set becomes more complex (from STO-3G to def2-SVP to cc-pVTZ), the computed potential energy values decrease, in accordance with the variational principle. The raw data is available in the GitHub repository, should you wish to replot any of the results alongside those from the variational quantum algorithms (VQAs).

![Figure 3. Classical benchmarks for computing potential energy surfaces of O2 in its ground state T0 or first excited state S1. ](images/figure3.png)
**Figure 3.** Classical benchmarks for computing potential energy surfaces of O2 in its ground state T0 or first excited state S1. 

From these plots we have estimated the following vertical excitation energy between the ground state and the first singlet state. They are given in electron volts (eV) for consistency with the spectroscopy literature (1 Hartree = 27.2114 eV).

## Table 1. Vertical excitation energies for the T0->S1 transition. 

| Method    | Excitation energy / eV |
| -------- | ------- |
| Experiment  | 0.980  |
| HF/STO-3G   | 2.278  |
| HF/DEF2-SVP | 2.321  |
| CCSD(T)/STO-3G | 1.334  |
| CCSD(T)/DEF2-SVP | 1.401  |
| CCSD(T)/cc-pVTZ | 1.360 |

The provided PennyLane notebook implements Variational Quantum Eigensolver (VQE) and Variational Quantum Deflation (VQD) to optimize the energies of O₂ in its ground state and first excited state, using a small active space and a minimal basis set. With the default settings, it yields a vertical excitation energy of 1.467 eV, which is slightly less accurate than the results from CCSD(T) calculations.

# Hardware Architecture

For this hackathon, you will be using Alice & Bob emulators. These emulators reproduce the behavior of quantum computers based on cat qubits, superconducting qubits that are naturally protected against one type of noise (bit-flip errors).

## About cat qubits

You can get a broad overview of how cat qubits work in the third part of our [whitepaper](https://alice-bob.com/wp-content/uploads/2024/12/Think-Inside-The-Box-Alice-Bob-Whitepaper.pdf), and you can play with the interesting properties of these qubits in the following [tutorial](https://github.com/Alice-Bob-SW/felis/blob/main/samples/An%20introduction%20to%20cat%20qubits.ipynb) (note that going from physical qubits to logical qubits by explicitly implementing quantum error correction is outside the scope of this hackathon).

For a deeper dive, a selection of relevant scientific articles is available at https://felis.alice-bob.com/docs/reference/papers/.

## About Alice & Bob emulators

Alice & Bob's Felis library lets you use cat qubits emulator under the form of Qiskit backends reproducing the gate set and noise model of quantum computers based on cat qubits: https://github.com/Alice-Bob-SW/qiskit-alice-bob-provider. Note that we provide a Pennylane connector so that you don't necessarily need to work with Qiskit.

Two types of emulators are available: "physical" and "logical". As the "physical" emulators are focused on running error correction experiments and do not feature a universal gate set, make sure you use "logical" emulators, which behave as an error-corrected (but not error-free) quantum computer would. For more about the distinction, you may read https://felis.alice-bob.com/docs/backends/logical_physical/.

"Logical" emulators have two important features that will require you to adapt your algorithms:
- They feature a discrete gate set, meaning your circuits must first be transpiled to this gate set in order to run. To test this feature in isolation, you may use the `EMU:40Q:LOGICAL_NOISELESS` backend (https://felis.alice-bob.com/docs/backends/backends_list/logical_noiseless/), which does not have a noise model and should therefore return the same results as your favorite generic emulator.
- They feature a noise model, meaning your results will feature errors. This model depends on several parameters: `distance`, `kappa_1`, `kappa_2` and `average_nb_photons`. You can read more about these parameters at https://felis.alice-bob.com/docs/backends/set_parameters/. Backends such as `EMU:15Q:LOGICAL_EARLY` (https://felis.alice-bob.com/docs/backends/backends_list/logical_early/) have preset values for these parameters, but you can override the preset values by specifying your desired value upon backend creation.


# Hackathon challenge! Can you complete as many as possible of the objectives below?


## Tutorial mode

Can you execute the provided notebook and obtain a similar vertical excitation energy mentioned below (ca. 1.467 ev)?  See also tutorial #1 for additional insights in VQE and VQD. 

**Google cloud: Can you provide instructions/starting points to execute the provided Pennylane notebook on a Google colab.**

## Scientific Objectives

These objectives assess how well the quantum simulations reproduce experimental data—a key step in validating more complex future simulations of O₂.

•	S1) Can you optimize the algorithm to compute a vertical excitation energy for the first excited state that agrees within 0.1–0.2 eV of the experimental value?

•	S2) Can you compute an accurate vertical excitation energy for the second excited state? Experimental data shows that the S₂ singlet lies 1.627 eV above the T₀ ground state.

•	S3) Can you compute an accurate bond dissociation energy for O₂ in its T₀ ground state? This can be determined by measuring the difference in potential energy between O₂ at its equilibrium bond length and at a large separation of the two oxygen atoms, where no chemical bonding occurs.

•	Bonus – S4) Can you compute potential energy scans for O₂ in the S₁ and S₂ states and recover the energy values of dissociated O₂ in its ground state? From this, you can also estimate adiabatic excitation energies and compare them to vertical excitation energies.

These objectives can be pursued using algorithms that may offer limited accuracy. However, it is preferable to optimize and deploy algorithms that demonstrate competitive performance with classical quantum chemistry methods.

## Technical Objectives

These objectives assess how well you can adapt your work to some of the constraints introduced by error-corrected quantum hardware. Read above for an introduction to Alice & Bob's hardware and emulators.

•	T1) Can you modify the provided code to deploy variational quantum algorithms on Alice and Bob emulators? Can you optimize algorithmic choices to reduce the size of transpiled circuits and improve speed, without compromising accuracy?

Tips:

- While the emulators are Qiskit backends, Pennylane has a Qiskit connector that you can use. See https://github.com/amoutenet/pennylane-alice-bob for a sample implementation.
- Transpiling will be an issue, as targeting the gate set of logical qubits requires knowing the exact value of the parametrized gates typically used in variational circuits. You may need to change your algorithm's behavior so it transpiles right before running the circuit. This will likely affect your algorithm's speed (both because transpilation needs to be done several times, and because circuits will be deeper).

•	T2) Compare the results of the VQAs run via noiseless emulation to those obtained on noisy backends. Can you develop error mitigation strategies to reduce the impact of noise on computational accuracy?

Tips:

- To compare noisy against noiseless results, you may draw inspiration from the following notebook: https://github.com/Alice-Bob-SW/felis/blob/main/samples/2_algorithms/4%20-%20Benchmarking%20a%20logical%20cat%20qubit%20processor%20through%20the%20SWAP%20test.ipynb. It shows how to explore the parameter space of the noise model and plot the results to identify acceptable combinations. Note that real hardware will have a fixed $\kappa_1/\kappa_2$ value, flexible `average_nb_photons` and might have some control over `distance`. It can be interesting to identify a $\kappa_1/\kappa_2$ value giving mixed results and identify what distance and number of photons should be used, knowing that a larger distance means more qubits on the chip.

•	Bonus – T3) From NISQ to FTQC: Variational algorithms were first developed to take advantage of NISQ machines. Based on your above experiments, can you understand why, and what kind of limitations they face? A typical fault-tolerant algorithm to determine eigenvalues is the Quantum Phase Estimation (QPE). Can you use such a method, rather than VQE, to reproduce or even outperform some of the scientific objectives above?

Tips:

- If the resulting circuit is too large to be emulated, you may explore the Alice & Bob resource estimator at https://github.com/Alice-Bob-SW/qsharp-alice-bob-resource-estimator

## Hints / Suggestions

•	Break the work into independent tasks that can be pursued individually and consolidated later. For example: scientific accuracy, hardware deployment, and circuit optimization.

•	Be cautious of spin contamination. The ⟨S²⟩ expectation value of the optimized states can help confirm whether you have converged to a triplet or singlet state.

•	To improve accuracy, consider:
i) Increasing the active space size
ii) Using a larger basis set
Larger active spaces that include more virtual orbitals are expected to yield more accurate results but may quickly exceed the capabilities of emulators or quantum hardware. You might try using a differential Hartree–Fock solver to optimize orbital coefficients, allowing for more accurate results with a smaller basis set. See Tutorial #5 for guidance.

•	To reduce qubit counts or circuit size, consider Hamiltonian optimizations such as exploiting Z₂ symmetries. See Tutorial #4 for examples. Additional Hamiltonian compression methods are discussed in Tutorial #3.

•	To compute the second excited state, you can extend the VQD code by adding another orthogonality constraint to the loss function. You may also be able to reduce circuit complexity by replacing the standard SWAP test with a destructive SWAP test. Alternatively, consider implementing a different algorithm, such as FoldedVQE.

•	As the O₂ bond is stretched, the Hartree–Fock initial state may exhibit poor overlap with the true wavefunction. In such cases, alternative classical state preparation methods (e.g., CCSD) may provide a better starting point. See Tutorial #2 for inspiration.

# Relevant Pennylane tutorials 

1)	https://pennylane.ai/qml/demos/tutorial_vqe_vqd
2)	https://pennylane.ai/qml/demos/tutorial_initial_state_preparation
3)	https://pennylane.ai/qml/demos/tutorial_how_to_build_compressed_double_factorized_hamiltonians
4)	https://pennylane.ai/qml/demos/tutorial_qubit_tapering
5)	https://pennylane.ai/qml/demos/tutorial_differentiable_HF

# Relevant references

Higgott, Oscar and Wang, Daochen and Brierley, Stephen “Variational Quantum Computation of Excited States” Quantum 3, 156 (2019)

Folded Spectrum VQE: A Quantum Computing Method for the Calculation of Molecular Excited States Lila Cadi Tazi and Alex J. W. Thom Journal of Chemical Theory and Computation 2024 20 (6), 2491-2504 DOI: 10.1021/acs.jctc.3c01378


Happy hacking !!
