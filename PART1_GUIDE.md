# Guide: NQCC Hackathon 2025 — Part 1 (Tutorial Mode)

This guide walks you through setting up, running, and understanding **Part 1 (Tutorial Mode)** of the [NQCC Hackathon 2025 repository](https://github.com/OpenBioSim/nqcc-hackathon-2025).

---

## 🎯 Goal of Part 1 (Tutorial Mode)

As specified in the hackathon challenge:
1. **Execute the VQE + VQD workflow** for molecular oxygen ($\text{O}_2$) near equilibrium geometry ($r = 2.30\text{ Bohr}$).
2. **Compute Ground State ($T_0$)**: Using VQE to obtain the triplet ground state energy ($E_{T_0} \approx -147.632\text{ Ha}$) and verify its spin $\langle S^2 \rangle \approx 2.0$.
3. **Compute First Excited State ($S_1$)**: Using VQD with a SWAP-test overlap penalty to obtain the first singlet excited state energy ($E_{S_1} \approx -147.577\text{ Ha}$) and verify its spin $\langle S^2 \rangle \approx 0.0$.
4. **Obtain Vertical Excitation Energy**:
   $$\Delta E = E_{S_1} - E_{T_0} \approx 0.0539\text{ Ha} \approx 1.467\text{ eV} \quad (1\text{ Ha} = 27.2114\text{ eV})$$
5. **Verify against Exact Diagonalization** and compare with classical benchmarks ($\text{CCSD(T)}$ and Experiment).

---

## 🚀 How to Run

You have two primary ways to run Part 1:

### Option A: Google Colab (Recommended by Hackathon Organizers)

Since Google Colab provides native Linux, free GPU access, and pre-configured environments, it avoids Windows compilation issues with chemistry libraries like `PySCF`.

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **Upload** and upload the newly prepared notebook:
   `nqcc-hackathon-2025/VQD-O2-colab.ipynb`
   *(Alternatively, you can open directly via the "Open in Colab" badge).*
3. Run the first cell to automatically install the dependencies:
   ```python
   !pip install -q pennylane pennylane-lightning openfermion openfermionpyscf pyscf jax optax
   ```
4. Click **Runtime** > **Run all** (or step through cells sequentially).

---

### Option B: Local Execution via WSL (Ubuntu)

On Windows, `openfermionpyscf` and `pyscf` require Linux. Since you have Ubuntu WSL2 installed, we prepared a setup script:

1. Open PowerShell and run:
   ```powershell
   wsl
   ```
2. Navigate to this directory:
   ```bash
   cd /mnt/c/Users/Yuko/Documents/ENS-PSL/nqcc-hackathon-2025
   ```
3. Run the automated setup script:
   ```bash
   ./setup_wsl.sh
   ```
4. Activate the environment and run either the standalone script or Jupyter:
   ```bash
   source ~/.envs/nqcc-hackathon/bin/activate
   # Run the full VQE/VQD workflow directly from CLI:
   python run_vqd.py

   # Or launch Jupyter Notebook:
   jupyter notebook --no-browser
   ```

---

## 📊 Expected Benchmark Results

| Method | Excitation Energy ($\Delta E$) | Spin State Description |
| :--- | :--- | :--- |
| **Exact Diagonalization (in active space)** | **1.467 eV** | $T_0$ ($\langle S^2 \rangle = 2$) $\to$ $S_1$ ($\langle S^2 \rangle = 0$) |
| **Your VQE + VQD Simulation** | **~1.467 – 1.514 eV** | Converged within $10^{-6}$ tolerance |
| **CCSD(T) / STO-3G** | **1.334 eV** | Classical Post-HF benchmark |
| **HF / STO-3G** | **2.278 eV** | Mean-field benchmark (spin contaminated) |
| **Experimental Value** | **0.980 eV** | Target physical spectroscopy value |

---

## 📁 Repository Structure Overview

- [`VQD-O2-colab.ipynb`](file:///c:/Users/Yuko/Documents/ENS-PSL/nqcc-hackathon-2025/VQD-O2-colab.ipynb): Turnkey Colab notebook with built-in installer, clean step annotations, and benchmark comparison table.
- [`VQD-O2-hackathon.ipynb`](file:///c:/Users/Yuko/Documents/ENS-PSL/nqcc-hackathon-2025/VQD-O2-hackathon.ipynb): Original hackathon starter notebook.
- [`run_vqd.py`](file:///c:/Users/Yuko/Documents/ENS-PSL/nqcc-hackathon-2025/run_vqd.py): Standalone Python script for head-to-head headless simulation and automated convergence check.
- [`setup_wsl.sh`](file:///c:/Users/Yuko/Documents/ENS-PSL/nqcc-hackathon-2025/setup_wsl.sh): Automated environment installer for Ubuntu WSL.
- [`requirements.txt`](file:///c:/Users/Yuko/Documents/ENS-PSL/nqcc-hackathon-2025/requirements.txt): Pip package specifications.
- [`environment.yml`](file:///c:/Users/Yuko/Documents/ENS-PSL/nqcc-hackathon-2025/environment.yml): Conda environment definition.
