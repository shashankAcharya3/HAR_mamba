# HAR-Mamba: Physics-Informed Self-Supervised HAR via Bidirectional State Space Models

## 1. Project Overview & Goal
**Objective:** Build a clean-slate, state-of-the-art Human Activity Recognition (HAR) framework from scratch. 
**The Pitch:** We propose a physics-informed, self-supervised HAR framework using a Bidirectional Mamba encoder. By introducing a cross-sensor kinematic prior and using motion-aware probabilistic masking (rather than random masking), we force the model to learn highly semantic movement transitions, drastically reducing the reliance on massive labeled datasets.

**Dataset:** Initially, we are targeting the **UCI HAR** dataset for rapid prototyping, but the dataloader must be modular enough to swap in PAMAP2 later. reference it in the directory.

---

## 2. The Three Core Novelties (Strict Requirements)

### Pillar 1: Kinematic Interaction Tensor (KIT)
Standard models just concatenate accelerometer ($a_t$) and gyroscope ($\omega_t$) data. We must compute a new physical prior to represent rotational jerk/Coriolis-like effects.
* **Implementation:** The dataloader/preprocessing step must dynamically compute the cross product: $F_{int,t} = a_t \times \omega_t$.
* **Input Vector:** The final input to the model per timestep is the concatenation of $[a_t, \omega_t, F_{int,t}]$.

### Pillar 2: Probabilistic Dynamics-Guided Masking (PDGM)
We are explicitly abandoning standard Random Masking (like MAE).
* **Implementation:** Calculate the motion intensity (gradient/norm) for a given patch: $||\Delta x_t||$.
* **Mechanism:** Convert these intensities into a probability distribution. High-dynamic patches must have a *higher probability* of being masked. 
* **Goal:** Force the model to reconstruct complex movement transitions using only the static context, without entirely blinding it to the action.

### Pillar 3: Self-Supervised Bidirectional Mamba
* **Implementation:** The core sequence encoder must be a **Bidirectional Mamba** block (processing past $\rightarrow$ future and future $\rightarrow$ past). 
* **Architecture:** Use a linear-complexity SSM. The model will have an Encoder (Bi-Mamba) and a lightweight Decoder (for the SSL reconstruction task).

---

## 3. Implementation Roadmap (Step-by-Step)

**AGENT INSTRUCTION:** Do NOT write the entire codebase in one go. Follow these phases sequentially. Confirm with the user after completing each phase before moving to the next.

### Phase 1: The Data Pipeline (`dataset.py`)
1. Download/Load the UCI HAR dataset.
2. **Subject-Independent Splitting:** This is non-negotiable. Split the train/val/test sets by *user ID*, not by random shuffling, to prevent data leakage.
3. Apply standard cleaning (gravity removal, low-pass filtering).
4. Compute the **Kinematic Interaction Tensor (KIT)** and concatenate it to the feature dimension.
5. Implement Patchification: Group the time-series windows into manageable patches.

### Phase 2: The Core Architecture (`model.py`)
1. Implement a clean `BiMambaBlock` in PyTorch.
2. Implement the `PDGM` (Probabilistic Dynamics-Guided Masking) module. It must output both the masked sequence and the mask indices.
3. Assemble the `MaskedMambaAutoencoder` containing the Bi-Mamba Encoder and a simple linear/MLP Decoder.
4. Write a simple `if __name__ == "__main__":` block to pass a dummy tensor of shape `(Batch, Patches, Channels)` through the model to ensure gradients flow and shapes align.

### Phase 3: Pretraining (`pretrain.py`)
1. Write the self-supervised training loop.
2. **Loss Function:** Compute Mean Squared Error (MSE) *only* on the masked patches, not the visible ones.
3. Save the best `pretrained_encoder.pth` based on validation reconstruction loss.

### Phase 4: Fine-Tuning & Evaluation (`finetune.py` & `eval.py`)
1. Strip the Decoder and attach a Classification Head.
2. Implement **Linear Probing:** Freeze the encoder, train only the head.
3. Implement **Full Fine-Tuning:** Unfreeze the network. 
4. Implement a "Data Scarcity" training loop: Train on 1%, 5%, 10%, 25%, and 100% of the labeled data to prove label efficiency.

---

## 4. Expected Outputs & Artifacts
The code must automatically generate the following results/artifacts for the research paper:
1. `metrics.csv`: Logging Accuracy, Macro F1-Score, Precision, and Recall.
2. `confusion_matrix.png`: To visualize class-level performance.
3. `tsne_embeddings.pdf`: Extract encoder embeddings pre- and post-finetuning to visualize clustering.
4. `reconstruction_plot.png`: A visual plot showing a raw sensor wave, highlighting the sections chosen by PDGM, and overlaying the model's reconstructed wave.

---

## 5. Strict Coding Guidelines for the AI Agent
* **Clean Slate:** Do not attempt to salvage messy, legacy code from previous HAR/Mamba repositories. Write pristine, modern PyTorch.
* **Modularity:** Keep configurations (hyperparameters, sequence lengths, patch sizes) in a dedicated `config.py` or `dataclass`.
* **Typing & Comments:** Use Python type hinting extensively. Write highly descriptive docstrings for the PDGM and Bi-Mamba math so human researchers can read the code like a paper.
* **Device Agnostic:** Ensure all tensors are properly mapped to `cuda`, `mps`, or `cpu`.