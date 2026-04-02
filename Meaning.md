Here is a detailed breakdown of what each command does, referencing the specific components of your HAR-Mamba architecture.

To understand these commands, we first need to look at your architecture, which has three main parts:

The Input Pipeline (KIT + PDGM): Creates the 12-channel input and dynamically masks parts of the signal.
The Bi-Mamba Encoder: The core "brain" that learns to understand human motion from the sequences.
The "Head" (attached to the end of the Encoder):
During Pretraining: It’s an MLP Decoder tasked with reconstructing the missing (masked) signal.
During Fine-tuning: It’s a Classification Head tasked with guessing the activity (Walking, Sitting, etc.).
Phase 1: Pretraining
bash
python main.py --phase pretrain
What it does: This runs the Self-Supervised Learning (SSL) phase. The model is given sensor data without any labels (it doesn't know if the person is walking or sitting).
Active Architectural Components: Bi-Mamba Encoder + MLP Decoder
What is happening: The PDGM algorithm hides the most "dynamic" parts of the signal. The Bi-Mamba Encoder looks at the remaining static context, and the MLP Decoder tries to redraw the missing signal. The model calculates its loss based on how accurately it re-drew the wave.
Output: It saves pretrained_encoder.pth. You now have an Encoder that deeply understands human kinematics, but it still doesn't know the names of the activities.
Phase 2: Full Fine-Tuning
bash
python main.py --phase finetune
# OR: python finetune.py --mode full --label-fraction 1.0
What it does: This teaches the pre-trained model to actually classify the 6 activities using 100% of the labeled data.
Active Architectural Components: Bi-Mamba Encoder (Loaded from Phase 1) + Classification Head. (The MLP Decoder is thrown away).
What is happening: We attach a new Classification Head to the end of your Encoder. We then feed it data with the labels. Crucially, in "Full" mode, the entire network is updated. The gradients flow all the way back through the Classification Head and update the weights inside the Bi-Mamba Encoder.
Why use it: This usually yields the highest possible accuracy.
Phase 3: Linear Probing
bash
python main.py --phase linear
# OR: python finetune.py --mode linear_probe
What it does: It acts as a strict test of how good your Pretraining phase (Phase 1) actually was.
Active Architectural Components: Bi-Mamba Encoder (Frozen) + Classification Head (Trainable).
What is happening: We "unplug" the gradients from the Bi-Mamba Encoder. Its weights are locked in place. We only train the final Classification Head.
Why use it: If the accuracy is high during Linear Probing, it scientifically proves that your Self-Supervised Pretraining (KIT + PDGM) was highly successful. It means the Encoder learned such good fundamental representations of human movement that a simple linear classifier on top can easily separate walking from sitting.
Phase 4: Data Scarcity Experiment
bash
python main.py --phase scarcity
# OR: python finetune.py --mode scarcity
What it does: It proves the "Label Efficiency" of your framework.
Active Architectural Components: Bi-Mamba Encoder + Classification Head
What is happening: It runs multiple full fine-tuning loops sequentially, but it artificially restricts the dataset. It trains first on only 1% of the labeled data, then 5%, 10%, 25%, and 100%.
Why use it: This is the core selling point of Self-Supervised Learning (SSL) for a research paper. You are trying to prove that because your model pre-trained on the physical structure of the data (using PDGM), it only needs a tiny handful of labeled examples (e.g., 5%) to achieve the same accuracy as a standard model trained on 100% of the data.
Phase 5: Evaluation
bash
python main.py --phase eval
What it does: Runs the final trained model strictly against the unseen test dataset split.
What is happening: No training occurs. It just pushes the test data through the fully fine-tuned model and records the predictions.
Output: It generates the final benchmarking numbers (Accuracy, F1-Score) and produces the confusion_matrix.png and tsne.png plots for your paper.
