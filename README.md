# Modular-MoE-Flow-Matching
Modular MoE-Flow MatchingA compact implementation of Flow Matching with a Mixture of Experts (MoE) backbone. It partitions vector fields into $K$ independent, non-overlapping regions for modular generative control.

🛠 FeaturesVector Field Partitioning: Uses Gumbel-Softmax gating to assign experts to specific spatial regions.Expert Knock-out: Supports disabling specific experts during inference to generate incomplete/ablated manifolds.Diversity Training: Employs Cosine Diversity and Coefficient of Variation (CV) losses to prevent expert collapse.📐 ArchitectureThe model predicts velocity $v$ as a weighted sum of $K$ experts:$$v = \sum_{i=1}^{K} \alpha_i e_i$$Where $\alpha$ is the gate and $e_i$ are the independent expert networks.
![anim_particles](https://github.com/user-attachments/assets/8d53e8b6-1a03-476f-b0e6-cc81f75bc0c2)
![anim_vfields](https://github.com/user-attachments/assets/22f3bb05-5b9e-4d62-a1ad-f5777fd05850)


On 3D Swiss Roll Data

![anim_particles_3d](https://github.com/user-attachments/assets/986a6b83-3374-41a3-9f9e-922b6efc0e10)
