# Modular-MoE-Flow-Matching


Official implementation of Modular Flow Matching. This framework leverages a Mixture of Experts (MoE) backbone to decompose complex vector fields into $K$ independent, non-overlapping "programs.

(Core Concept)Unlike standard Flow Matching, this architecture partitions the generative process. Each expert specializes in a specific region of the manifold, 
allowing for Expert Knock-out—the ability to disable specific experts during inference to generate incomplete or ablated vector fields.

🛠 Key Features

Spatial Specialization: Gumbel-Softmax gating learns a hard partition of the vector field across experts.
Modular Control: Zero out specific experts at test-time to observe localized manifold collapse.

Diversity Objective: Cosine diversity and Coefficient of Variation (CV) losses prevent expert redundancy and collapse.
Demonstrated Tasks: Successfully learns 2D Checkerboard and 3D Swiss Roll geometries.

📐 ArchitectureThe model predicts the velocity field $v$ by gating $K$ independent expert heads:
$$v(x, t) = \sum_{i=1}^{K} \alpha_i(x, t) \cdot e_i(x, t)$$

Backbone: Shared Sinusoidal Time Embeddings + LayerNorm MLP.

Gating: Decisive routing via temperature-annealed Gumbel-Softmax.




![anim_particles](https://github.com/user-attachments/assets/8d53e8b6-1a03-476f-b0e6-cc81f75bc0c2)
![anim_vfields](https://github.com/user-attachments/assets/22f3bb05-5b9e-4d62-a1ad-f5777fd05850)


On 3D Swiss Roll Data

![anim_particles_3d](https://github.com/user-attachments/assets/986a6b83-3374-41a3-9f9e-922b6efc0e10)
