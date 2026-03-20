# Modular-MoE-Flow-Matching


Official implementation of Modular Flow Matching. This framework leverages a Mixture of Experts (MoE) backbone to decompose complex vector fields into $K$ independent, non-overlapping "programs.

Core concept: Unlike standard Flow Matching, the architecture partitions the generative process. Each expert specializes in a specific region of the manifold, 
allowing for Expert Knock-out—the ability to disable specific experts during inference to generate incomplete or ablated vector fields.

🚀 Key Features

Regional Specialization: Using a Gumbel-Softmax gate, the model learns to partition the 2D/3D space so that experts do not overlap in their responsibilities.

Expert Ablation (Knock-out): Since experts are modular, you can manually zero out specific experts during inference to see how the vector field collapses or which parts of the distribution are lost.

Diverse Vector Fields: Implements a Cosine Diversity Loss to ensure experts learn unique directional strategies rather than redundant ones.

Dynamic Gating: Features a temperature-scheduled Gumbel-Softmax to transition from soft exploration to hard, decisive routing during training.


🧠 Architecture Overview

The core model  consists of a shared backbone that processes position and time embeddings, followed by a gating network and a set of independent expert heads.


The model learns a vector field $v_\theta(x, t)$ that pushes a base distribution $p_0$ (Gaussian noise) 
toward a target distribution $p_1$ (Data) following the probability flow:

$$\dot{x}_t = v(x_t, t)$$

MoE IntegrationThe final velocity is a weighted sum of $K$ expert predictions:

$$v_{final} = \sum_{i=1}^{K} \alpha_i(x, t) \cdot e_i(x, t)$$

Where $\alpha$ is the routing probability provided by the gate.


🔧 Loss Functions

The training objective is a multi-objective loss designed for modularity:

Flow Matching Loss ($L_{fm}$): Standard MSE between predicted and target velocity.

Coefficient of Variation ($L_{cv}$): Ensures load balancing so one expert doesn't collapse and take over the entire field.

Cosine Diversity ($L_{div}$): Penalizes experts for pointing in the same direction:

$$L_{div} = \frac{1}{n} \sum_{i < j} \frac{v_i \cdot v_j}{\|v_i\| \|v_j\|}$$

Magnitude Regularization ($L_{mag}$): Prevents exploding gradients in dormant experts.



📊 Results on 2D Checkerboard

![anim_particles](https://github.com/user-attachments/assets/8d53e8b6-1a03-476f-b0e6-cc81f75bc0c2)
![anim_vfields](https://github.com/user-attachments/assets/22f3bb05-5b9e-4d62-a1ad-f5777fd05850)


📊 Results On 3D Swiss Roll Data

![anim_particles_3d](https://github.com/user-attachments/assets/986a6b83-3374-41a3-9f9e-922b6efc0e10)
