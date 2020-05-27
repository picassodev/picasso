# Nearly Incompressible APIC Time Step #

1. P2G

   * $p_p = -K (J_p^{-\gamma} - 1)$

   * $\mathbf{f}_i = \sum_p -v^0_p J_p p_p \mathbf{\nabla} N^2_{ip}$

   * $m_i = \sum_p N^2_{ip} m_p$

   * $\mathbf{D}_p = \sum_i N^2_{ip} (\mathbf{x}_i -
     \mathbf{x}_p)(\mathbf{x}_i-\mathbf{x}_p)^T = \frac{1}{4}\Delta x^2
     \mathbf{I}$

   * $\mathbf{mu}_i = \sum_p N^2_{ip} m_p (\mathbf{u}_p + \frac{4}{\Delta
     x^2}\mathbf{B}_p (\mathbf{x}_i - \mathbf{x}_p))$

1. Field Solve

   * $\mathbf{u}_i = (\mathbf{mu}_i + \Delta t \mathbf{f}_i) / m_i + \Delta t
     \mathbf{g}$

   * $\mathbf{u}_i \leftarrow BC(\mathbf{u}_i)$

1. G2P

    * $\mathbf{u}_p = \sum_i N^2_{ip} \mathbf{u}_i$

    * $\mathbf{B}_p = \sum_i N^2_{ip} \mathbf{u}_i (\mathbf{x}_i -
      \mathbf{x}_p)^T$

    * $(\mathbf{\nabla} \cdot \mathbf{u})_p = \sum_i \mathbf{\nabla} N^2_{ip}
      \cdot \mathbf{u}_i$

    * $J_p \leftarrow J_p e^{\Delta t (\mathbf{\nabla} \cdot \mathbf{u})_p}$

    * $\mathbf{x}_p \leftarrow \mathbf{x}_p + \Delta t \mathbf{u}_p$

    * $\rho_c = \sum_p N^{1'}_{cp} \frac{m_p}{\Delta x^3}$

    * $k_c = \sum_p N^{0'}_{cp}$

1. Position Correction

    * if cell in boundary $\rightarrow k_c = 1$

    * $\rho_c \leftarrow (k_c > 0) ? \rho_c : max(\rho_c,\rho_0)$

    * $\mathbf{\delta x}_i = -\sum_c \mathbf{\nabla} N^1_{ic} \frac{\Delta
      t^2}{\rho_0} \kappa (1 - \frac{\rho_c}{\rho_0})$

    * $\mathbf{\delta x}_i \leftarrow BC(\mathbf{\delta x}_i)$

    * $\mathbf{x}_p \leftarrow \mathbf{x}_p + \sum_i N^{2'}_{ip}
      \mathbf{\delta x}_i$
