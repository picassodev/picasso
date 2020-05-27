# Explicit Laser Powder Bed Fusion (LPBF) MPM Time Step #

## Particle State ##

* $x_p$ - position

* $m_p$ - mass

* $v_p$ - volume

* $e_p$ - specific internal energy

## Auxiliary particle variables ##

* $\hat{\phi}_p$ -regularized signed distance to zero isocontour (free
  surface)

* $\kappa_p$ - thermal conductivity

* $c_{vp}$ - isovolumetric specific heat

* $\mathbf{F}_p$ - heat flux

* $\rho_p$ - density

* $S_p^{laser}$ -laser specific energy

## Primary grid variables ##

* $m_i$ - node mass

* $\tilde{e}_i$ - node Favre averaged internal energy

## Auxiliary grid variables ##

* $\tilde{e}_i^*$ - node updated Favre averaged internal energy

* $(m\tilde{e})_i$ - node mass-weighted Favre averaged internal energy

* $\phi_i$ - node signed distance to zero isocontour (free surface)

* $\hat{\phi}_i$ - node regularized signed distance to zero isocontour (free
  surface)

## Material Properties ##

* $\rho$ - density

* $C_p$ - isobaric specific heat capacity

* $\kappa$ - thermal conductivity

## Shape Functions ##

* $N_{ip}^2$ - quadratic node-to-particle B-spline

* $\nabla N_{ip}^2$ - gradient of quadratic node-to-particle B-spline

\newpage

## Algorithm ##

1. Compute signed distance function, $\phi_i$, to the free surface from the
   particle positions, $x_p$

    * See signed distance algorithm on how to compute $\phi_i$ from $x_p$

    * All particles are used to compute the free surface. Only those that are
      liquid will have a non-zero surface tension coefficient

1. Compute a regularized signed distance function so the level set is now
   scaled to the range [-0.5,0.5] with a sharp gradient at the zero isocontour
   with negative values inside the volume (Grid)

   NOTE: The choice of $\epsilon$ here can vary (e.g. $\Delta x$ could also be
   used)

    * $\epsilon = \frac{\Delta x}{2}$

    * $\hat{\phi}_i = \frac{1}{2} - \frac{1}{1+\exp(\frac{\phi_i}{\epsilon})}$

1. Update grid state (P2G)

    * $m_i = \sum_p N^2_{ip} m_p$

    * $(m\tilde{e})_i = \sum_p N^2_{ip} m_p e_p$

1. Compute Favre averaged internal energy (Grid)

    * $\tilde{e}_i = \frac{(m\tilde{e})_i}{m_i}$

1. Apply thermal diffusion operator and Neumann conditions (G2P2G)

    NOTE: Here the Neumann condition is either zero on the adiabatic
    boundaries or is equal to the deposition from laser energy, loss due to
    evaporation, loss due to radiative flux, or re-absorption of radiative
    losses emitted from elsewhere in the problem.

    NOTE: See the document on the laser source for its formulation

    * $\hat{\phi}_p = \sum_i N^2_{ip} \hat{\phi}_i$

    * $$
      \begin{aligned}
        S_p^{laser} &=
            \begin{cases}
                e_v^{laser}(x_p,t), & \text{if } |\hat{\phi}_p| \leq 0.45 \\
                0, & \text{otherwise}
            \end{cases}
        \end{aligned}
      $$

    * $\rho_p = \frac{m_p}{v_p}$

    * $\kappa_p = EOS( e_p, \rho_p )$

    * $C_{vp} = EOS( e_p, \rho_p )$

    * $\mathbf{F}_p = -\frac{\kappa_p}{C_{vp}} \sum_i \mathbf{\nabla}
      N^2_{ip}$

    * $\Delta(m\tilde{e})_i = \Delta t \sum_p v_p (S_p^{laser} - \mathbf{F}_p \cdot
      \mathbf{\nabla} N^2_{ip})$

1. Compute updated grid energy and apply Dirichlet boundary conditions (Grid)

    NOTE: Neumann conditions are applied directly to the particles in the
    previous algorithmic step

    * $\tilde{e}_i^* = \tilde{e}_i + \frac{\Delta(m\tilde{e})_i}{m_i}$

    * $\tilde{e}_i^* \leftarrow BC(\tilde{e}_i^*)$

1. Update particle internal energy from the grid energy increment (G2P)

    * $e_p \leftarrow e_p + \sum_i N^2_{ip} (\tilde{e}_i^* - \tilde{e}_i)$
