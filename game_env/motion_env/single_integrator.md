# Riccati Equation for a Zero-Sum Single Integrator LQ Pursuit-Evasion Game

## System Dynamics

Let's consider a single integrator system for both the pursuer and the evader. The state \( x \) represents the relative position between the pursuer and the evader.

The dynamics of the system can be written as:
\[ \dot{x} = u_p - u_e \]
where:
- \( u_p \) is the control input of the pursuer.
- \( u_e \) is the control input of the evader.

## Cost Function

The cost function for a zero-sum game can be represented as:
\[ J = \int_0^\infty (x^T Q x + u_p^T R_p u_p - u_e^T R_e u_e) \, dt \]
where:
- \( Q \) is a positive semi-definite state weighting matrix.
- \( R_p \) is a positive definite control weighting matrix for the pursuer.
- \( R_e \) is a positive definite control weighting matrix for the evader.

## Hamiltonian

The Hamiltonian for this system can be written as:
\[ H = x^T Q x + u_p^T R_p u_p - u_e^T R_e u_e + \lambda^T (u_p - u_e) \]
where \( \lambda \) is the costate (or adjoint variable).

## Optimal Control

To find the optimal control inputs \( u_p \) and \( u_e \), we take the partial derivatives of \( H \) with respect to \( u_p \) and \( u_e \) and set them to zero:
\[ \frac{\partial H}{\partial u_p} = 2 R_p u_p + \lambda = 0 \]
\[ \frac{\partial H}{\partial u_e} = -2 R_e u_e - \lambda = 0 \]

Solving for \( u_p \) and \( u_e \), we get:
\[ u_p = -\frac{1}{2} R_p^{-1} \lambda \]
\[ u_e = \frac{1}{2} R_e^{-1} \lambda \]

## Costate Dynamics

The dynamics of the costate \( \lambda \) are given by:
\[ \dot{\lambda} = -\frac{\partial H}{\partial x} = -2 Q x \]

## Closed-Loop System Dynamics

Substitute \( u_p \) and \( u_e \) back into the system dynamics:
\[ \dot{x} = -\frac{1}{2} R_p^{-1} \lambda - \frac{1}{2} R_e^{-1} \lambda \]
\[ \dot{x} = -\frac{1}{2} (R_p^{-1} + R_e^{-1}) \lambda \]

## Differential Riccati Equation

We assume the costate \( \lambda \) is related to the state \( x \) via a time-varying matrix \( P(t) \):
\[ \lambda = 2 P(t) x \]

Differentiating \( \lambda \) with respect to time, we get:
\[ \dot{\lambda} = 2 \dot{P}(t) x + 2 P(t) \dot{x} \]

Using the costate dynamics and substituting \( \dot{\lambda} = -2 Q x \):
\[ -2 Q x = 2 \dot{P}(t) x + 2 P(t) \dot{x} \]
\[ -Q x = \dot{P}(t) x + P(t) \dot{x} \]

Substitute the closed-loop system dynamics:
\[ \dot{x} = -\frac{1}{2} (R_p^{-1} + R_e^{-1}) \lambda = - (R_p^{-1} + R_e^{-1}) P(t) x \]

Thus:
\[ -Q x = \dot{P}(t) x + P(t) \left( - (R_p^{-1} + R_e^{-1}) P(t) x \right) \]
\[ -Q x = \dot{P}(t) x - P(t) (R_p^{-1} + R_e^{-1}) P(t) x \]

Equating the coefficients of \( x \), we obtain the Riccati differential equation:
\[ \dot{P}(t) = P(t) (R_p^{-1} + R_e^{-1}) P(t) - Q \]

## Steady-State Solution

For the steady-state solution, we set \(\dot{P}(t) = 0\):
\[ 0 = P (R_p^{-1} + R_e^{-1}) P - Q \]
\[ P (R_p^{-1} + R_e^{-1}) P = Q \]
\[ P = Q^{1/2} \left( R_p^{-1} + R_e^{-1} \right)^{-1/2} \]

Thus, the steady-state Riccati equation for the zero-sum single integrator LQ pursuit-evasion game is given by:
\[ P = Q^{1/2} \left( R_p^{-1} + R_e^{-1} \right)^{-1/2} \]

This provides the optimal state feedback gains for both the pursuer and the evader.
