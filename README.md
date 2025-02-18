# AI-Driven Simulation of Contact and Plasticity Phenomena with MeshGraphNets

## Overview

This repository contains the implementation and dataset for AI-driven simulations of contact and plasticity phenomena in solid mechanics. The study focuses on a worst-case scenario involving highly nonlinear behaviors such as plastic deformation and contact mechanics.

## Problem Definition

We simulate the impact of a cylindrical actuator on a rectangular plate, where plasticity and contact interactions are the primary nonlinearities. The main characteristics of the setup are:

- **Plate dimensions**: Length = 0.5, Width = 0.25.
- **Material properties**:
  - Young’s modulus: 210
  - Poisson’s coefficient: 0.3
  - Yield stress: 300 (with hardening behavior)
- **Actuator**: Cylindrical rigid body impacting the plate at varying positions.
- **Mesh information**:
  - Plate: ~600 nodes (varies slightly based on hole position and size).
  - Actuator: 450 nodes.
- **Dataset**:
  - 135 high-fidelity finite element simulations.
  - Each simulation consists of 435 pseudo-time steps under quasi-static conditions.
  - Variability in boundary conditions and geometry (holes of different sizes and positions, variable thickness).

## Approach

To model these highly nonlinear behaviors, we extend **MeshGraphNets (MGN)** to account for plasticity and energy dissipation due to permanent deformations. Previous works using MGN have been mostly limited to hyperelastic materials, making this an important step in capturing more complex material responses.

### Dataset Split

- **Training Set**: 80 trajectories
- **Validation Set**: 10 trajectories
- **Test Set**: 20 trajectories
- **Out-of-distribution Test Set (Dextra)**: 25 trajectories with unseen geometries

## Repository Structure

