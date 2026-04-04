# Project Memory

## User Profile
- AD/robotics C++ engineer, working on mipilot project
- Interested in stochastic global optimization for trajectory planning
- Communicates in Chinese

## Global Optimizer Project

**Spec Doc:** docs/spec.md

**Tech Stack:** C++17, Eigen, CMake, Google Test

**Algorithms (V1):** CMA-ES, xNES, MPPI, SVGD

**Architecture:**
- `Problem` interface for static optimization (with optional gradient for SVGD)
- `TrajectoryProblem` interface for MPPI (dynamics + cost)
- `Solver` / `TrajectorySolver` base classes
- All algorithms minimize by convention

**Implementation Phases:**
1. Core framework (interfaces, random utils, benchmark functions)
2. Algorithms: CMA-ES -> xNES -> SVGD -> MPPI
3. Playground: benchmark runner + CSV output + Python visualization
4. Planning demo: point robot navigation, car-like robot tracking

**References:** PDFs in 随机全局优化/ and 随机全局优化planning应用/
- NES/xNES: Exponential Natural Evolution Strategies
- CMA-ES/IGO: IGO paper, Information-Geometric Optimization Tutorial, Design Principles for Matrix Adaptation Evolution
- MPPI: TOR MPPI, MPPI from theory to parallel control
- SVGD: Stein Variational Gradient Descent
- Planning applications: Cross-entropy motion planning, VP-STO, trajectory distribution control
