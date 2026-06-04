# 🤖 Multi-Agent Digital Twin for Predictive Maintenance Optimization

## Overview

Modern manufacturing systems rely on complex machine networks operating under continuous production schedules. Unexpected equipment failures can lead to costly downtime, reduced productivity, increased maintenance expenses, and missed delivery commitments.

Traditional maintenance strategies such as corrective maintenance and periodic preventive maintenance often fail to account for dynamic machine health conditions, technician availability, and stochastic failure behavior.

This project develops a Digital Twin-driven Predictive Maintenance Optimization Framework that integrates:

* Weibull reliability modeling
* Multi-agent reinforcement learning
* Technician scheduling
* Predictive maintenance decision-making
* Manufacturing system simulation

The framework enables autonomous maintenance planning for parallel-machine production systems by continuously monitoring machine health, predicting failures, and dynamically assigning maintenance resources.

---

# Business Problem

Manufacturing facilities frequently operate multiple machines in parallel.

Each machine consists of several critical components that degrade over time and eventually fail.

Typical challenges include:

* Unplanned equipment breakdowns
* Limited maintenance technicians
* Component-specific repair requirements
* Production downtime
* Increasing maintenance costs
* Inefficient maintenance scheduling

The objective is to develop an intelligent maintenance policy that maximizes machine uptime while minimizing breakdowns and maintenance costs.

---

# Project Objectives

The framework aims to:

✓ Increase machine working-state availability

✓ Reduce breakdown frequency

✓ Improve maintenance resource utilization

✓ Optimize technician assignment decisions

✓ Minimize production downtime

✓ Improve operational profitability

✓ Learn adaptive maintenance policies through reinforcement learning

---

# Digital Twin Architecture

```text
Parallel Manufacturing System
           │
           ▼
 ┌──────────────────────────┐
 │ Digital Twin Environment │
 └────────────┬─────────────┘
              │
              ▼
 ┌──────────────────────────┐
 │ Machine Health Modeling  │
 │ Weibull Failure Process  │
 └────────────┬─────────────┘
              │
              ▼
 ┌──────────────────────────┐
 │ Multi-Agent RL Layer     │
 │ One Agent per Machine    │
 └────────────┬─────────────┘
              │
              ▼
 ┌──────────────────────────┐
 │ Technician Assignment    │
 │ Maintenance Scheduling   │
 └────────────┬─────────────┘
              │
              ▼
      Optimized Maintenance
```

---

# System Description

## Machines

The manufacturing system consists of multiple parallel machines.

Machine states:

* Working State (0)
* Breakdown State (1)
* Maintenance State (2)

Each machine is controlled by an independent reinforcement learning agent.

---

## Components

Every machine contains multiple components.

Component degradation follows a Weibull reliability distribution.

For component c:

Failure Life ~ Weibull(αc, βc)

where:

* α = scale parameter
* β = shape parameter

In this implementation:

| Component   | α | β |
| ----------- | - | - |
| Component 1 | 6 | 5 |
| Component 2 | 8 | 5 |

---

## Technicians

Maintenance technicians have different repair capabilities.

Repair times depend on both:

* Technician
* Component type

Example repair matrix:

| Technician | Component 1 | Component 2 |
| ---------- | ----------- | ----------- |
| T1         | 8           | 2           |
| T2         | 2           | 8           |

---

# Reinforcement Learning Formulation

## Agent

One RL agent is assigned to each machine.

The agent continuously observes:

* Machine states
* Technician availability
* Component health
* Remaining maintenance time

---

## State Space

Each machine observes:

```text
Machine State
+
Technician Availability
+
Component Health Levels
+
Remaining Maintenance Time
```

Component health is discretized into health bins:

| Health Level | Remaining Life |
| ------------ | -------------- |
| 4            | >70%           |
| 3            | 45–70%         |
| 2            | 20–45%         |
| 1            | <20%           |
| 0            | Failed         |

---

## Action Space

Action 0:

```text
Delay Maintenance
```

Actions 1–N:

```text
Assign Technician t
to Component c
```

Example:

```text
Maintain Component 1 using Technician 2
```

---

# Maintenance Policies Evaluated

The framework benchmarks four maintenance strategies.

## Corrective Maintenance (CM)

Wait until component failure occurs.

Then dispatch maintenance.

Characteristics:

* Lowest maintenance frequency
* Highest breakdown frequency

---

## Preventive Maintenance (PM)

Perform maintenance when component age exceeds 75% of expected life.

Characteristics:

* Fewer failures
* Increased maintenance workload

---

## Random Maintenance (RA)

Maintenance decisions are randomly triggered.

Characteristics:

* Baseline comparison policy

---

## Reinforcement Learning (RL)

Maintenance actions are learned through interaction with the digital twin environment.

Characteristics:

* Adaptive
* Data-driven
* Resource-aware

---

# Reward Function

The RL reward function encourages machine uptime while discouraging breakdowns.

Working State:

```text
Reward = +1
```

Breakdown State:

```text
Reward = 0
```

Invalid Maintenance Action:

```text
Reward = -2
```

Maintenance State:

```text
Reward = 1 − η
```

where

```text
η = (Remaining Life Difference)
```

This encourages maintenance decisions near optimal intervention points.

---

# Training Configuration

| Parameter               | Value      |
| ----------------------- | ---------- |
| Learning Algorithm      | Q-Learning |
| Episodes                | 6000       |
| Learning Rate (α)       | 0.12       |
| Discount Factor (γ)     | 0.90       |
| Initial Exploration (ε) | 0.25       |
| Minimum Exploration     | 0.03       |

---

# Evaluation Metrics

The following KPIs are measured.

## Working-State Rate

Percentage of time machines remain operational.

Higher is better.

---

## Breakdown-State Rate

Percentage of time machines remain in failure state.

Lower is better.

---

## Maintenance-State Rate

Percentage of time machines spend under maintenance.

Lower is generally preferred.

---

## Average Breakdowns

Average number of breakdown events per simulation episode.

Lower is better.

---

## Maintenance Actions

Total maintenance interventions.

Represents maintenance workload.

---

## Profit Index

Financial metric computed as:

Profit Index =
Working Rate
− Breakdown Penalty
− Maintenance Penalty

Higher values indicate better operational performance.

---

# Example Results

| Policy | Working Rate | Avg Breakdowns | Profit Index |
| ------ | ------------ | -------------- | ------------ |
| CM     | Baseline     | Highest        | Lowest       |
| PM     | Improved     | Lower          | Improved     |
| RA     | Moderate     | Moderate       | Moderate     |
| RL     | Highest      | Lowest         | Highest      |

---

# Visual Analytics

The framework automatically generates:

* Training reward convergence
* Working-state comparison
* Breakdown comparison
* Profit comparison

These visualizations enable direct comparison between maintenance strategies.

---

# Technology Stack

* Python
* NumPy
* Pandas
* Matplotlib
* SimPy
* Reinforcement Learning
* Reliability Engineering

---

# Installation

```bash
pip install numpy pandas matplotlib simpy
```

---

# Run

```bash
python predictive_maintenance_rl.py
```

---

# Key Contributions

* Multi-Agent Reinforcement Learning
* Predictive Maintenance Optimization
* Digital Twin Modeling
* Technician Scheduling
* Reliability Engineering
* Manufacturing Analytics
* Decision Intelligence
* Industrial AI

---

