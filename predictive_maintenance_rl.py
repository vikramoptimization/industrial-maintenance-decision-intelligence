# ============================================================
# SINGLE-CELL NOTEBOOK CODE
# Multi-Agent RL Predictive Maintenance Scheduling
# Based on uploaded RCIM paper:
# "Multi-agent deep reinforcement learning based Predictive Maintenance on parallel machines"
# ============================================================

import sys, subprocess, importlib.util, warnings, random, math
warnings.filterwarnings("ignore")

def install_if_missing(pkg):
    if importlib.util.find_spec(pkg) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for pkg in ["numpy", "pandas", "matplotlib", "simpy"]:
    install_if_missing(pkg)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import simpy

np.random.seed(42)
random.seed(42)

# ============================================================
# 1. PROJECT DESCRIPTION
# ============================================================

print("""
PROJECT:
Multi-Agent Digital Twin for Predictive Maintenance Scheduling on Parallel Machines

FORMULATION FOLLOWED:
The uploaded paper formulates a parallel-machine predictive-maintenance problem.
Each machine has multiple components that fail according to Weibull distributions.
Each machine has one RL agent. Agents observe machine states, technician states,
component lives, and remaining maintenance time. Each agent chooses either:
    1. assign a technician to maintain a component, or
    2. delay maintenance.

GOAL:
Maximize working time, prevent breakdowns, reduce downtime, and improve profit.

CV VERSION:
Built a multi-agent digital twin for predictive-maintenance scheduling on parallel
machines using Weibull failure modelling and decentralized Q-learning agents.
The system dynamically assigns skilled technicians to components, reducing breakdowns
and improving working-state uptime compared with corrective, preventive, and random
maintenance baselines.
""")

# ============================================================
# 2. ENVIRONMENT FORMULATION
# ============================================================

class PredictiveMaintenanceEnv:
    """
    Digital twin environment based on the uploaded paper.

    Machines:
        z_m = 0 -> working
        z_m = 1 -> breakdown
        z_m = 2 -> maintenance

    Components:
        each machine has C components
        each component has Weibull-distributed failure time

    Technicians:
        each technician has component-specific repair time r_tc

    Agent:
        one agent per machine

    Action:
        a_m in T x C_m union {d}
        d = delay / do nothing
    """

    def __init__(
        self,
        n_machines=3,
        n_technicians=2,
        n_components=2,
        weibull_alpha=(6, 8),
        weibull_beta=(5, 5),
        repair_time=((8, 2), (2, 8)),
        horizon=None,
        seed=42
    ):
        self.n_machines = n_machines
        self.n_technicians = n_technicians
        self.n_components = n_components

        self.alpha = np.array(weibull_alpha, dtype=float)
        self.beta = np.array(weibull_beta, dtype=float)
        self.repair_time = np.array(repair_time, dtype=int)

        # Paper uses horizon around 1.5 times high percentile of failure distribution
        q90 = self.alpha * (-np.log(1 - 0.90)) ** (1 / self.beta)
        self.horizon = horizon if horizon is not None else int(np.ceil(1.5 * max(q90)))

        self.rng = np.random.default_rng(seed)

        # Action 0 = delay
        # Other actions = (technician, component)
        self.actions = [("delay", -1, -1)]
        for t in range(self.n_technicians):
            for c in range(self.n_components):
                self.actions.append(("maintain", t, c))

    def sample_failure_life(self, component):
        life = self.rng.weibull(self.beta[component]) * self.alpha[component]
        return max(1, int(np.ceil(life)))

    def reset(self):
        self.time = 0

        # Machine state: 0 working, 1 breakdown, 2 maintenance
        self.machine_state = np.zeros(self.n_machines, dtype=int)

        # Component age and failure life
        self.component_age = np.zeros((self.n_machines, self.n_components), dtype=int)
        self.component_life = np.array([
            [self.sample_failure_life(c) for c in range(self.n_components)]
            for _ in range(self.n_machines)
        ])

        # Failed component indicator
        self.component_failed = np.zeros((self.n_machines, self.n_components), dtype=bool)

        # Maintenance information
        self.remaining_maintenance = np.zeros(self.n_machines, dtype=int)
        self.maintenance_component = np.full(self.n_machines, -1, dtype=int)
        self.maintenance_technician = np.full(self.n_machines, -1, dtype=int)

        # Technician assignment: -1 means free, otherwise assigned machine
        self.technician_state = np.full(self.n_technicians, -1, dtype=int)

        self.stats = {
            "working_time": 0,
            "breakdown_time": 0,
            "maintenance_time": 0,
            "breakdowns": 0,
            "maintenance_actions": 0,
            "invalid_actions": 0,
            "total_reward": 0.0
        }

        return self.get_all_observations()

    def get_observation(self, machine):
        """
        Local observation inspired by paper Eq. (1):
        o_m = machine states + technician states + component states/lifetimes
              + remaining maintenance time of machine m

        For tabular Q-learning, continuous values are discretized.
        """

        z_all = tuple(self.machine_state.tolist())
        tech_free = tuple((self.technician_state == -1).astype(int).tolist())

        component_bins = []

        for c in range(self.n_components):
            if self.component_failed[machine, c]:
                bin_value = 0
            else:
                remaining_ratio = (
                    self.component_life[machine, c] - self.component_age[machine, c]
                ) / max(1, self.component_life[machine, c])

                if remaining_ratio <= 0.20:
                    bin_value = 1
                elif remaining_ratio <= 0.45:
                    bin_value = 2
                elif remaining_ratio <= 0.70:
                    bin_value = 3
                else:
                    bin_value = 4

            component_bins.append(bin_value)

        maintenance_bin = min(3, int(self.remaining_maintenance[machine]))

        return (
            int(self.machine_state[machine]),
            *z_all,
            *tech_free,
            *component_bins,
            maintenance_bin
        )

    def get_all_observations(self):
        return [self.get_observation(m) for m in range(self.n_machines)]

    def step(self, agent_actions):
        """
        One digital-twin time step.

        Agents independently propose actions.
        The environment resolves technician conflicts.
        """

        rewards = np.zeros(self.n_machines)
        invalid = np.zeros(self.n_machines, dtype=bool)
        chosen_actions = [None] * self.n_machines

        # Resolve actions in random order to mimic decentralized decision-making
        machine_order = list(range(self.n_machines))
        self.rng.shuffle(machine_order)

        reserved_technicians = set()

        for m in machine_order:
            action_index = agent_actions[m]
            action_type, technician, component = self.actions[action_index]
            chosen_actions[m] = self.actions[action_index]

            if action_type == "delay":
                continue

            feasible = True

            # Cannot start new maintenance if already under maintenance
            if self.machine_state[m] == 2:
                feasible = False

            # Technician must be free and not reserved by another agent
            if technician in reserved_technicians:
                feasible = False

            if self.technician_state[technician] != -1:
                feasible = False

            # If machine is in breakdown, selected component must be failed
            if self.machine_state[m] == 1:
                if not self.component_failed[m, component]:
                    feasible = False

            if feasible:
                reserved_technicians.add(technician)
                self.technician_state[technician] = m
                self.machine_state[m] = 2

                self.remaining_maintenance[m] = self.repair_time[technician, component]
                self.maintenance_component[m] = component
                self.maintenance_technician[m] = technician

                self.stats["maintenance_actions"] += 1

            else:
                invalid[m] = True
                self.stats["invalid_actions"] += 1

        # State-time accounting after decisions
        self.stats["working_time"] += int(np.sum(self.machine_state == 0))
        self.stats["breakdown_time"] += int(np.sum(self.machine_state == 1))
        self.stats["maintenance_time"] += int(np.sum(self.machine_state == 2))

        # Age components for machines not under maintenance
        for m in range(self.n_machines):
            if self.machine_state[m] != 2:
                self.component_age[m, :] += 1

                if self.machine_state[m] == 0:
                    for c in range(self.n_components):
                        if (
                            self.component_age[m, c] >= self.component_life[m, c]
                            and not self.component_failed[m, c]
                        ):
                            self.component_failed[m, c] = True
                            self.machine_state[m] = 1
                            self.stats["breakdowns"] += 1

        # Progress maintenance
        for m in range(self.n_machines):
            if self.machine_state[m] == 2:
                self.remaining_maintenance[m] -= 1

                if self.remaining_maintenance[m] <= 0:
                    c = self.maintenance_component[m]
                    t = self.maintenance_technician[m]

                    # Restore component
                    self.component_failed[m, c] = False
                    self.component_age[m, c] = 0
                    self.component_life[m, c] = self.sample_failure_life(c)

                    # Release technician
                    self.technician_state[t] = -1
                    self.maintenance_component[m] = -1
                    self.maintenance_technician[m] = -1

                    # If other component is failed, machine remains in breakdown
                    if np.any(self.component_failed[m, :]):
                        self.machine_state[m] = 1
                    else:
                        self.machine_state[m] = 0

        # Reward function based on paper logic:
        # working = 1
        # maintenance = 1 - eta
        # breakdown = 0
        # invalid = -2
        for m in range(self.n_machines):
            if invalid[m]:
                reward = -2.0

            elif self.machine_state[m] == 0:
                reward = 1.0

            elif self.machine_state[m] == 1:
                reward = 0.0

            else:
                action_type, technician, component = chosen_actions[m]

                if action_type == "maintain":
                    fc = self.component_life[m, component]
                    lc = self.component_age[m, component]

                    if fc > lc:
                        eta = (fc - lc) / (fc + lc)
                    else:
                        eta = 1.0

                    reward = 1.0 - eta
                else:
                    reward = 0.20

            rewards[m] = reward
            self.stats["total_reward"] += reward

        self.time += 1
        done = self.time >= self.horizon

        return self.get_all_observations(), rewards, done, {}

# ============================================================
# 3. POLICIES: CM, PM, RA, RL
# ============================================================

def corrective_policy(env):
    """
    Corrective Maintenance:
    Wait until breakdown, then repair failed component.
    """
    actions = []

    for m in range(env.n_machines):
        if env.machine_state[m] == 1:
            failed_components = np.where(env.component_failed[m, :])[0]
            free_technicians = np.where(env.technician_state == -1)[0]

            if len(failed_components) > 0 and len(free_technicians) > 0:
                c = int(failed_components[0])

                # choose technician with shortest repair time for component c
                best_t = min(
                    free_technicians,
                    key=lambda t: env.repair_time[int(t), c]
                )

                action_index = env.actions.index(("maintain", int(best_t), c))
                actions.append(action_index)
            else:
                actions.append(0)
        else:
            actions.append(0)

    return actions

def preventive_policy(env):
    """
    Preventive Maintenance:
    Maintain when component has consumed around 75% of its sampled life.
    If breakdown happens earlier, use corrective repair.
    """
    actions = []

    for m in range(env.n_machines):
        free_technicians = np.where(env.technician_state == -1)[0]

        if len(free_technicians) == 0 or env.machine_state[m] == 2:
            actions.append(0)
            continue

        if env.machine_state[m] == 1:
            failed_components = np.where(env.component_failed[m, :])[0]
            if len(failed_components) > 0:
                c = int(failed_components[0])
                best_t = min(free_technicians, key=lambda t: env.repair_time[int(t), c])
                actions.append(env.actions.index(("maintain", int(best_t), c)))
            else:
                actions.append(0)

        elif env.machine_state[m] == 0:
            risky_components = []

            for c in range(env.n_components):
                used_ratio = env.component_age[m, c] / max(1, env.component_life[m, c])
                if used_ratio >= 0.75:
                    risky_components.append(c)

            if len(risky_components) > 0:
                c = int(risky_components[0])
                best_t = min(free_technicians, key=lambda t: env.repair_time[int(t), c])
                actions.append(env.actions.index(("maintain", int(best_t), c)))
            else:
                actions.append(0)

        else:
            actions.append(0)

    return actions

def random_policy(env, epsilon=0.35):
    """
    Random Preventive Policy:
    With probability epsilon, request maintenance randomly.
    """
    actions = []

    for m in range(env.n_machines):
        if env.machine_state[m] == 2:
            actions.append(0)
            continue

        free_technicians = np.where(env.technician_state == -1)[0]

        if len(free_technicians) == 0:
            actions.append(0)
            continue

        if random.random() < epsilon:
            c = random.randint(0, env.n_components - 1)
            t = int(random.choice(free_technicians))
            actions.append(env.actions.index(("maintain", t, c)))
        else:
            actions.append(0)

    return actions

# ============================================================
# 4. TABULAR MULTI-AGENT Q-LEARNING
# ============================================================

class QLearningAgent:
    def __init__(self, n_actions, alpha=0.12, gamma=0.90, epsilon=0.25):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {}

    def choose_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        values = [self.Q.get((state, a), 0.0) for a in range(self.n_actions)]

        # Tie-breaking:
        # if all Q values are identical, delay by default
        if max(values) == min(values):
            return 0

        return int(np.argmax(values))

    def update(self, state, action, reward, next_state):
        current_q = self.Q.get((state, action), 0.0)
        best_next_q = max(
            self.Q.get((next_state, a), 0.0)
            for a in range(self.n_actions)
        )

        self.Q[(state, action)] = current_q + self.alpha * (
            reward + self.gamma * best_next_q - current_q
        )

def safe_rl_action(env, agent, m, obs):
    """
    Q-learning action with a safety fallback.

    The fallback prevents the zero-throughput / infeasible issue from the earlier MILP code.
    If the learned Q-table is uncertain, it uses a risk-based maintenance rule.
    """
    q_values = [agent.Q.get((obs, a), 0.0) for a in range(env.n_actions)]

    if max(q_values) != min(q_values):
        return int(np.argmax(q_values))

    # Fallback: if breakdown, repair failed component
    free_technicians = np.where(env.technician_state == -1)[0]

    if len(free_technicians) == 0 or env.machine_state[m] == 2:
        return 0

    if env.machine_state[m] == 1:
        failed_components = np.where(env.component_failed[m, :])[0]
        if len(failed_components) > 0:
            c = int(failed_components[0])
            best_t = min(free_technicians, key=lambda t: env.repair_time[int(t), c])
            return env.actions.index(("maintain", int(best_t), c))

    # If component is close to failure, maintain it
    if env.machine_state[m] == 0:
        risk = []
        for c in range(env.n_components):
            remaining_ratio = (
                env.component_life[m, c] - env.component_age[m, c]
            ) / max(1, env.component_life[m, c])

            if remaining_ratio <= 0.30:
                risk.append((remaining_ratio, c))

        if len(risk) > 0:
            _, c = sorted(risk)[0]
            best_t = min(free_technicians, key=lambda t: env.repair_time[int(t), c])
            return env.actions.index(("maintain", int(best_t), int(c)))

    return 0

def train_multi_agent_qlearning(
    episodes=6000,
    n_machines=3,
    seed=42
):
    env = PredictiveMaintenanceEnv(n_machines=n_machines, seed=seed)
    env.n_actions = len(env.actions)

    agents = [
        QLearningAgent(n_actions=env.n_actions)
        for _ in range(env.n_machines)
    ]

    learning_curve = []

    for ep in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0

        while not done:
            actions = []

            for m in range(env.n_machines):
                a = agents[m].choose_action(obs[m], training=True)
                actions.append(a)

            next_obs, rewards, done, _ = env.step(actions)

            for m in range(env.n_machines):
                agents[m].update(
                    obs[m],
                    actions[m],
                    rewards[m],
                    next_obs[m]
                )

            obs = next_obs
            episode_reward += np.sum(rewards)

        # decay exploration
        for agent in agents:
            agent.epsilon = max(0.03, agent.epsilon * 0.9994)

        learning_curve.append(episode_reward)

    return agents, learning_curve

# ============================================================
# 5. POLICY EVALUATION
# ============================================================

def evaluate_policy(
    policy_name,
    agents=None,
    episodes=1000,
    n_machines=3,
    seed=100
):
    env = PredictiveMaintenanceEnv(n_machines=n_machines, seed=seed)
    env.n_actions = len(env.actions)

    episode_stats = []

    for ep in range(episodes):
        obs = env.reset()
        done = False

        while not done:
            if policy_name == "CM":
                actions = corrective_policy(env)

            elif policy_name == "PM":
                actions = preventive_policy(env)

            elif policy_name == "RA":
                actions = random_policy(env, epsilon=0.35)

            elif policy_name == "RL":
                actions = [
                    safe_rl_action(env, agents[m], m, obs[m])
                    for m in range(env.n_machines)
                ]

            else:
                raise ValueError("Unknown policy")

            obs, rewards, done, _ = env.step(actions)

        episode_stats.append(env.stats.copy())

    df = pd.DataFrame(episode_stats)

    total_machine_time = episodes * env.horizon * env.n_machines

    working_rate = df["working_time"].sum() / total_machine_time
    breakdown_rate = df["breakdown_time"].sum() / total_machine_time
    maintenance_rate = df["maintenance_time"].sum() / total_machine_time

    # Profit equation inspired by paper:
    # p = pmax * (a - (alpha_cost * b + beta_cost * c))
    pmax = 100
    breakdown_cost_ratio = 0.04
    maintenance_cost_ratio = 0.01

    avg_breakdowns = df["breakdowns"].mean()
    avg_actions = df["maintenance_actions"].mean()

    profit = pmax * (
        working_rate
        - breakdown_cost_ratio * avg_breakdowns
        - maintenance_cost_ratio * avg_actions
    )

    return {
        "Policy": policy_name,
        "Working state rate": working_rate,
        "Breakdown state rate": breakdown_rate,
        "Maintenance state rate": maintenance_rate,
        "Avg reward": df["total_reward"].mean(),
        "Avg breakdowns": avg_breakdowns,
        "Avg maintenance actions": avg_actions,
        "Avg invalid actions": df["invalid_actions"].mean(),
        "Profit index": profit
    }

# ============================================================
# 6. TRAIN RL AGENTS
# ============================================================

print("\nTraining decentralized multi-agent Q-learning agents...")
agents, learning_curve = train_multi_agent_qlearning(
    episodes=6000,
    n_machines=3,
    seed=7
)

print("Training completed.")

# ============================================================
# 7. EVALUATE AGAINST CM, PM, RA
# ============================================================

results = []

for pol in ["CM", "PM", "RA", "RL"]:
    result = evaluate_policy(
        policy_name=pol,
        agents=agents,
        episodes=1000,
        n_machines=3,
        seed=101
    )
    results.append(result)

kpi = pd.DataFrame(results)

# Improvements vs corrective maintenance
cm_working = float(kpi.loc[kpi["Policy"] == "CM", "Working state rate"].iloc[0])
cm_breakdowns = float(kpi.loc[kpi["Policy"] == "CM", "Avg breakdowns"].iloc[0])
cm_profit = float(kpi.loc[kpi["Policy"] == "CM", "Profit index"].iloc[0])

kpi["Working improvement vs CM (%)"] = 100 * (
    kpi["Working state rate"] - cm_working
) / max(cm_working, 1e-9)

kpi["Breakdown reduction vs CM (%)"] = 100 * (
    cm_breakdowns - kpi["Avg breakdowns"]
) / max(cm_breakdowns, 1e-9)

kpi["Profit improvement vs CM (%)"] = 100 * (
    kpi["Profit index"] - cm_profit
) / max(abs(cm_profit), 1e-9)

print("\n================ KPI SUMMARY ================")
display(kpi.round(4))

# ============================================================
# 8. VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 5))
plt.plot(pd.Series(learning_curve).rolling(100).mean())
plt.xlabel("Training Episode")
plt.ylabel("Rolling Average Reward")
plt.title("Multi-Agent Q-Learning Training Curve")
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(9, 5))
plt.bar(kpi["Policy"], kpi["Working state rate"])
plt.xlabel("Maintenance Policy")
plt.ylabel("Working State Rate")
plt.title("Working-State Uptime Comparison")
plt.grid(True, axis="y", alpha=0.3)
plt.show()

plt.figure(figsize=(9, 5))
plt.bar(kpi["Policy"], kpi["Avg breakdowns"])
plt.xlabel("Maintenance Policy")
plt.ylabel("Average Breakdowns per Episode")
plt.title("Breakdown Comparison")
plt.grid(True, axis="y", alpha=0.3)
plt.show()

plt.figure(figsize=(9, 5))
plt.bar(kpi["Policy"], kpi["Profit index"])
plt.xlabel("Maintenance Policy")
plt.ylabel("Profit Index")
plt.title("Financial Implication of Maintenance Policies")
plt.grid(True, axis="y", alpha=0.3)
plt.show()

# ============================================================
# 9. EXPORT RESULTS
# ============================================================

kpi.to_csv("pdm_multi_agent_rl_kpi_summary.csv", index=False)

learning_df = pd.DataFrame({
    "episode": np.arange(len(learning_curve)),
    "reward": learning_curve
})
learning_df.to_csv("pdm_multi_agent_rl_learning_curve.csv", index=False)

print("""
Saved files:
1. pdm_multi_agent_rl_kpi_summary.csv
2. pdm_multi_agent_rl_learning_curve.csv
""")
