import networkx as nx
import numpy as np

from .inputs import validate_damage


def node_performance(distances, population):
    n = len(population)
    population = np.asarray(population, dtype=np.float64)
    total_population = population.sum()
    values = np.zeros(n, dtype=np.float64)
    for node, paths in distances.items():
        total = 0.0
        for target, distance in paths.items():
            if distance == 0:
                continue
            total += (population[node] + population[target]) / distance / (2.0 * total_population)
        values[node] = total / (n - 1)
    return values


def shortest_travel_times(network, damage):
    speed = network.speed_mps * (1.0 - damage)
    graph = nx.Graph()
    graph.add_nodes_from(range(network.population.size))
    for road, (u, v) in enumerate(network.endpoints.T):
        if speed[road] > 0:
            graph.add_edge(int(u), int(v), time=float(network.length_m[road] / speed[road]))
    return dict(nx.all_pairs_dijkstra_path_length(graph, weight="time"))


class RecoveryEnvironment:
    def __init__(self, network, repair_length_unit_m=500.0):
        if not 0 < repair_length_unit_m < float("inf"):
            raise ValueError("repair_length_unit_m must be finite and positive")
        self.network = network
        self.repair_length_unit_m = repair_length_unit_m
        intact = shortest_travel_times(network, np.zeros(network.length_m.size))
        self.normal_x = node_performance(intact, network.population)
        self.reset(np.zeros(network.length_m.size))

    def reset(self, damage):
        self.damage = validate_damage(damage, self.network.length_m.size)
        self.distances = shortest_travel_times(self.network, self.damage)
        self.x = node_performance(self.distances, self.network.population)
        self.elapsed_time = 0.0
        self.resilience = 0.0
        return self.observe()

    def observe(self):
        return {"x": self.x.copy(), "damage": self.damage.copy(), "mask": self.damage > 0}

    def step(self, action):
        if not isinstance(action, (int, np.integer)) or not 0 <= action < self.damage.size:
            raise ValueError("Action must be a road index")
        if self.damage[action] <= 0:
            raise ValueError("The selected road is intact or already repaired")
        duration = self.damage[action] * self.network.length_m[action] / self.repair_length_unit_m
        before = float(self.x.mean())
        self.damage[action] = 0.0
        self.distances = shortest_travel_times(self.network, self.damage)
        self.x = node_performance(self.distances, self.network.population)
        after = float(self.x.mean())
        reward = (after - before) / duration
        self.elapsed_time += duration
        self.resilience += (before + after) * duration / 2.0
        info = {"duration": float(duration), "elapsed_time": float(self.elapsed_time),
                "efficiency": after, "resilience": float(self.resilience)}
        return self.observe(), float(reward), not bool((self.damage > 0).any()), info
