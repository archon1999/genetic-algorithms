import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class TravelingSalesmanSolution():
    def __init__(self, dist, path):
        self.dist = dist
        self.path = path

    def __lt__(self, other):
        return self.dist < other.dist


class TravelingSalesmanSolver():
    def __init__(self, distances, n_ants, n_best, n_iterations, decay,
                 alpha=1, beta=1):
        self.distances = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.global_best_solution = TravelingSalesmanSolution(np.inf, [])
        self.solutions = []

    def run(self):
        for _ in range(self.n_iterations):
            all_solutions = self.generate_solutions()
            self.daemon_actions(all_solutions)
            self.pheromone_update(all_solutions)

        return self.global_best_solution

    def generate_solutions(self):
        all_solutions = [self.ant_run(0) for _ in range(self.n_ants)]
        all_solutions.sort()
        return all_solutions

    def daemon_actions(self, all_solutions):
        best_solution = min(all_solutions)
        self.update_global_solution(best_solution)

    def pheromone_update(self, all_solutions):
        self.spread_pheronome(all_solutions)
        self.pheromone *= self.decay

    def update_global_solution(self, solution):
        self.global_best_solution = min(self.global_best_solution,
                                        solution)
        self.solutions.append((solution.dist, self.global_best_solution.dist))

    def spread_pheronome(self, all_solutions):
        for solution in all_solutions[:self.n_best]:
            for move in solution.path:
                self.pheromone[move] += 1.0 / self.distances[move]

    def ant_run(self, start):
        path = []
        dist = 0
        visited = set()
        visited.add(start)
        prev = start
        for _ in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev],
                                  visited)
            path.append((prev, move))
            dist += self.distances[(prev, move)]
            prev = move
            visited.add(move)

        path.append((prev, start))
        dist += self.distances[(prev, start)]
        return TravelingSalesmanSolution(dist, path)

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0
        row = pheromone ** self.alpha * ((1.0 / dist) ** self.beta)
        norm_row = row / row.sum()
        all_numbers = range(len(self.distances))
        move = np.random.choice(all_numbers, 1, p=norm_row)[0]
        return move


def main():
    n = np.random.randint(41, 100)
    n_iters = 500
    n_ants = 4
    n_best = 4
    np.random.seed(1)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                distances[i][j] = np.inf
            else:
                distances[i][j] = distances[j][i] = np.random.randint(1, 10000)

    solver = TravelingSalesmanSolver(distances, n_ants, n_best, n_iters, 0.95,
                                     alpha=1, beta=1)
    best_solution = solver.run()
    min_dist = int(best_solution.dist)
    path = best_solution.path
    subax1 = plt.subplot(121)
    subax1.set_title('График')
    plt.plot(solver.solutions)
    plt.ylabel('Минимальное расстояние')
    plt.xlabel('Популяция')

    subax2 = plt.subplot(122)
    subax2.set_title('Наилучшее решение')
    subax2.text(-30, 0, f'Минимальное расстояние: {min_dist}')
    subax2.text(-30, 10, f'Количество итераций: {n_iters}')
    G = nx.Graph(path)
    pos = nx.circular_layout(G, scale=100)
    nx.draw(G, pos=pos, with_labels=True, font_weight='bold')
    plt.show()


if __name__ == '__main__':
    for i in range(21, 30):
        main(i)
