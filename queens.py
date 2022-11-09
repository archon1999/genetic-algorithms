from copy import deepcopy

import numpy as np


class Individual():
    def __init__(self, number_of_queens):
        self.gene = np.arange(0, number_of_queens)
        self.fitness = np.inf

    def update_fitness(self):
        self.fitness = 0
        for i in range(len(self.gene)):
            for j in range(i+1, len(self.gene)):
                col_i = self.gene[i]
                col_j = self.gene[j]
                if col_i - i == col_j - j or col_i + i == col_j + j:
                    self.fitness += 1

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __str__(self):
        return str(self.chromosomes)


class QueensProblemSolver():
    def __init__(self, number_of_queens,
                 population_size=10,
                 selection_size=5,
                 mutation_probality=0.5):
        self.number_of_queens = number_of_queens
        self.population_size = population_size
        self.selection_size = selection_size
        self.mutation_probality = mutation_probality
        self.population: list[Individual] = None
        self.solutions = []

    def run(self, max_iters=1000):
        self.population = self.generate_initial_population()
        for _ in range(max_iters):
            if self.check_terminate():
                break

            self.mutation()
            self.selection()
            self.save_solution()

        return self.population[0]

    def check_terminate(self):
        return self.population[0].fitness == 0

    def generate_initial_population(self):
        population = [Individual(self.number_of_queens)
                      for _ in range(self.population_size)]
        return population

    def selection(self):
        self.population.sort()
        self.population = self.population[:self.selection_size]
        for i in range(self.population_size - self.selection_size):
            self.population.append(deepcopy(self.population[i]))

    def mutation(self):
        for i in range(self.population_size):
            if np.random.random() <= self.mutation_probality:
                individual = self.population[i]
                i = np.random.randint(0, self.number_of_queens-1)
                j = np.random.randint(0, self.number_of_queens-1)
                gene = individual.gene
                gene[i], gene[j] = gene[j], gene[i]
                individual.update_fitness()


def main():
    population_size = 100
    selection_size = 50
    mutation_probality = 0.6
    number_of_queens = 100
    solver = QueensProblemSolver(number_of_queens=number_of_queens,
                                 population_size=population_size,
                                 selection_size=selection_size,
                                 mutation_probality=mutation_probality)
    solution = solver.run(10000)

    print(solution.gene)
    print(solution.fitness)
    board = [['.' for _ in range(number_of_queens)]
             for _ in range(number_of_queens)]
    for row_index, col_index in enumerate(solution.gene):
        board[row_index][col_index] = 'Q'

    for row in board:
        print(*row)


if __name__ == '__main__':
    main()
