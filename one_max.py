from copy import deepcopy

import numpy as np


class Individual():
    def __init__(self, number_of_bits):
        self.bits = np.zeros(number_of_bits)
        self.fitness = 0

    def update_fitness(self):
        self.fitness = np.sum(self.bits)

    def __lt__(self, other):
        return self.fitness > other.fitness

    def __str__(self):
        return str(self.bits)


class OneMaxSolver():
    def __init__(self, number_of_digits,
                 population_size=10,
                 selection_size=5,
                 mutation_probality=0.5):
        self.number_of_digits = number_of_digits
        self.population_size = population_size
        self.selection_size = selection_size
        self.mutation_probality = mutation_probality
        self.population: list[Individual] = None

    def run(self, max_iters=1000):
        self.population = self.generate_initial_population()
        for _ in range(max_iters):
            if self.check_terminate():
                return self.population[0]

            self.mutation()
            self.selection()

    def generate_initial_population(self):
        population = [Individual(self.number_of_digits)
                      for _ in range(self.population_size)]
        return population

    def check_terminate(self):
        return self.population[0].fitness == self.number_of_digits

    def selection(self):
        self.population.sort()
        self.population = self.population[:self.selection_size]
        for i in range(self.population_size - self.selection_size):
            self.population.append(deepcopy(self.population[i]))

    def mutation(self):
        for i in range(self.population_size):
            if np.random.random() <= self.mutation_probality:
                gene = self.population[i]
                j = np.random.randint(0, self.number_of_digits)
                gene.bits[j] = np.random.randint(0, 2)
                gene.update_fitness()


def main():
    population_size = 10
    mutation_probality = 0.5
    selection_size = 5
    solver = OneMaxSolver(10,
                          population_size=population_size,
                          selection_size=selection_size,
                          mutation_probality=mutation_probality)
    solution = solver.run()
    print(solution)


if __name__ == '__main__':
    main()
