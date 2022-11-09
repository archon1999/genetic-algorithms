import string
from copy import deepcopy

import numpy as np


class Individual():
    def __init__(self, number_of_coefficients):
        self.gene = np.zeros(number_of_coefficients)
        self.fitness = np.inf

    def update_fitness(self, coefficients, c):
        self.fitness = abs(np.sum(self.gene*coefficients) - c)

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __str__(self):
        return ', '.join(map(lambda x: f'{x[1]}={int(x[0])}',
                         zip(self.gene, string.ascii_lowercase)))


class DiophantineEquationSolver():
    def __init__(self, coefficients, c,
                 population_size=10,
                 selection_size=5,
                 mutation_probality=0.5):
        self.coefficients = coefficients
        self.c = c
        self.number_of_coefficients = len(coefficients)
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
        population = [Individual(self.number_of_coefficients)
                      for _ in range(self.population_size)]
        return population

    def check_terminate(self):
        return self.population[0].fitness == 0

    def selection(self):
        self.population.sort()
        self.population = self.population[:self.selection_size]
        for i in range(self.population_size - self.selection_size):
            self.population.append(deepcopy(self.population[i]))

    def mutation(self):
        for i in range(self.population_size):
            if np.random.random() <= self.mutation_probality:
                individual = self.population[i]
                j = np.random.randint(0, self.number_of_coefficients)
                individual.gene[j] += np.random.randint(-10, 10)
                individual.update_fitness(self.coefficients, self.c)

    def __str__(self):
        equation_info = ' + '.join(map(lambda x: f'{int(x[0])}{x[1]}',
                                   zip(self.coefficients,
                                       string.ascii_lowercase)))
        equation_info += ' = ' + str(self.c)
        return equation_info


def main():
    coefficients = [1, 2, 3, 4, 5, 6]
    c = 123456
    population_size = 50
    selection_size = 10
    mutation_probality = 0.6
    solver = DiophantineEquationSolver(coefficients, c,
                                       population_size=population_size,
                                       selection_size=selection_size,
                                       mutation_probality=mutation_probality)
    print(solver)
    solution = solver.run(10000)
    print(solution)


if __name__ == '__main__':
    main()
