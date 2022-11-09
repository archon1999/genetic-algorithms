from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np


class Individual():
    def __init__(self, number_of_items):
        self.gene = np.zeros(number_of_items)
        self.fitness = 0

    def update_fitness(self, weights, values, max_capacity):
        if np.sum(self.gene*weights) > max_capacity:
            self.fitness = 0
        else:
            self.fitness = np.sum(self.gene*values)

    def __lt__(self, other):
        return self.fitness > other.fitness

    def __str__(self):
        return str(self.gene)


class KnapsackProblemSolver():
    def __init__(self, weights, values, labels, max_capacity,
                 population_size=10,
                 selection_size=5,
                 mutation_probality=0.5):
        self.number_of_items = len(weights)
        self.weights = weights
        self.values = values
        self.labels = labels
        self.max_capacity = max_capacity
        self.population_size = population_size
        self.selection_size = selection_size
        self.mutation_probality = mutation_probality
        self.population: list[Individual] = None
        self.solutions = []

    def run(self, max_iters=1000):
        self.population = self.generate_initial_population()
        for _ in range(max_iters):
            self.mutation()
            self.selection()
            self.save_solution()

        return self.population[0]

    def generate_initial_population(self):
        population = [Individual(self.number_of_items)
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
                j = np.random.randint(0, self.number_of_items)
                gene = individual.gene
                gene[j] = np.random.randint(0, 2)
                individual.update_fitness(self.weights,
                                          self.values,
                                          self.max_capacity)

    def save_solution(self):
        population_max_fitness = self.population[0].fitness
        fitness_list = list(map(lambda g: g.fitness, self.population))
        population_avg_fitness = np.average(fitness_list)
        self.solutions.append((population_avg_fitness,
                               population_max_fitness))

    def print_individual_info(self, individual: Individual):
        info = str()
        sum_c = 0
        sum_w = 0
        for g, label, w, c in zip(individual.gene,
                                  self.labels,
                                  self.weights,
                                  self.values):
            if g:
                info += f'{label}: w={w}, c={c}\n'
                sum_c += c
                sum_w += w

        info += f'Суммарная ценность: {sum_c}\n'
        info += f'Суммарный вес: {sum_w}\n'
        print(info)


def main():
    np.random.seed(1)
    values = [150, 35, 200, 160, 60, 45, 60, 40, 30, 10, 70, 30,
              15, 10, 40, 70, 75, 80, 7, 12, 50, 10]
    weights = [9, 13, 153, 50, 15, 68, 27, 39, 23, 52, 11, 32,
               24, 48, 73, 42, 43, 22, 7, 18, 4, 30]
    labels = ['Карта', 'Компас', 'Вода', 'Сэндвич', 'Глюкоза',
              'Кружка', 'Банан', 'Яблоко', 'Сыр', 'Пиво',
              'Кремот загара', 'Камера', 'Футболка', 'Брюки',
              'Зонтик', 'Непромокаемые штаны', 'Непромокаемый плащ',
              'Бумажник', 'Солнечны еочки', 'Полотенце', 'Носки', 'Книга']
    max_capacity = 400
    population_size = 20
    selection_size = 5
    mutation_probality = 0.6
    solver = KnapsackProblemSolver(weights, values, labels, max_capacity,
                                   population_size=population_size,
                                   selection_size=selection_size,
                                   mutation_probality=mutation_probality)
    solution = solver.run(50)
    solver.print_individual_info(solution)

    plt.plot(solver.solutions)
    plt.title('Зависимость макс. и средней приспособленности от поколения')
    plt.ylabel('Макс./средняя приспособленность')
    plt.xlabel('Популяция')
    plt.show()


if __name__ == '__main__':
    main()
