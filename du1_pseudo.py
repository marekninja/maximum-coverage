#Maximum coverage problem solver
import argparse
from datetime import datetime
import sys

import pandas as pd
import numpy as np
import itertools
import random
import copy
parser = argparse.ArgumentParser(description='Genetic algo solver of Maximum Coverage Problem')
parser.add_argument('-m', type=int, help='number of warehouses')
parser.add_argument('-n', type=int, help='number of locations')
parser.add_argument('-v', type=int, help='maximum distance of warehouse reach')
parser.add_argument('-p',"--pop", type=int, help='desired size of population')
parser.add_argument('-g',"--gen", type=int, help='maximum number of generations')
parser.add_argument('-e',"--end", type=int, help='alternative way of ending, how many generations to have same max')
parser.add_argument("--mutprob", type=float, help='percentage (of population size) of mutated individuals')
parser.add_argument("--map", type=str, help='path to csv of map describing graph')
parser.add_argument("--weights", type=str, help='path to csv describing weights of locations')
args = parser.parse_args()

class OneSolution:
    def __init__(self, chosen_vertices = [0]*args.n, covered_locations = [0]*args.n, fitness = 0):
        self.chosen_vertices = list(chosen_vertices)
        self.covered_locations = list(covered_locations)
        self.fitness = fitness

# vytvorenie inicialnej populacie 
# metoda permutacii n-tice 1 v binarnom poli
def create_initial(m,n):
    first = []
    for i in range(n):
        if i < m:
            first.append(1)
        else:
            first.append(0)
    whole_population = list(set(itertools.permutations(first)))
    
    random.shuffle(whole_population)

    if args.pop < len(whole_population):
        population = whole_population[:args.pop]
    else:
        population = whole_population
        args.pop = len(whole_population)

    for i in range(len(population)):
        population[i] = OneSolution(population[i], [0]*args.n)
    return population

# urci fitnes jedinca
def eval_one(one,matrix,weights):
    # prechadza indexami
    for location_index in range(len(one.chosen_vertices)):
        # kontroluje, ci je vrchol vybrany
        if one.chosen_vertices[location_index] == 1:
            # ked je vybrany prejde riadok toho vrcholu v matici susednosti
            for i in range(len(matrix[location_index])):
                # ak nanho mame dosah
                if matrix[location_index][i] > 0:
                    # tak ho pridame do pola pokrytych lokalit
                    one.covered_locations[i] = one.covered_locations[i] + 1
    # mame vytvorene pole pokrytych lokalit, chceme fitness
    for index_of_location in range(len(one.covered_locations)):
        # ak je >0, mame ju pokrytu
        if one.covered_locations[index_of_location] > 0:
            # pripocitam vahu tejto lokality do fitnes jedinca
            one.fitness = one.fitness + weights[index_of_location]

# urci fitnes vsetkych jedincov populacie
def eval_initial(population,matrix,weights):
    for i in range(len(population)):
        eval_one(population[i],matrix,weights)

def select_one_parent(max,population):
    choice = random.random()*max
    sum = 0
    for one in population:
        sum = sum + one.fitness 
        if sum >= choice:
            return one


# vymeni dva vrcholi v poli a prepocita fitness
def swap_vertices(mutant,one,zero, matrix, weights):
    mutant.chosen_vertices[one] = 0
    mutant.chosen_vertices[zero] = 1

    # prepocitam ubytky a prirastky v pokryti a fitness
    for i in range(len(mutant.chosen_vertices)):
        if matrix[one][i] > 0:
            if mutant.covered_locations[i] == 0:
                break

            mutant.covered_locations[i] = mutant.covered_locations[i] - 1

            if mutant.covered_locations[i] == 0:
                mutant.fitness = mutant.fitness - weights[i]

        if matrix[zero][i] > 0:
            if mutant.covered_locations[i] == 0:
                mutant.fitness = mutant.fitness + weights[i]
            mutant.covered_locations[i] = mutant.covered_locations[i] + 1


# vymena dvoch nahodnych bitov v binarnom poli zvolenych vrcholov jedinca
# pocita inkrementalne fitness
def create_mutant(parent,matrix,weights):
    mutant = copy.deepcopy(parent)

    index1 = random.choice(tuple(range(args.n)))
    index2 = 0
    chosen1 = 0
    chosen2 = 0
    _check = True
    while (_check):
        index2 = random.choice(tuple(range(args.n)))
        chosen1 = mutant.chosen_vertices[index1]
        chosen2 = mutant.chosen_vertices[index2]
        if (index2 != index1) and (chosen1 != chosen2):
            _check = False

    # if chosen1 == 1 and chosen2 == 1: 
    # if chosen1 == 0 and chosen2 == 0:
    if chosen1 == 1 and chosen2 == 0:
        swap_vertices(mutant, index1, index2, matrix, weights)
    if chosen1 == 0 and chosen2 == 1:
        swap_vertices(mutant, index2, index1, matrix, weights)
    return mutant

# vytvori zmutovanu cast populacie
def generate_mutations(population,matrix,weights):
    mutants = []
    for i in range(round(len(population)*args.mutprob)):
        max = sum(one.fitness for one in population)
        parent = select_one_parent(max,population)
        mutants.append(create_mutant(parent,matrix,weights))
    return mutants

# vyberie elitnych a doplni vsetkych mutovanych
def create_new_population(population, mutants):
    elitsize = 1 - round(args.pop * args.mutprob)

    # zoradi populaciu podla fitness od najvacsej po najmensiu
    new_population = sorted(population, key=lambda one: one.fitness, reverse=True)
    new_population = new_population[:elitsize]

    if elitsize == 0 and len(mutants) == 0:
        sys.exit("Invalid combination mutprob and pop - when population size 1, use mutprob 1 or 0")
    new_population.extend(mutants)
    top = max(new_population, key=lambda one: one.fitness)
    return new_population, top


def genetic_algo(m,n,matrix,weights):
    population = create_initial(m,n)
    eval_initial(population,matrix,weights)
    if (m == n):
        return population[0]

    max_list = []
    for i in range(args.gen):
        mutants = generate_mutations(population,matrix,weights)
        population, top = create_new_population(population, mutants)
        print_top(top)
        max_list.append(top)
        # udrziavame pocet args.end v max liste
        max_list = max_list[-args.end:]
        # kontrola, ci mame pole maxov z rovnakych hodnot
        if (c == max_list[0] for c in max_list ) and len(max_list) == args.end:
            return i,True,max(max_list, key =lambda one: one.fitness)
    return 0, False, max(max_list, key=lambda one: one.fitness)

# vymaze z matice susednosti cesty, ktore su viac ako v
def prepare_matrix(v,matrix):
    # matrix = matrix.values
    for cell in np.nditer(matrix, op_flags=['readwrite']):
        if cell == 0:
            cell[...] = 1
        if cell > v:
            cell[...] = 0

    return matrix

def print_top(top):
    print("najlepsi jedinec: vybrate vrcholy: ", top.chosen_vertices, "pokryte oblasti: ", top.covered_locations,
          " jeho fitness: ", top.fitness)

if __name__ == '__main__':
    m = args.m
    if args.m == 0:
        print("Can't build warehouse! Set m > 0")
        exit(-1)
    n = args.n
    v = args.v
    matrix = pd.read_csv(args.map, sep=";").values
    weights = pd.read_csv(args.weights, sep=";").values[0]

    prepare_matrix(v,matrix)
    random.seed(datetime.now())
    i, _check, top = genetic_algo(m,n,matrix, weights)
    if (_check):
        print("Ukoncene predcasne! Po ", i+1, " generaciach.")
    else:
        print("Pocet generacii vycerpany!")
    print_top(top)
    exit(0)



