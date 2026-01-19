# Genetic Algorithm in Python

## Description

This project aims to test the genetic algorithm method to play games, such as the Snake game.

### What is a Genetic Algorithm ?

A genetic algorithm (GA) is a type of evolutionary algorithms inspired by the process of natural selection. Genetic algorithms are commonly used to generate solutions to optimization and search problems via biologically inspired technics: selection, crossover and mutation.

### Methodology

- Start from N individuals with randomly generated neurons

For the next generation follow these steps:
    - Compute the fitness score (i.e. how well they performed to the task) of individuals of the previous generation.
    - Elitism: From the previous generation, keep the K best individuals as is. They will represent 2% of the new generation.
    - Tournament Selection: From the previous generation, choose the best individual in a random sample of individuals. They will represent 48% of the new population.
    - Crossover and mutation:  From the previous kept individuals, mix their neurons weights randomly and add noise. They will represent 35% of the new population.
    - New individuals: create random new individuals. They will represent 15% of the new population.

### Fitness function and rewards

