import numpy as np
import random
import matplotlib.pyplot as plt


wnet_file = 'wnet.txt'


def generate_initial_population(pop_size):
    population = []
    for _ in range(pop_size):
        # Generate a random number of layers between 2 and 5
        num_layers = np.random.randint(2, 6)

        # Generate random layer sizes for the internal layers
        internal_layer_sizes = np.random.randint(2, 16, size=num_layers - 2)

        # Concatenate the layer sizes to create the complete layer_sizes list
        layer_sizes = [16] + list(internal_layer_sizes) + [1]

        neural_network = NeuralNetwork(layer_sizes)
        population.append(neural_network)
    return population

def fitness_score(neural_network, learning_data):
    correct_predictions = 0
    total_predictions = len(learning_data)
    for input_data, target in learning_data:
        output = neural_network.feedforward(input_data)
        predicted_label = int(round(output))
        if predicted_label == target:
            correct_predictions += 1
    accuracy = correct_predictions / total_predictions
    return accuracy


def crossover(parent1, parent2):
    offspring_population = []
    child = NeuralNetwork(parent1.layer_sizes)  # Create a new neural network
    min_layers = min(len(parent1.weights), len(parent2.weights))
    crossover_point = np.random.randint(0, min_layers)

    child.weights[:crossover_point] = parent1.weights[:crossover_point]
    child.weights[crossover_point:] = parent2.weights[crossover_point:]

    child.biases[:crossover_point] = parent1.biases[:crossover_point]
    child.biases[crossover_point:] = parent2.biases[crossover_point:]

    return child


def mutate(candidate):
    mutated_offsprings = []
    mutated_weights = []
    mutated_biases = []
    for weight, bias in zip(candidate.weights, candidate.biases):
        mutated_weight = weight + np.random.normal(0, 0.1, size=weight.shape)
        mutated_bias = bias + np.random.normal(0, 0.1, size=bias.shape)
        mutated_weights.append(mutated_weight)
        mutated_biases.append(mutated_bias)
        candidate.weights = mutated_weights
        candidate.biases = mutated_biases
    return candidate


def calc_score(population,learnData):
    fitness_scores = []
    sumFintnesScore = 0
    worst_score = float('inf')
    # Compute fitness scores
    for individual in population:
        score = fitness_score(individual,learnData)
        fitness_scores.append(score)
        sumFintnesScore += score
        if score < worst_score:
            worst_score = score
    avg_score = sumFintnesScore / len(population)
    best_score=max(fitness_scores)
    return fitness_scores, sumFintnesScore, avg_score, worst_score,best_score

def evolve_population(population,fitness_scores,sumFintnesScore,crossover_rate,mutation_rate):

    relative_fitness_scores = []
    for score in fitness_scores:
        relative_fitness_score = score / sumFintnesScore
        relative_fitness_scores.append(relative_fitness_score)

    # Calculate the weighted selection probabilities
    probabilities = [score for score in relative_fitness_scores]

    # Normalize the probabilities to sum up to 1
    total_probability = sum(probabilities)
    probabilities = [score / total_probability for score in probabilities]


    # Perform crossover
    num_offspring = int(len(population) * crossover_rate)
    offspring = []
    for _ in range(num_offspring):
        # Choose two random parents from the population, biased by relative fitness scores
        parents = random.choices(population, weights=relative_fitness_scores, k=2)
        parent1, parent2 = parents

        # Perform crossover operation to create a child
        child = crossover(parent1, parent2)
        offspring.append(child)

    # Perform mutation
    num_mutants = int((len(population)  * mutation_rate))
    mutants = []
    for i in range(num_mutants):
        parent = random.choice(population)
        mutant = mutate(parent)
        mutants.append(mutant)
    new_population = population + offspring + mutants
    return new_population
def cut_population(population, fitness_scores,popsize):
    sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)

    # Keep only the top  indices
    top_indices = sorted_indices[:popsize]

    # Create a new population containing only the top individuals
    new_population = [population[i] for i in top_indices]
    fitness_scores=[fitness_scores[i] for i in top_indices]

    return new_population,fitness_scores
def chosen_sol(population,fitness_scores):
    sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
    solution=population[sorted_indices[0]]
    best_f=fitness_scores[sorted_indices[0]]

    return solution,best_f

def run_genetic_algorithm(learnData,population_size, crossover_rate, mutation_rate, max_generations,isfirst=0):
    convergence=0
    # Generate initial population
    generations=[]
    avg_scores = []  # replace with actual average scores for each generation
    bad_scores = []  # replace with actual bad scores for each generation
    best_scores=[]
    population = generate_initial_population(population_size)
    # Evolve population
    for generation in range(max_generations):
        fitness_scores, sumFintnesScore, avg_score,worst_score,best_score = calc_score(population,learnData)
        generations.append(generation)
        avg_scores.append(avg_score)
        bad_scores.append(worst_score)
        best_scores.append(best_score)
        # Apply selection, crossover, and mutation operators to population
        population = evolve_population(population,fitness_scores,sumFintnesScore,crossover_rate,mutation_rate)
        fitness_scores, sumFintnesScore, avg_score,worst_score,best_score = calc_score(population,learnData)
        population,fitness_scores=cut_population(population,fitness_scores,population_size)
        avg_score=sum(fitness_scores)/len(fitness_scores)
        if abs(avg_score-avg_scores[-1]) <= 0.05:
            convergence += 1
        else:
            convergence=0
        if convergence == 10:
            sol, fitness = chosen_sol(population, fitness_scores)
            return True, sol, fitness, generations, avg_scores, bad_scores,best_scores
    sol, fitness = chosen_sol(population, fitness_scores)
    if(isfirst==1):
        graph_and_txt(sol,generations,avg_scores,bad_scores,best_scores,population_size,crossover_rate,mutation_rate,max_generations)
    return False, sol, fitness, generations, avg_scores, bad_scores,best_scores
def graph_and_txt(sol, generations, avg_scores, bad_scores, best_scores, pop_size, crossover_rate, mutation_rate, max_generations):
    with open(wnet_file, 'w') as file:
        for weights, biases in zip(sol.weights, sol.biases):
            file.write("Weights:\n")
            np.savetxt(file, weights)
            file.write("Biases:\n")
            np.savetxt(file, biases)
            file.write("\n")

    # Create a figure and axis object
    fig, ax = plt.subplots( figsize=(14, 6))

    # Set the axis labels and title for the score distribution plot
    ax.set_xlabel("Generation")
    ax.set_ylabel("Score")
    ax.set_title("Average, Bad, and Best Scores by Generation")

    # Set the bar width
    bar_width = 0.35

    # Plot the average scores as a blue bar
    ax.bar(generations, avg_scores, color='#5DA5DA', width=bar_width, label='Average Scores')

    # Plot the bad scores as a red bar
    ax.bar([g + bar_width for g in generations], bad_scores, color='#FAA43A', width=bar_width, label='Bad Scores')

    # Add a line plot of the best scores over time
    ax.plot(generations, best_scores, color='#60BD68', linewidth=2, label='Best Scores')

    # Add the parameters as text above the title
    plt.text(0.5, 1.1, f"Population size: {pop_size}   Crossover rate: {crossover_rate}   Mutation rate: {mutation_rate}   Max generations: {max_generations}   ", ha='center', va='bottom', transform=ax.transAxes)
    # Add a legend to the score distribution plot
    ax.legend()

    # Save the figure as a PNG file
    plt.savefig("plot.png")

    # Show the figure
    plt.show()
def run_genetic_algorithm_wrapper_to_check_conv(learnData,pop_size=0 ,crossover_rate=0, mutation_rate=0, max_generations=0):

    flag, sol, fitness, generations, avg_scores, worst_score, best_scores = run_genetic_algorithm(learnData,
            pop_size, crossover_rate, mutation_rate, max_generations, 1)

    best_fitness_scores=[]
    solutions=[]
    generations_conv=[]
    avg_scores_conv=[]
    bad_scores=[]
    best_scores_conv=[]
    score_calls_conv=[]
    mutation_rates=[]
    if(flag==True):
        for i in range(5):
            best_fitness_scores.append(fitness)
            solutions.append(sol)
            generations_conv.append(generations)
            avg_scores_conv.append(avg_scores)
            bad_scores.append(worst_score)
            best_scores_conv.append(best_scores)
            mutation_rates.append(mutation_rate)
            mutation_rate=random.uniform(mutation_rate, 1)
            flag, sol, fitness, generations, avg_scores, worst_score, best_scores = run_genetic_algorithm(learnData,
                    pop_size, crossover_rate, mutation_rate, max_generations, 0)

        max_index = best_fitness_scores.index(max(best_fitness_scores))
        graph_and_txt(solutions[max_index],generations_conv[max_index],avg_scores_conv[max_index],bad_scores[max_index],best_scores_conv[max_index],pop_size,crossover_rate,mutation_rates[max_index],generations_conv[max_index][-1])




class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]

    def sigmoid(self,z):
        return 1.0 / (1.0 + np.exp(-z))
    def feedforward(self, a):
        a = np.array(list(map(int, a)))  # Convert string to array of integers
        a = a.astype(float)  # Convert input data to float type
        a = a.reshape(len(a), 1)  # Reshape input data as a column vector
        for w, b in zip(self.weights, self.biases):
            try:
                a = self.sigmoid(np.dot(w, a) + b)
            except:
                print("f")
        # Apply thresholding
        if(a<0.5): # Set values less than 0.5 to 0
            return 0
        else:        # Set values greater than or equal to 0.5 to 1

            return 1


    def load_weights(self, wnet_file):
        with open(wnet_file, 'r') as file:
            weights = []
            biases = []
            for line in file:
                line = line.strip()
                if line == "Weights:":
                    current_weights = []
                    continue
                elif line == "Biases:":
                    weights.append(np.array(current_weights))
                    current_weights = []
                    continue
                current_weights.append(list(map(float, line.split())))
            biases.append(np.array(current_weights))

        self.weights = weights
        self.biases = biases





def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            binary_string = line[:16]
            label = int(line[-1])
            data.append((binary_string, label))
    return data
def load_data_without_label(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            binary_string = line[:16]
            data.append(binary_string)
    return data


# Example usage
