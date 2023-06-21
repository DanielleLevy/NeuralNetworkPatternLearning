import numpy as np
import random
import matplotlib.pyplot as plt


wnet_file = 'wnet.txt'


def generate_initial_population(pop_size):
    population = []
    for _ in range(pop_size):
        # Generate a random number of layers
        num_layers = np.random.randint(2, 10)

        # Generate random layer sizes for the internal layers
        internal_layer_sizes = np.random.randint(2, 16, size=num_layers - 2)

        # Concatenate the layer sizes to create the complete layer_sizes list
        layer_sizes = [16] + list(internal_layer_sizes) + [1]

        neural_network = NeuralNetwork(layer_sizes)
        population.append(neural_network)
    return population


def fitness_score(neural_network, input,output):
    correct_predictions = 0
    total_predictions = len(input)


    # Pass the entire dataset to feedforward and get predicted labels
    predicted_labels = neural_network.feedforward(input)

    # Compare predicted labels with targets
    for predicted_label, target in zip(predicted_labels, output):
        if predicted_label == target:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy

def unflatten_list(flat_list, shapes):
    unflattened_list = []
    index = 0

    for shape in shapes:
        size = np.prod(shape)
        sublist = np.array(flat_list[index:index+size]).reshape(shape)
        unflattened_list.append(sublist)
        index += size

    return unflattened_list



def crossover(parent_1_nn, parent_2_nn):
    shapes1 = [a.shape for a in parent_1_nn.weights]
    shapes2 = [a.shape for a in parent_2_nn.weights]
    genes1 = np.concatenate([a.flatten() for a in parent_1_nn.weights])
    genes2 = np.concatenate([a.flatten() for a in parent_2_nn.weights])
    child_genes = []

    # Padding
    if genes1.size <= genes2.size:
        split = random.randint(0, len(genes1) - 1)
        child_genes = np.asarray(genes1[0:split].tolist() + genes2[split:].tolist())
        shapes = shapes2
        num_layers = parent_2_nn.layer_sizes

    elif genes1.size > genes2.size:
        split = random.randint(0, len(genes2) - 1)
        child_genes = np.asarray(genes2[0:split].tolist() + genes1[split:].tolist())
        shapes = shapes1
        num_layers = parent_1_nn.layer_sizes

    child = NeuralNetwork(num_layers)
    child.activation_function = random.choice([parent_1_nn.activation_function, parent_2_nn.activation_function])
    child.weights = unflatten_list(child_genes, shapes)
    return child


def flatten_list(arr):
    flattened = []
    for sublist in arr:
        if isinstance(sublist, np.ndarray):
            flattened.extend(sublist.flatten())
        else:
            flattened.extend(sublist)
    return flattened

def reshape_list(arr, shape):
    reshaped = []
    index = 0
    for dim in shape:
        if isinstance(dim, list):
            subshape = dim
            sublist_len = np.prod(subshape)
            sublist = reshape_list(arr[index:index+sublist_len], subshape)
            reshaped.append(sublist)
            index += sublist_len
        else:
            sublist = arr[index:index+dim]
            reshaped.append(sublist)
            index += dim
    return reshaped

def mutate(candidate):
    mutated_weights = []
    for weight in candidate.weights:
        mutated_weight = weight + np.random.normal(0, 0.1, size=weight.shape)
        mutated_weights.append(mutated_weight)
        candidate.weights = mutated_weights
    return candidate


def calc_score(population,input,output):
    fitness_scores = []
    sumFintnesScore = 0
    worst_score = float('inf')
    # Compute fitness scores
    for individual in population:
        score = fitness_score(individual,input,output)
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
def split_data(data):
    binary_strings = []
    labels = []
    for item in data:
        binary_string, label = item
        binary_strings.append(binary_string)
        labels.append(label)
    return binary_strings, labels

def run_genetic_algorithm(learnData,testData, population_size, crossover_rate, mutation_rate, max_generations, isfirst=0):
    convergence=0
    # Generate initial population
    generations=[]
    avg_scores = []  # replace with actual average scores for each generation
    bad_scores = []  # replace with actual bad scores for each generation
    best_scores=[]
    train_inputs, train_labels=split_data(learnData)
    test_inputs, test_labels = split_data(testData)
    population = generate_initial_population(population_size)
    # Evolve population
    for generation in range(max_generations):
        fitness_scores, sumFintnesScore, avg_score,worst_score,best_score = calc_score(population, train_inputs,train_labels)
        generations.append(generation)
        avg_scores.append(avg_score)
        bad_scores.append(worst_score)
        best_scores.append(best_score)
        # Apply selection, crossover, and mutation operators to population
        population = evolve_population(population,fitness_scores,sumFintnesScore,crossover_rate,mutation_rate)
        fitness_scores, sumFintnesScore, avg_score,worst_score,best_score = calc_score(population, train_inputs,train_labels)
        population,fitness_scores=cut_population(population,fitness_scores,population_size)
        avg_score=sum(fitness_scores)/len(fitness_scores)
        if abs(avg_score-avg_scores[-1]) == 0:
            convergence += 1
        else:
            convergence=0
        if convergence == 10:
            sol, fitness = chosen_sol(population, fitness_scores)
            return True, sol, fitness, generations, avg_scores, bad_scores,best_scores
    sol, fitness = chosen_sol(population, fitness_scores)
    if(isfirst==1):
        graph_and_txt(sol,generations,avg_scores,bad_scores,best_scores,population_size,crossover_rate,mutation_rate,max_generations,test_inputs,test_labels)
    return False, sol, fitness, generations, avg_scores, bad_scores,best_scores
def graph_and_txt(sol, generations, avg_scores, bad_scores, best_scores, pop_size, crossover_rate, mutation_rate, max_generations, test_inputs, test_labels):
    with open(wnet_file, 'w') as file:
        file.write("Activation Function: " + str(sol.activation_function) + "\n")  # Write the activation function to the WNET file
        for weights in sol.weights:
            file.write("Weights:\n")
            np.savetxt(file, weights)
    acc = fitness_score(sol, test_inputs, test_labels)
    print("Accuracy:", acc)
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
def run_genetic_algorithm_wrapper_to_check_conv(learnData,testData,pop_size=0 ,crossover_rate=0, mutation_rate=0, max_generations=0):
    test_inputs, test_labels=split_data(testData)
    flag, sol, fitness, generations, avg_scores, worst_score, best_scores = run_genetic_algorithm(learnData,testData,
            pop_size, crossover_rate, mutation_rate, max_generations, 1)

    best_fitness_scores=[]
    solutions=[]
    generations_conv=[]
    avg_scores_conv=[]
    bad_scores=[]
    best_scores_conv=[]
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
            flag, sol, fitness, generations, avg_scores, worst_score, best_scores = run_genetic_algorithm(learnData,testData,
                    pop_size, crossover_rate, mutation_rate, max_generations, 0)

        max_index = best_fitness_scores.index(max(best_fitness_scores))
        graph_and_txt(solutions[max_index],generations_conv[max_index],avg_scores_conv[max_index],bad_scores[max_index],best_scores_conv[max_index],pop_size,crossover_rate,mutation_rates[max_index],generations_conv[max_index][-1],test_inputs,test_labels)

def xavier_init(shape):
        n_inputs, n_outputs = shape[1], shape[0]
        limit = np.sqrt(6 / (n_inputs + n_outputs))
        return np.random.uniform(-limit, limit, shape)


def load_weights(wnet_file):
        with open(wnet_file, 'r') as file:
            lines = file.readlines()

        activation_function = int(lines[0].split(':')[1].strip())
        weights = []
        current_line = 1

        while current_line < len(lines):
            if lines[current_line].startswith('Weights:'):
                current_line += 1
                weight_matrix = []
                while current_line < len(lines) and lines[current_line].strip() != 'Weights:':
                    row = [float(val) for val in lines[current_line].strip().split()]
                    weight_matrix.append(row)
                    current_line += 1
                weights.append(np.array(weight_matrix))
            else:
                current_line += 1
        layer_sizes = [weights[0].shape[0]] + [w.shape[1] for w in weights]
        nn=NeuralNetwork(layer_sizes)
        nn.weights=weights
        nn.activation_function=activation_function
        return nn





class NeuralNetwork:
    def __init__(self, layer_sizes):
            self.layer_sizes = layer_sizes
            self.num_layers = len(layer_sizes)
            self.activation_function = random.choice([0, 1, 2])
            self.weights = [xavier_init((x, y)) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]

    def sigmoid(self,z):
        return 1.0 / (1.0 + np.exp(-z))

    def likki(self, Z):
        # Likki activation function
        return np.log(1 + np.exp(Z))

    def relu(self, Z):
        # ReLU activation function
        return np.maximum(0, Z)

    def feedforward(self, inputs):
        # Perform feed-forward using the selected activation function
        if self.activation_function == 0:
            activation = self.sigmoid
        elif self.activation_function == 1:
            activation = self.relu
        elif self.activation_function == 2:
            activation = self.likki
        a = inputs
        for w in self.weights:
            z = np.dot(a, w)
            a = activation(z)

        # Apply a threshold of 0.5 to classify outputs
        predicted_labels = np.where(a < 0.5, 0, 1)

        return predicted_labels




def load_data(file_path_learn,file_path_test):
    dataLearn = []
    dataTest=[]
    with open(file_path_learn, 'r') as file:
        for line in file:
            line = line.strip()
            binary_string = line[:16]
            binary_string= np.array([int(bit) for bit in binary_string])
            label = int(line[-1])
            dataLearn.append((binary_string, label))
    with open(file_path_test, 'r') as file:
        for line in file:
            line = line.strip()
            binary_string = line[:16]
            binary_string = np.array([int(bit) for bit in binary_string])
            label = int(line[-1])
            dataTest.append((binary_string, label))
    return dataLearn,dataTest
def load_data_without_label(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            binary_string = line[:16]
            binary_string = np.array([int(bit) for bit in binary_string])
            data.append(binary_string)
    return data


# Example usage
