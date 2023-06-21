import main


# Parameters
POPULATION_SIZE =200
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
MAX_GENERATIONS = 300



def run_buildnet(test_file, train_file):
    # Load the data
    learning_data, test_data = main.load_data(train_file, test_file)
    main.run_genetic_algorithm_wrapper_to_check_conv(learning_data,test_data,POPULATION_SIZE,CROSSOVER_RATE,MUTATION_RATE,MAX_GENERATIONS)