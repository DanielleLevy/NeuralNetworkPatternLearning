import main
# Parameters
POPULATION_SIZE = 100
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
MAX_GENERATIONS = 100
score_calls = 0
N=10
if __name__ == '__main__':
    learning_file = 'nn0.txt'
    test_file='test.txt'
    wnet_file = 'wnet.txt'

    # Load the learning data
    learning_data = main.load_data(learning_file)
    main.run_genetic_algorithm_wrapper_to_check_conv(learning_data,POPULATION_SIZE,CROSSOVER_RATE,MUTATION_RATE,MAX_GENERATIONS)
    #test_data = main.load_data_without_label(test_file)
    #for input_data in test_data:
     #   output = neural_network.feedforward(input_data)
      #  f.write(f'{output}\n')

