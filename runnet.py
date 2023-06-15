import main
if __name__ == '__main__':
    learning_file = 'nn0.txt'
    wnet_file = 'wnet.txt'
    output_file = 'output.txt'  # Specify the name of the output file


    layer_sizes = [16, 8, 1]  # Modify this according to your desired network structure
    neural_network = main.NeuralNetwork(layer_sizes)
    neural_network.load_weights(wnet_file)
    learning_data = main.load_data_without_label(learning_file)
    with open(output_file, 'w') as f:
        for input_data in learning_data:
            output = neural_network.feedforward(input_data)
            f.write(f'{output}\n')


