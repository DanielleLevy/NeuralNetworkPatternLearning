import main

def run_runnet(structure_file,data_file):
    network = main.load_weights(structure_file)
    data = main.load_data_without_label(data_file)
    output = network.feedforward(data)
    with open("output.txt", 'w') as file:
            for label in output:
                file.write(str(label.item()) + '\n')
