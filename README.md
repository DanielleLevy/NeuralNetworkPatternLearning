# NeuralNetworkPatternLearning
This project implements a genetic algorithm-based approach to identify patterns in binary strings using neural networks. The goal is to build neural networks that can predict the legality of binary strings based on patterns learned from training data.

## Prerequisites
Python 3.x
Standard libraries: numpy, random, matplotlib
## Usage
Clone the repository or download the source code.

Open the terminal or command prompt and navigate to the project directory.

Run the GUI program by executing the following command:

Copy code
python gui.py
The GUI window will open, allowing you to choose the options and input files for building and running the neural networks.

Fill in the file paths for the training and test files in the GUI.

Click the "Run buildnet" button to start the genetic algorithm-based network building process. This may take some time.

Once the building process is complete, a file named wnet.txt will be generated, which contains the network structure and weights.

Fill in the file paths for the structure file (wnet.txt) and the data file for classification in the GUI.

Click the "Run runnet" button to run the neural network on the test data. The classification results will be saved in an output file.

View the output file to see the predicted classifications for the test data.

## File Descriptions
gui.py: GUI program for selecting options and input files, and running the buildnet and runnet programs.
buildnet.py: Module for running the genetic algorithm-based network building process.
runnet.py: Module for running the neural network on new data for classification.
main.py: Main module containing the core functions for loading data, fitness scoring, evolution operators, and running the genetic algorithm.
##Data Files
The project requires two data files named nn0.txt and nn1.txt, which contain 20,000 binary strings each with a corresponding legality label (0 or 1). nn0.txt is easier to identify the legality, while nn1.txt is more difficult.

## Outputs
The output of the buildnet program is the wnet.txt file, which contains the definitions of the network structure and the weights of the built network.

The output of the runnet program is an output file with the classification results for the input data.

## Performance Evaluation
The performance of the neural networks is evaluated based on their accuracy in predicting the legality of binary strings. The accuracy is calculated as the ratio of correct predictions to the total number of predictions.

The performance of the networks on the learning group and the test group should be described in the report that accompanies the submission.
## Background
The Neural Network Pattern Learning project aims to develop a system that can identify patterns in binary strings using neural networks. The goal is to create neural networks capable of predicting the legality of binary strings based on patterns learned from training data.

Neural networks are computational models inspired by the structure and function of the human brain. They are composed of interconnected nodes, called neurons, organized into layers. Each neuron receives input signals, processes them, and produces an output signal that can be passed to other neurons. Neural networks have the ability to learn and generalize from input-output patterns, making them suitable for pattern recognition tasks.

In this project, a genetic algorithm-based approach is employed to build the neural networks. Genetic algorithms are optimization algorithms inspired by the process of natural selection and evolution. They work by iteratively generating a population of candidate solutions, evaluating their fitness, and using genetic operators (such as selection, crossover, and mutation) to evolve better solutions over generations.

## Code Explanation
The code consists of several modules and files that work together to implement the pattern learning system using neural networks. Here's an overview of the key components:

gui.py: This module provides a graphical user interface (GUI) for the user to interact with the system. It allows the user to choose options, input files, and run the building and running processes.

buildnet.py: This module contains the implementation of the genetic algorithm-based network building process. It defines the procedures for initializing the population, evaluating fitness, selecting parents, applying crossover and mutation, and generating the final network structure and weights.

runnet.py: This module implements the functionality for running the trained neural network on new data for classification. It takes the network structure and weights obtained from the building process and applies them to make predictions on test data.

main.py: This module contains the core functions used by buildnet.py and runnet.py. It includes functions for loading the input data, scoring the fitness of candidate networks, implementing the evolution operators, and executing the genetic algorithm.


During the building process, the genetic algorithm evolves a population of neural networks by iteratively selecting parents, performing crossover and mutation, and evaluating their fitness based on how well they predict the legality of the binary strings. The process continues until a satisfactory network structure and weights are obtained, which are saved in the wnet.txt file.

The runnet process utilizes the trained network structure and weights from wnet.txt to classify new binary strings and produce an output file with the classification results.

It is important to evaluate the performance of the trained neural networks on both the learning group (training data) and the test group. The accuracy of the networks in predicting the legality of binary strings can be used as a measure of performance.

## Notes
The code provided in this repository is written in Python and can be executed on standard Windows/Linux computers.
The program has been designed to work with the provided data files, but you can modify the code to work with different data files if needed.
If the program generates large output files or exceeds the size limit for submission, provide a link to a GitHub repository containing the complete code and outputs.
