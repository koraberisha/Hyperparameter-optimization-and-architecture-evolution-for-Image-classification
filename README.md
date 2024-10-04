# Hyperparameter Optimization and Architecture Evolution in CNNs for Image Classification

This repository contains the implementation and research paper for a project on evolving convolutional neural network (CNN) architectures and hyperparameters using genetic algorithms, focused on image classification tasks.

## Paper

The paper [Hyperparameter optimization and architecture evolution in Convolutional Neural Networks for Image classification](Neuroevolution-2021-05-28-21-25.pdf) presents a novel approach to autonomously generating effective CNN architectures. Key aspects include:

- Utilizing genetic algorithms to evolve CNN architecture-parameter combinations
- Evaluating performance on CIFAR-10 and CIFAR-100 datasets
- Encoding CNN structures and hyperparameters into binary chromosomes
- Analyzing the impact of different CNN components on network efficacy
- Balancing performance optimization with computational constraints

## Implementation

The core components of the system are:

- `GA.py`: Implements the genetic algorithm, including:
  - Population initialization
  - Fitness evaluation
  - Selection (tournament-based)
  - Crossover (single-point)
  - Mutation
  
- `tensorTest.py`: Handles CNN-related operations:
  - Dynamic model generation based on chromosome encoding
  - Model compilation and training
  - Performance evaluation

### Key Features

- Flexible CNN architecture generation using a base feature extraction block
- Binary encoding scheme for hyperparameters and network structure
- Fitness evaluation using validation accuracy
- Adaptive network depth and structure based on genetic encoding

## Methodology

1. Initialize a population of binary-encoded CNN configurations
2. Generate and train CNNs based on each chromosome
3. Evaluate fitness using validation accuracy
4. Select best-performing individuals
5. Apply genetic operators (crossover, mutation) to create next generation
6. Repeat for specified number of generations

## Results

The project demonstrates the potential of genetic algorithms in CNN optimization:

- Performance improved over generations, with later generations showing more consistent results
- Best evolved architecture achieved ~78% accuracy on CIFAR-10 and ~45% on CIFAR-100
- Analysis of evolved architectures provides insights into effective CNN designs
- Early-epoch performance showed correlation with final model efficacy

## Limitations and Future Work

- Computational constraints limited the population size and number of generations
- Potential for improvement with distributed or parallel computing systems
- Opportunity to expand the parameter space and apply to more complex datasets
- Further investigation into the relationship between early-epoch performance and final model accuracy

## Usage

To run the genetic algorithm optimization:
