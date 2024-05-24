# Classical-Quantum (or Hybrid) Neural Network with Adversarial Training Defense

This repository contains an implementation of a Classical-Quantum (or Hybrid) Neural Network (HNN) that combines a Quantum Neural Network (QNN) and a Convolutional Neural Network (CNN) for digit recognition on handwritten digits (e.g., MNIST, EMNIST Digits, etc.) datasets. The model is protected against compounded adversarial attacks using adversarial training.

## Motivating Article and Related Work
West, Maxwell T, et al. “Benchmarking Adversarially Robust Quantum Machine Learning at Scale.” Physical Review Research, vol. 5, no. 2, 23 June 2023, doi: 10.1103/physrevresearch.5.023186.

TorchAttacks
https://adversarial-attacks-pytorch.readthedocs.io/en/latest/

PyTorch Adversarial Example Generation
https://pytorch.org/tutorials/beginner/fgsm_tutorial.html

Adversarial-Attacks-PyTorch
https://github.com/Harry24k/adversarial-attacks-pytorch

## Results

### Adversarial Training Defense Mechanism for HNN Model with MNIST Dataset

|                  |                  |                  |                    |
|------------------|------------------|------------------|--------------------|
| **Compounded Attack**   | **Pre-Attack Accuracy** | **Post Attack Accuracy** | **Post Attack Defense Applied Accuracy** |
|                   | **MNIST: Original** | **MNIST: Adversarial Examples** | **MNIST: Combined (Original + Adversarial Examples)** |
| FGSM + CW        | 98.0%           | 20.0%           | 100.0%             |
| FGSM + PGD       | 98.0%           | 20.0%           | 98.0%             |
| CW + PGD         | 100.0%           | 89.0%           | 100.0%             |


## Requirements

To run the code, ensure you have the following dependencies installed:

- Python version: 3.8.18
- torch: 2.2.1
- torchvision: 0.17.1
- torchattacks: 3.5.1
- numpy: 1.23.5
- tabulate: 0.9.0
- cirq: 1.3.0

You can install the required packages using pip:

```
pip install torch torchvision torchattacks numpy tabulate cirq
```

## Dataset

The code assumes that the datasets are stored on Google Drive and that Google Drive will be mounted with the default directory structure. Ensure that you have the necessary datasets (MNIST, EMNIST, SVHN, USPS, or Semeion) in the appropriate locations on your Google Drive.

## Model Architecture

### Convolutional Neural Network (CNN)

The CNN architecture is designed to recognize digits 0 to 9 in the MNIST-type datasets. It consists of the following layers:

- Two convolutional layers with ReLU activation and max pooling
- Two fully connected layers with ReLU activation
- Output layer with log softmax activation

The CNN learns to extract relevant features and patterns from the input images for digit recognition.

### Hybrid Neural Network (HNN)

The HNN combines the classical CNN with a quantum circuit to enhance the model's performance. The HNN takes the classical model as an input parameter and integrates it with the quantum circuit.

The HNN initializes trainable parameters (`theta` and `phi`) for the quantum circuit, which represent the angles of rotation gates. These parameters introduce additional non-linearity and expressiveness to the model.

During the forward pass, the input data is passed through the classical CNN, and the extracted features are used as input to the quantum circuit. The quantum circuit applies quantum operations based on the learned parameters to transform the input features. The output of the quantum circuit is then processed and combined with the classical model's predictions to produce the final output.

### Quantum Circuit

The quantum circuit is implemented using the Cirq library. It consists of the following components:

- Rotation gates (`cirq.ry`) applied to each qubit, alternating between the angles `theta` and `phi`
- Entangling gates (`cirq.CNOT`) applied between pairs of adjacent qubits

The rotation gates introduce single-qubit operations that can manipulate the state of individual qubits, while the entangling gates create correlations between the states of different qubits.

The number of qubits in the quantum circuit is determined based on the output dimension of the model. The circuit is created dynamically based on the learned parameters `theta` and `phi`.

[![hnn_quantum_circuit](https://github.com/ericyoc/adversarial-defense-hnn/blob/main/qnn_circuit/qnn_circuit.jpg?raw=true)](https://github.com/ericyoc/adversarial-defense-hnn/blob/main/qnn_circuit/qnn_circuit.jpg)

## Adversarial Attacks

The code implements compounded white-box targeted adversarial attacks using the TorchAttacks library. The available attack options include:

- FGSM + CW attack
- FGSM + PGD attack
- CW + PGD attack
- PGD + BIM attack
- FGSM + BIM attack
- CW + BIM attack
- FGSM + DeepFool attack
- PGD + DeepFool attack
- CW + DeepFool attack
- BIM + DeepFool attack

These attacks are designed to generate adversarial examples that fool the model into making incorrect predictions. The attacks are performed in a white-box setting, where the attacker has full knowledge of the model architecture and parameters. The targeted nature of the attacks means that the adversarial examples are crafted to cause the model to misclassify the input as a specific target class.

## Adversarial Training Defense

The code includes an implementation of adversarial training as a defense mechanism against adversarial attacks. Adversarial training involves the following steps:

1. Generate adversarial examples using the specified compounded attack on the clean training data.
2. Combine the clean training data with the generated adversarial examples to create an augmented training dataset.
3. Retrain the model on the augmented training dataset, allowing it to learn to correctly classify both clean and adversarial examples.

By exposing the model to adversarial examples during training, adversarial training helps improve the model's robustness and resilience against adversarial attacks.

## Evaluation

The code evaluates the model's performance on clean data, under adversarial attacks without defense, and under adversarial attacks with defense. It reports metrics such as loss, accuracy, precision, recall, F1-score, and ROC AUC score. It also provides visualizations of misclassified examples for each scenario.

## Results

The code summarizes the model's performance in a tabular format and displays example misclassifications for each scenario (clean, no defense attack, and with defense attack). The results provide insights into the effectiveness of the adversarial training defense against the specified compounded white-box targeted attack [HNN Results](https://github.com/ericyoc/adversarial-defense-hnn/tree/main/hnn_results)).

## Acknowledgments

This code makes use of the following libraries and frameworks:

- PyTorch
- TorchAttacks
- Cirq
- NumPy
- Tabulate

We would like to acknowledge the creators and contributors of these libraries for their valuable work.

## License

This project is licensed under the [MIT License](LICENSE).

**Disclaimer**
This repository is intended for educational and research purposes.

