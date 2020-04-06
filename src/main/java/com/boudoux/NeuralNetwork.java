package com.boudoux;

import java.util.Date;
import java.util.Random;
import java.util.function.DoubleFunction;

/**
 *
 */
public class NeuralNetwork {

    public static final DoubleFunction<Double> SIGMOID_FUNCTION = input -> 1.0 / (1.0 + Math.pow(Math.E, -1 * input));

    public static final double DEFAULT_LEARNING_RATE = 0.5;

    /**
     * The learning rate. Defaults to 0.5
     */
    private double learningRate = DEFAULT_LEARNING_RATE;

    private Random random = new Random(new Date().getTime());

    /**
     * The threshold function
     */
    private DoubleFunction<Double> thresholdDoubleFunction;

    private int totalInputNodes;

    private int totalHiddenLayers;

    private int totalHiddenNodes;

    private int totalOutputNodes;

    /**
     *
     */
    private double[][] inputLayerLinkWeights;

    /**
     * Dimensions:
     *  1- Layers (totalLayers - 1)
     *  2- Nodes
     *  3- Links (Links (weights) for each node of the next layer)
     */
    private double[][][] hiddenLayerLinkWeights;

    /**
     * This is for the connection between the last hidden layer and the output layer.
     */
    private double[][] hiddenToOutputLinkWeights;

    private boolean initialized;

    // To be called by Deserialization process
    public NeuralNetwork() {}

    public NeuralNetwork(int totalInputNodes, int totalHiddenLayers, int totalHiddenNodes, int totalOutputNodes) {
        this(DEFAULT_LEARNING_RATE, SIGMOID_FUNCTION, totalInputNodes, totalHiddenLayers, totalHiddenNodes,
                totalOutputNodes);
    }

    public NeuralNetwork(double learningRate, DoubleFunction<Double> thresholdDoubleFunction, int totalInputNodes,
                         int totalHiddenLayers, int totalHiddenNodes, int totalOutputNodes) {
        if(totalHiddenLayers < 1)
            throw new IllegalArgumentException("Total layers must be greater than 0");

        if(totalInputNodes < 1 || totalHiddenNodes < 1 || totalOutputNodes < 1)
            throw new IllegalArgumentException("Total nodes (totalInputNodes, totalHiddenNodes, totalOutputNodes) must be greater than 0");

        this.learningRate = learningRate;
        this.thresholdDoubleFunction = thresholdDoubleFunction;
        this.totalInputNodes = totalInputNodes;
        this.totalHiddenLayers = totalHiddenLayers;
        this.totalHiddenNodes = totalHiddenNodes;
        this.totalOutputNodes = totalOutputNodes;

        this.inputLayerLinkWeights = new double[totalInputNodes][totalHiddenNodes];
        this.hiddenLayerLinkWeights = new double[totalHiddenLayers - 1][totalHiddenNodes][totalHiddenNodes];
        this.hiddenToOutputLinkWeights = new double[totalHiddenNodes][totalOutputNodes];

        this.init();

        initialized = true;
    }

    private void init() {
        // just in case of deserialization the value has already been initialized
        if(this.initialized) return;

        // update the weights for the input layer
        for(int idxNode = 0; idxNode < this.totalInputNodes; idxNode++) {
            for(int idxLink = 0; idxLink < this.totalHiddenNodes; idxLink++) {
                this.inputLayerLinkWeights[idxNode][idxLink] = random(this.totalHiddenNodes);
            }
        }

        // updates the weights for the hidden layers
        for(int idxLayer = 0; idxLayer < this.hiddenLayerLinkWeights.length; idxLayer++) {
            for(int idxNode = 0; idxNode < this.totalHiddenNodes; idxNode++) {
                for(int idxLink = 0; idxLink < this.totalHiddenNodes; idxLink++) {
                    this.hiddenLayerLinkWeights[idxLayer][idxNode][idxLink] = random(this.totalHiddenNodes);
                }
            }
        }

        // updates the weights for the output layer
        for(int idxNode = 0; idxNode < this.totalHiddenNodes; idxNode++) {
            for(int idxLink = 0; idxLink < this.totalOutputNodes; idxLink++) {
                this.hiddenToOutputLinkWeights[idxNode][idxLink] = random(this.totalOutputNodes);
            }
        }
    }

    private double random(int totalNodes) {
        // TODO change to return a negative number and maybe +-(1/sqrt(total_nodes))
        return (random.nextInt(99) / 100d + 0.01) * (random.nextInt() % 2 == 0 ? -1 : 1);
    }

    public double[] fire(double[] pInput) {
        if(pInput == null || pInput.length != this.totalInputNodes)
            throw new IllegalArgumentException(String.format("The parameter must have the same length of the total nodes (%d)", totalInputNodes));

        // sum up the result of the product input * weight
        double[] inputHiddenLayer = calculateInputHiddenLayer(pInput);

        // calculates the input for the last hidden layer
        inputHiddenLayer = calculateInputLastHiddenLayer(inputHiddenLayer);

        // calculates the values for the output layer
        double[] output = new double[this.totalOutputNodes];
        for(int idxNode = 0; idxNode < this.totalHiddenNodes; idxNode++) {
            for(int idxLink = 0; idxLink < this.totalOutputNodes; idxLink++) {
                output[idxLink] += inputHiddenLayer[idxNode] * this.hiddenToOutputLinkWeights[idxNode][idxLink];
            }
        }

        // computes the output
        for(int idxNode = 0; idxNode < this.totalOutputNodes; idxNode++)
            output[idxNode] = this.thresholdDoubleFunction.apply(output[idxNode]);

        return output;
    }

    private double[] calculateInputLastHiddenLayer(double[] inputHiddenLayer) {
        double[] currentOutput;
        for(int idxLayer = 0; idxLayer < this.hiddenLayerLinkWeights.length; idxLayer++) {
            currentOutput = new double[this.totalHiddenNodes];
            for(int idxNode = 0; idxNode < this.totalHiddenNodes; idxNode++) {
                for(int idxLink = 0; idxLink < this.totalHiddenNodes; idxLink++) {
                    currentOutput[idxLink] += inputHiddenLayer[idxNode] * this.hiddenLayerLinkWeights[idxLayer][idxNode][idxLink];
                }
            }

            // computes the output
            for(int idxNode = 0; idxNode < this.totalHiddenNodes; idxNode++)
                inputHiddenLayer[idxNode] = this.thresholdDoubleFunction.apply(currentOutput[idxNode]);
        }

        return inputHiddenLayer;
    }

    private double[] calculateInputHiddenLayer(double[] pInput) {
        double[] inputHiddenLayer = new double[this.totalHiddenNodes];
        for(int idxNode = 0; idxNode < this.totalInputNodes; idxNode++) {
            for(int idxLink = 0; idxLink < this.totalHiddenNodes; idxLink++) {
                inputHiddenLayer[idxLink] += pInput[idxNode] * this.inputLayerLinkWeights[idxNode][idxLink];
            }
        }

        // apply the threshold function to the values in the array
        for(int idxNode = 0; idxNode < inputHiddenLayer.length; idxNode++) {
            inputHiddenLayer[idxNode] = this.thresholdDoubleFunction.apply(inputHiddenLayer[idxNode]);
        }

        return inputHiddenLayer;
    }

    /**
     * Trains the Neural Network using the current Input / Expected Target
     *
     * @param input
     * @param expectedTarget
     */
    public void train(double[] input, double[] expectedTarget) {
        // trigger the NN using the provided input
        this.fire(input);

        // takes the resultant output from the output layer
        double gradientDecent = 1.0; // this variable is used in Gradient Decent (increase/decrease)
        double currentOutput;
        double inputFromPreviousNodeLayer;
        double weightFromPreviousNodeLayer;
        double currentTarget;
        double gradient;
        double newWeight;
/*
        for(int idxCurrentOutputLayer = this.totalLayers - 1; idxCurrentOutputLayer >= 1; idxCurrentOutputLayer--) {
            for (int idxNode = 0; idxNode < this.totalNodes; idxNode++) {
                currentOutput = this.outputs[idxCurrentOutputLayer][idxNode]; // output of the current node x layer

                if(idxCurrentOutputLayer == this.totalLayers - 1) {
                    currentTarget = expectedTarget[idxNode]; // training data - the target for the current node
                    gradient = currentTarget - currentOutput;
                } else {
                    currentTarget = 0.0;
                    for(int idxLink = 0; idxLink < this.totalNodes; idxLink++) {
                        currentTarget += this.hiddenLayerLinkWeights[idxCurrentOutputLayer][idxNode][idxLink];
                    }
                    gradient = currentTarget;
                }

                for (int idxLink = 0; idxLink < this.totalNodes; idxLink++) {
                    inputFromPreviousNodeLayer = this.outputs[idxCurrentOutputLayer - 1][idxLink];
                    // link's weight from the previous layer/node to the current node
                    weightFromPreviousNodeLayer = this.hiddenLayerLinkWeights[idxCurrentOutputLayer - 1][idxLink][idxNode];

                    if (currentOutput > currentTarget)
                        gradientDecent *= -1;
                    else
                        gradientDecent = 1;

                    // Formula Gradient Decent: Wn +/- σ * -(En - On) * On * (1 - On) * O(n - 1)
                    newWeight = weightFromPreviousNodeLayer + gradientDecent * (learningFactor * (-1 * gradient *
                            currentOutput * (1 - currentOutput)) * inputFromPreviousNodeLayer);

                    // updates to the new weight
                    this.hiddenLayerLinkWeights[idxCurrentOutputLayer - 1][idxLink][idxNode] = newWeight;
                }
            }
        }
 */
    }

}
