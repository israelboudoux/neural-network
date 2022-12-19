package com.boudoux;

import com.boudoux.util.Utils;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Date;
import java.util.Random;
import java.util.function.DoubleFunction;

/**
 *
 */
public class NeuralNetwork implements Serializable {

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

    // The quantity of nodes defined for each layer
    private int[][] nodesByHiddenLayer;

    private int totalOutputNodes;

    private int totalLayers;

    /**
     * Dimensions:
     *  1- Layers (totalLayers - 1)
     *  2- Nodes
     *  3- Links (Links (weights) for each node of the next layer)
     */
    private double[][][] layerLinkWeights;

    // Stores the output for all layers
    private double[][] layerOutputs;

    private boolean initialized;

    public NeuralNetwork(int totalInputNodes, int[][] nodesByHiddenLayer, int totalOutputNodes) {
        this(DEFAULT_LEARNING_RATE, SIGMOID_FUNCTION, totalInputNodes, nodesByHiddenLayer, totalOutputNodes, null);
    }

    public NeuralNetwork(int totalInputNodes, int[][] nodesByHiddenLayer, int totalOutputNodes, double[][][] preDefinedWeights) {
        this(DEFAULT_LEARNING_RATE, SIGMOID_FUNCTION, totalInputNodes, nodesByHiddenLayer, totalOutputNodes, preDefinedWeights);
    }

    public NeuralNetwork(double learningRate, int totalInputNodes, int[][] nodesByHiddenLayer, int totalOutputNodes) {
        this(learningRate, SIGMOID_FUNCTION, totalInputNodes, nodesByHiddenLayer, totalOutputNodes, null);
    }

    public NeuralNetwork(double learningRate, DoubleFunction<Double> thresholdDoubleFunction, int totalInputNodes,
                         int[][] nodesByHiddenLayer, int totalOutputNodes, double[][][] preDefinedWeights) {

        this.validate(learningRate, thresholdDoubleFunction, totalInputNodes, nodesByHiddenLayer, totalOutputNodes, preDefinedWeights);

        this.learningRate = learningRate;
        this.thresholdDoubleFunction = thresholdDoubleFunction;

        this.totalInputNodes = totalInputNodes;
        this.nodesByHiddenLayer = nodesByHiddenLayer;
        this.totalOutputNodes = totalOutputNodes;

        // input_layer + hidden_layers
        this.totalLayers = nodesByHiddenLayer.length + 1;

        if(preDefinedWeights == null) {
            this.setupLayers();

            this.init();
        } else {
            this.layerLinkWeights = preDefinedWeights;
        }

        initialized = true;
    }

    // does consistency validation
    private void validate(double learningRate, DoubleFunction<Double> thresholdDoubleFunction, int totalInputNodes,
                          int[][] nodesByHiddenLayer, int totalOutputNodes, double[][][] preDefinedWeights) {

    }

    private void setupLayers() {
        if(this.initialized) return;

        this.layerLinkWeights = new double[this.totalLayers][][];

        // does the setup for the first layer (named Input Layer)
        // More than 2 layers means we have Hidden Layers
        if(this.totalLayers > 1) {
            this.layerLinkWeights[0] = new double[this.totalInputNodes][this.nodesByHiddenLayer[0][0]];
        } else {
            // Otherwise we only have Input and Output Layers
            this.layerLinkWeights[0] = new double[this.totalInputNodes][this.totalOutputNodes];
        }

        // We setup for the Hidden Layers, except the last Hidden Layer
        for(int idxLayer = 1; idxLayer < this.layerLinkWeights.length - 1; idxLayer++) {
            this.layerLinkWeights[idxLayer] =
                    new double[this.nodesByHiddenLayer[idxLayer - 1][0]][this.nodesByHiddenLayer[idxLayer][0]];
        }

        if(this.nodesByHiddenLayer.length > 0) {
            // Defines the connection between the last Hidden Layer and the Output Layer
            this.layerLinkWeights[this.layerLinkWeights.length - 1] =
                    new double[this.nodesByHiddenLayer[this.nodesByHiddenLayer.length - 1][0]][this.totalOutputNodes];
        }
    }

    private void init() {
        // just in case of deserialization the value has already been initialized
        if(this.initialized) return;

        // update the weights for each layer
        for(int idxLayer = 0; idxLayer < this.layerLinkWeights.length; idxLayer++) {
            for(int idxNode = 0; idxNode < this.layerLinkWeights[idxLayer].length; idxNode++) {
                for(int idxLink = 0; idxLink < this.layerLinkWeights[idxLayer][idxNode].length; idxLink++) {
                    this.layerLinkWeights[idxLayer][idxNode][idxLink] = random(this.layerLinkWeights[idxLayer][idxNode].length);
                }
            }
        }
    }

    public double[] fire(double[] pInput) {
        if(pInput == null || pInput.length != this.totalInputNodes)
            throw new IllegalArgumentException(String.format("The parameter must have the same length of the total nodes (%d)", totalInputNodes));

        // holds the outputs for each layer (input_layer + hidden_layers + output_layer)
        this.layerOutputs = new double[this.layerLinkWeights.length + 1][];

        // stores the output for the Input Layer
        this.layerOutputs[0] = Arrays.copyOf(pInput, pInput.length);

        // aux array to hold the Inputs for the current layer
        double[] inputOutput = Arrays.copyOf(pInput, pInput.length);

        for(int idxLayer = 0, idxLayerOutput = 1; idxLayer < this.layerLinkWeights.length; idxLayer++, idxLayerOutput++) {
            // creates the array for the current layer with the length of Links
            this.layerOutputs[idxLayerOutput] = new double[this.layerLinkWeights[idxLayer][0].length];

            // sum up all the LINK_WEIGHT * INPUT_NODE
            for(int idxNode = 0; idxNode < this.layerLinkWeights[idxLayer].length; idxNode++) {
                for(int idxLink = 0; idxLink < this.layerLinkWeights[idxLayer][idxNode].length; idxLink++) {
                    this.layerOutputs[idxLayerOutput][idxLink] += this.layerLinkWeights[idxLayer][idxNode][idxLink] * inputOutput[idxNode];
                }
            }

            // applies the threshold function
            for(int idxNode = 0; idxNode < this.layerOutputs[idxLayerOutput].length; idxNode++) {
                this.layerOutputs[idxLayerOutput][idxNode] = this.thresholdDoubleFunction.apply(this.layerOutputs[idxLayerOutput][idxNode]);
            }

            // updates the array with the Input Values from the current Hidden Layer to be used by the next one
            inputOutput = Arrays.copyOf(this.layerOutputs[idxLayerOutput], this.layerOutputs[idxLayerOutput].length);
        }

        return inputOutput;
    }

    /**
     * Trains the Neural Network using the Backpropagation approach.
     *
     * @param input
     * @param expectedTarget
     */
    public void train(double[] input, double[] expectedTarget) {

        // trigger the NN using the provided input
        this.fire(input);

        // takes the resultant output from the output layer
        double currentOutput;
        double inputFromNodePreviousLayer;
        double linkWeightFromPreviousLayer;
        double currentTarget;
        double errorGradient;
        double newWeight;
        // stores the errors for the nodes in the previous layer
        double[][] previousNodeError = new double[this.layerLinkWeights.length][];

        // starting from the last to the before first layer
        for(int idxLayer = this.layerOutputs.length - 1, idxLayerLinkWeight = idxLayer - 1; idxLayer > 0; idxLayer--, idxLayerLinkWeight--) {
            previousNodeError[idxLayerLinkWeight] = new double[this.layerLinkWeights[idxLayerLinkWeight].length];

            for (int idxNode = 0; idxNode < this.layerOutputs[idxLayer].length; idxNode++) {
                currentOutput = this.layerOutputs[idxLayer][idxNode]; // output of the current node

                // the first iteration we go for the Output Layer
                if(idxLayer == this.layerOutputs.length - 1) {
                    currentTarget = expectedTarget[idxNode]; // training data - the target for the current node
                    errorGradient = currentTarget - currentOutput;
                } else { // So for the others layers
                    // the errorGradient is the sum of the link's weights's errors from the next layer for the current node
                    errorGradient = previousNodeError[idxLayerLinkWeight + 1][idxNode];
                }

                // filling the array with errors of Nodes from Previous Layer to the Current Layer
                for(int idxNodePrevLayer = 0; idxNodePrevLayer < previousNodeError[idxLayerLinkWeight].length; idxNodePrevLayer++) {
                    previousNodeError[idxLayerLinkWeight][idxNodePrevLayer] +=
                            this.layerLinkWeights[idxLayerLinkWeight][idxNodePrevLayer][idxNode] * errorGradient;
                }

                double error = this.learningRate * (errorGradient * currentOutput * (1.0 - currentOutput));

                for(int idxLink = 0; idxLink < this.layerLinkWeights[idxLayerLinkWeight].length; idxLink++) {
                    // input's value from the previous layer/node to the current node
                    inputFromNodePreviousLayer = this.layerOutputs[idxLayer - 1][idxLink];
                    // link's weight from the previous layer/node to the current node
                    linkWeightFromPreviousLayer = this.layerLinkWeights[idxLayerLinkWeight][idxLink][idxNode];

                    // Formula Gradient Decent: Wn +/- σ * -(En - On) * On * (1 - On) * O(n - 1)
                    double deltaWeight = error * inputFromNodePreviousLayer;
                    newWeight = linkWeightFromPreviousLayer + deltaWeight;

                    // updates the link's weight which points to the current node
                    this.layerLinkWeights[idxLayerLinkWeight][idxLink][idxNode] = newWeight;
                }
            }
        }
    }

    public double random(int totalNodes) {
        //return (random.nextInt(99) / 100d + 0.01) * (random.nextInt() % 2 == 0 ? -1 : 1);

        double calc = 1.0 / Math.sqrt(totalNodes);
        int multiplier = 1;

        while ((int) (calc * multiplier) / 10 < 1) {
            multiplier *= 10;
        }

        // +-(1/sqrt(total_nodes))
        return (random.nextInt((int) (calc * multiplier)) / (multiplier * 1.0) + 0.01) * (random.nextInt() % 2 == 0 ? -1 : 1);
    }

    public double[] getOutput() {
        if(layerOutputs == null || layerOutputs.length == 0)
            return null;

        return layerOutputs[layerOutputs.length - 1];
    }

    @Override
    public String toString() {
        return "NeuralNetwork{" +
                "learningRate=" + learningRate +
                ", random=" + random +
                ", thresholdDoubleFunction=" + thresholdDoubleFunction +
                ", totalInputNodes=" + totalInputNodes +
                ", nodesByHiddenLayer=" + Arrays.toString(nodesByHiddenLayer) +
                ", totalOutputNodes=" + totalOutputNodes +
                ", totalLayers=" + totalLayers +
                ", layerLinkWeights=" + Arrays.toString(layerLinkWeights) +
                ", layerOutputs=" + Arrays.toString(layerOutputs) +
                ", initialized=" + initialized +
                '}';
    }
}
