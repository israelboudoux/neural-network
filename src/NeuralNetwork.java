import java.util.Date;
import java.util.Random;
import java.util.function.DoubleFunction;

/**
 *
 */
public class NeuralNetwork {

    public static final DoubleFunction<Double> SIGMOID_DoubleFunction = (input) -> 1.0 / (1.0 + Math.pow(Math.E, -1 * input));

    private DoubleFunction<Double> thresholdDoubleFunction;

    private int totalLayers;

    private int totalNodes;

    /**
     * Dimensions:
     *  1- Layers (totalLayers - 1)
     *  2- Nodes
     *  3- Links (Links (weights) for each node of the next layer)
     */
    private double[][][] linkWeights;

    /**
     * Contains Outputs from each Layer x Node
     */
    private double[][] outputs;

    private boolean initialized;

    // To be called by Deserialization process
    public NeuralNetwork() {}

    public NeuralNetwork(int totalLayers, int totalNodes) {
        this(totalLayers, totalNodes, SIGMOID_DoubleFunction, null);
    }

    public NeuralNetwork(int totalLayers, int totalNodes, double[][][] linkWeights) {
        this(totalLayers, totalNodes, SIGMOID_DoubleFunction, linkWeights);
    }

    public NeuralNetwork(int totalLayers, int totalNodes, DoubleFunction<Double> thresholdDoubleFunction, double[][][] linkWeights) {
        if(totalLayers < 2)
            throw new IllegalArgumentException("Total layers must be greater than 1 (Min 2 layers: Input Layer + Output Layer)");

        if(totalNodes < 1)
            throw new IllegalArgumentException("Total layers must be greater than 0");

        if(linkWeights != null
                && (linkWeights.length != totalLayers - 1 || linkWeights[0].length != totalNodes || linkWeights[0][0].length != totalNodes))
            throw new IllegalArgumentException("The linkWeighs must be [totalLayers][totalNodes][totalNodes]");

        this.thresholdDoubleFunction = thresholdDoubleFunction;
        this.totalLayers = totalLayers;
        this.totalNodes = totalNodes;

        this.outputs = new double[totalLayers][totalNodes];

        if(linkWeights == null) {
            this.linkWeights = new double[totalLayers][totalNodes][totalNodes];
            init();
        } else {
            this.linkWeights = linkWeights;
        }

        initialized = true;
    }

    private void init() {
        // just in case of deserialization the value has already been initialized
        if(this.initialized) return;

        Random random = new Random(new Date().getTime());
        // updates the node's weight for all layer's, except the first and the last ones, to a random value between 0.01 and 0.99
        for(int idxLayer = 0; idxLayer < totalLayers; idxLayer++) {
            for(int idxNode = 0; idxNode < totalNodes; idxNode++) {
                for(int idxLink = 0; idxLink < totalLayers; idxLink++) {
                    this.linkWeights[idxLayer][idxNode][idxLink] = random.nextInt(99) / 100d + 0.01;
                }
            }
        }
    }

    public void fire(double[] pInput) {
        if(pInput == null || pInput.length != this.totalNodes)
            throw new IllegalArgumentException(String.format("The parameter must have the same length of the total nodes (%d)", totalNodes));

        // Sets the input values for the first layer
        for(int idx = 0; idx < this.totalNodes; idx++)
            this.outputs[0][idx] = pInput[idx];

        double[] currentOutput;
        for(int idxLayer = 1; idxLayer < this.totalLayers; idxLayer++) {
            currentOutput = new double[this.totalNodes];
            for(int idxNode = 0; idxNode < this.totalNodes; idxNode++) {
                for(int idxLink = 0; idxLink < this.totalNodes; idxLink++) {
                    currentOutput[idxLink] += this.outputs[idxLayer - 1][idxNode] * this.linkWeights[idxLayer - 1][idxNode][idxLink];
                }
            }

            // Copies the currentOutput result to the input ref to the current layer
            for(int idx = 0; idx < this.totalNodes; idx++)
                this.outputs[idxLayer][idx] = this.thresholdDoubleFunction.apply(currentOutput[idx]);
        }
    }

    /**
     * Trains the Neural Network using the current Input / Expected Target
     *
     * @param input
     * @param expectedTarget
     */
    public void train(double[] input, double[] expectedTarget) {

    }

    public double[][] getOutputs() {
        return outputs;
    }

    public static double[] arr(double... args) {
        return args;
    }

    public static void main(String[] args) {
        {
            NeuralNetwork nn = new NeuralNetwork(3, 3);

            nn.fire(arr(0.1d, 0.2d, 0.55d));

            System.out.println(nn);
        }
        {
            NeuralNetwork nn = new NeuralNetwork(2, 2, new double[][][]{{{0.9, 0.2}, {0.3, 0.8}}});

            nn.fire(arr(1.0, 0.5));

            System.out.println(nn);
        }
        {
            NeuralNetwork nn = new NeuralNetwork(3, 3,
                    new double[][][]{{{0.9, 0.2, 0.1}, {0.3, 0.8, 0.5}, {0.4, 0.2, 0.6}},
                                     {{0.3, 0.6, 0.8}, {0.7, 0.5, 0.1}, {0.5, 0.2, 0.9}}});

            nn.fire(arr(0.9, 0.1, 0.8));

            System.out.println(nn);
        }
    }
}
