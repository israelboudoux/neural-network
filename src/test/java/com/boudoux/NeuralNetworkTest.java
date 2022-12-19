package com.boudoux;

import com.boudoux.util.Utils;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

public class NeuralNetworkTest {

    @Test
    public void _1inputNodes_noHiddenLayers_2outputNodes() {
        NeuralNetwork nn = new NeuralNetwork(1, new int[0][0], 2);

        double[] output = nn.fire(Utils.arr(0.03d));

        Utils.print(output, System.out::println);
    }

    @Test
    public void _2inputNodes_noHiddenLayers_1outputNodes() {
        NeuralNetwork nn = new NeuralNetwork(2, new int[0][0],1);

        double[] output = nn.fire(Utils.arr(0.03, 0.8));

        Utils.print(output, System.out::println);
    }

    @Test
    public void _3inputNodes_noHiddenLayers_3outputNodes() {
        NeuralNetwork nn = new NeuralNetwork(3, new int[0][0], 3);

        double[] output = nn.fire(Utils.arr(-0.1d, -0.2d, 0.03d));

        Utils.print(output, System.out::println);
    }

    @Test
    public void _3inputNodes_3hiddenNodes_3outputNodes() {
        NeuralNetwork nn = new NeuralNetwork(3, new int[][]{{3}}, 3);

        double[] output = nn.fire(Utils.arr(-0.1d, -0.2d, 0.03d));

        Utils.print(output, System.out::println);
    }

    @Test
    public void _3inputNodes_5hiddenNodes_1outputNodes() {
        NeuralNetwork nn = new NeuralNetwork(3, new int[][]{{5}}, 1);

        double[] output = nn.fire(Utils.arr(0.1d, 0.2d, 0.55d));

        Utils.print(output, System.out::println);
    }

    @Test
    public void _5inputNodes_10hiddenNodes_15outputNodes() {
        NeuralNetwork nn = new NeuralNetwork(5, new int[][]{{10}}, 15);

        double[] output = nn.fire(Utils.arr(0.1d, 0.2d, 0.55d, 0.1d, 0.2d));

        Utils.print(output, System.out::println);
    }

    @Test
    public void _3inputNodes_3hiddenNodes_3outputNodes_with_weights() {
        NeuralNetwork nn = new NeuralNetwork(3, new int[][]{{3}}, 3,
                new double[][][]{{{0.9, 0.2, 0.1}, {0.3, 0.8, 0.5}, {0.4, 0.2, 0.6}},
                                {{0.3, 0.6, 0.8}, {0.7, 0.5, 0.1}, {0.5, 0.2, 0.9}}});

        double[] output = nn.fire(Utils.arr(0.9, 0.1, 0.8));

        Assert.assertTrue(output[0] >= 0.726 && output[0] < 0.727);
        Assert.assertTrue(output[1] >= 0.708 && output[1] < 0.709);
        Assert.assertTrue(output[2] >= 0.778 && output[2] < 0.779);

        Utils.print(output, System.out::println);
    }

    @Test
    public void _2inputNodes_1outputNode_AND_gate() {
        // AND logic gate
        NeuralNetwork nn = new NeuralNetwork(2, new int[0][], 1,
                new double[][][]{{{0.99}, {0.99}}});

        // true AND true
        double[] output = nn.fire(Utils.arr(0.99, 0.99));
        Assert.assertTrue(output[0] >= 0.87);

        // true AND false
        output = nn.fire(Utils.arr(0.99, 0.01));
        Assert.assertTrue(output[0] >= 0.729 && output[0] < 0.7299);

        // false AND false
        output = nn.fire(Utils.arr(0.01, 0.01));
        Assert.assertTrue(output[0] <= 0.50494983829);

        // false AND true
        output = nn.fire(Utils.arr(0.01, 0.99));
        Assert.assertTrue(output[0] >= 0.729 && output[0] < 0.7299);

        Utils.print(output, System.out::println);
    }

    @Test
    public void andGate_trainer() {
        NeuralNetwork nn = new NeuralNetwork(2, new int[0][], 1);

        for(int counter = 1; counter <= 10_000; counter++) {
            // true AND true
            nn.train(Utils.arr(0.99, 0.99), Utils.arr(0.876554595392));

            // true AND false
            nn.train(Utils.arr(0.99, 0.01), Utils.arr(0.729087922349));

            // false AND false
            nn.train(Utils.arr(0.01, 0.01), Utils.arr(0.50494983829));

            // false AND true
            nn.train(Utils.arr(0.01, 0.99), Utils.arr(0.729087922349));
        }

        System.out.println(nn);
    }

    @Test
    public void _2inputNodes_1outputNode_OR_gate() {
        // AND logic gate
        NeuralNetwork nn = new NeuralNetwork(2, new int[0][], 1,
                new double[][][]{{{0.99}, {0.99}}});

        // true OR true
        double[] output = nn.fire(Utils.arr(0.99, 0.99));
        Assert.assertTrue(output[0] >= 0.729);

        // true OR false
        output = nn.fire(Utils.arr(0.99, 0.01));
        Assert.assertTrue(output[0] >= 0.729);

        // false OR false
        output = nn.fire(Utils.arr(0.01, 0.01));
        Assert.assertTrue(output[0] < 0.51);

        // false OR true
        output = nn.fire(Utils.arr(0.01, 0.99));
        Assert.assertTrue(output[0] >= 0.729);

        Utils.print(output, System.out::println);
    }

    @Test
    public void _5inputNodes_manyHiddenLayersManyNodes_3outputNodes() {
        NeuralNetwork nn = new NeuralNetwork(5, new int[][]{{6}, {7}, {7}, {6}}, 3);

        double[] output = nn.fire(Utils.arr(0.1d, 0.2d, 0.55d, 0.1d, 0.2d));

        Utils.print(output, System.out::println);
    }

    @Test
    public void _train_2layers_5inputNodes_3outputNodes() {
        NeuralNetwork nn = new NeuralNetwork(5, new int[0][], 3);

        nn.train(Utils.arr(0.1d, 0.2d, 0.55d, 0.1d, 0.2d), Utils.arr(0.1d, 0.2d, 0.55d));
    }

    @Test
    public void _train_6layers_5inputNodes_4hiddenLayersManyNodes_3outputNodes() {
        NeuralNetwork nn = new NeuralNetwork(5, new int[][]{{6}, {7}, {7}, {6}}, 3);

        nn.train(Utils.arr(0.1d, 0.2d, 0.55d, 0.1d, 0.2d), Utils.arr(0.1d, 0.2d, 0.55d, 0.1d, 0.2d));
    }

    @Test
    public void _train_4layers_2inputNodes_2hiddenLayers3Nodes2Nodes_1outputNodes() {
        NeuralNetwork nn = new NeuralNetwork(2, new int[][]{{3}, {2}}, 1,
                new double[][][]{{{0.9, 0.2, 0.1}, {0.3, 0.8, 0.5}},
                        {{0.3, 0.6}, {0.7, 0.5}, {0.2, 0.9}},
                        {{0.09}, {0.65}}});

        nn.train(Utils.arr(0.2, 0.05), Utils.arr(0.2));

        System.out.println(nn);
    }

    @Test
    public void _train_3layers_3input_1hiddenLayer3Nodes_3output() {
        NeuralNetwork nn = new NeuralNetwork(0.3, 3, new int[][]{{3}}, 3);

        for(int count = 1; count <= 15_000; count++) {
            nn.train(Utils.arr(0.9, 0.1, 0.8), Utils.arr(0.726, 0.708, 0.778));
        }

        double[] output = nn.fire(Utils.arr(0.9, 0.1, 0.8));

        Assert.assertTrue(String.valueOf(output[0]), output[0] >= 0.725 && output[0] < 0.727);
        Assert.assertTrue(String.valueOf(output[1]), output[1] >= 0.707 && output[1] < 0.709);
        Assert.assertTrue(String.valueOf(output[2]), output[2] >= 0.770 && output[2] < 0.779);
    }
}
