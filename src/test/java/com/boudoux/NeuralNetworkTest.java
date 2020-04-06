package com.boudoux;

import com.boudoux.util.Utils;
import org.junit.Test;

public class NeuralNetworkTest {

    @Test
    public void _3inputNodes_3hiddenNodes_3outputNodes() {
        NeuralNetwork nn = new NeuralNetwork(3, 1, 3, 3);

        double[] output = nn.fire(Utils.arr(-0.1d, -0.2d, 0.03d));

        Utils.print(output, System.out::println);
    }

    @Test
    public void _3inputNodes_5hiddenNodes_1outputNodes() {
        NeuralNetwork nn = new NeuralNetwork(3, 3, 5, 1);

        double[] output = nn.fire(Utils.arr(0.1d, 0.2d, 0.55d));

        Utils.print(output, System.out::println);
    }

    @Test
    public void _5inputNodes_10hiddenNodes_15outputNodes() {
        NeuralNetwork nn = new NeuralNetwork(5, 2, 10, 15);

        double[] output = nn.fire(Utils.arr(0.1d, 0.2d, 0.55d, 0.1d, 0.2d));

        Utils.print(output, System.out::println);
    }
        /*
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
            nn.train(arr(0.9, 0.1, 0.8), arr(0.2, 0.3, 0.4));

            System.out.println(nn);
        }
         */

}
