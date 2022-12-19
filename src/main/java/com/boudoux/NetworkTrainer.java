package com.boudoux;

import com.boudoux.util.Utils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

public class NetworkTrainer {
    public static void main(String[] args) throws IOException {
        NeuralNetwork neuralNetwork = new NeuralNetwork(0.3, 784, new int[][] {{100}}, 10);

        List<String> fileLines = Files.readAllLines(Paths.get(NetworkTrainer.class.getClassLoader().getResource("mnist_train_100.csv").getPath()));

        boolean debug = false;

        for(int idx0 = 1; idx0 <= 1_000; idx0++) {
            System.out.println("Training set #" + idx0);
            fileLines.stream().forEach(l -> {
                double[] input = new double[784];
                String[] arrayInput = l.split(",");
                for (int idx = 1; idx < arrayInput.length; idx++) {
                    input[idx - 1] = Double.parseDouble(arrayInput[idx]) / 255.0 * 0.99 + 0.01;
                }

                double[] expectedTarget = new double[10];
                for (int idx = 0; idx < 10; idx++) {
                    expectedTarget[idx] = 0.01;
                }

                expectedTarget[Integer.parseInt(arrayInput[0])] = 0.99;

                neuralNetwork.train(input, expectedTarget);

                if(debug) {
                    double[] result = neuralNetwork.fire(input);
                    System.out.println("Result: " + arrayInput[0]);
                    Utils.print(result, System.out::println);
                    System.out.println();
                }
            });
        }

        fileLines = Files.readAllLines(Paths.get(NetworkTrainer.class.getClassLoader().getResource("mnist_test_10.csv").getPath()));
        fileLines.stream().forEach(l -> {
            double[] input = new double[784];
            String[] arrayInput = l.split(",");
            for(int idx = 1; idx < arrayInput.length; idx++) {
                input[idx - 1] = (Double.parseDouble(arrayInput[idx]) / 255.0) * 0.99 + 0.01;
            }

            int correctResult = Integer.parseInt(arrayInput[0]);

            double[] result = neuralNetwork.fire(input);

            System.out.println(String.format("Expected result: %d", correctResult));
            Utils.print(result, System.out::println);
        });

        System.out.println();
        double[] input = new double[784];
        Arrays.fill(input, 0.01);
        double[] result = neuralNetwork.fire(input);

        Utils.print(result, System.out::println);
    }

    private static void serialize() {

    }

    private static void deserialize() {

    }
}
