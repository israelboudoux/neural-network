package com.boudoux.util;

import java.util.function.Consumer;

public class Utils {

    public static double[] arr(double... args) {
        return args;
    }

    public static void print(double[] array, Consumer<Double> consumer) {
        for(double value: array) {
            consumer.accept(value);
        }
    }
}
