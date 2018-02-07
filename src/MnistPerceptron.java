import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Random;
import java.util.WeakHashMap;

class NoOutputException extends Exception {
    public NoOutputException(String msg) {
        super(msg);
    }
}

public class MnistPerceptron {
    static int TRAIN_NUM_IMAGES = 60000;
    static int TEST_NUM_IMAGES = 10000;
    static int IMAGES_RECOGNIZED_TRAIN = 0;
    static int IMAGES_SEEN_TRAIN = 0;
    static int IMAGES_RECOGNIZED_TEST = 0;
    static int IMAGES_SEEN_TEST = 0;
    static double ETA = 0.01;

    static DecimalFormat df = new DecimalFormat("0.000");

    public static double[][] initializeWeights(double[][] weights) {
        double randWeight;
        Random rand = new Random();
        for (int i = 0; i < 785; i++) {
            for (int j = 0; j < 10; j++) {
                randWeight = -0.5 + (0.5 - (-0.5)) * rand.nextDouble();
                weights[i][j] = randWeight;
            }
        }
        return weights;
    }

    public static int[] initializeLabels(int[] labels, String filename) {
        try {
            File file = new File(filename);
            FileReader fileReader = new FileReader(file);
            BufferedReader br = new BufferedReader(fileReader);
            String line;

            for (int i=0; i<labels.length; i++) {
                line = br.readLine();
//                labels[i] = Double.parseDouble(line);
                labels[i] = Integer.parseInt(line);
            }
            return labels;
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("This shouldn't run...");
        return null;
    }

    public static double[] getAndNormizeLine(BufferedReader br, double[] input) {
        try {
            String line = br.readLine();
            String[] str = line.split(",");

            int i = 0;

            for (String s : str) {
                if (i==0) {
                    input[i++] = Double.parseDouble(s);
                } else {
                    input[i++] = Double.parseDouble(s) / (double) 255;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return input;
    }

    public static double[] initializeInputVector(double[] input, double[] data) {
        // Input to bias is always 1
        input[0] = 1;
        // Input to remaining elements matches that of data array
        for (int i = 1; i < 785; i++) {
            input[i] = data[i - 1];
        }
        return input;
    }

    public static double[] clearDoubleArray(double[] array) {
        for (int i=0; i<array.length; i++) {
            array[i] = 0.0;
        }
        return array;
    }

    public static int[] clearIntArray(int[] array) {
        for (int i=0; i<array.length; i++) {
            array[i] = 0;
        }
        return array;
    }

    public static double[] calculateWeightedSum(double[][] weights, double[] neuronSum, double[] input) {
        for (int n=0; n<neuronSum.length; n++) {
            for (int i=0; i<weights.length; i++) {
                neuronSum[n] += weights[i][n] * input[i];

            }
        }
        return neuronSum;

    }

    public static int[] calculateOutput(double[] neuronSum, int[] output) {
        for (int i=0; i<neuronSum.length; i++) {
            if (neuronSum[i] > 0) {
                output[i] = 1;
            } else {
                output[i] = 0;
            }
        }
        return output;
    }

    public static int[] returnTargetVector(int[] targetVector, double label) {
//        System.out.println("Label: " + (int)label);
        int val = (int) label;

        for (int i = 0; i < 10; i++) {
            if (i == val) {
                targetVector[i] = 1;
            } else {
                targetVector[i] = 0;
            }
        }
        return targetVector;
    }

    public static double[][] updateWeightsIfNecessary(double[][] weights, double[] input, int[] output,  int[] target, double eta){
        boolean correctlyRecognized = true; // assume true
        for(int i=0; i<output.length; i++) {
            if(output[i] != target[i]) {
                correctlyRecognized = false;
//                System.out.println("Updating: " + i);
//                printDoubleMatrix(weights);
                weights = updateWeights(weights, input, output[i], target[i], i, eta);
//                printDoubleMatrix(weights);
            }
        }
        if (correctlyRecognized) {
            IMAGES_RECOGNIZED_TRAIN++;
        }
        return weights;
    }

    public static double[][] updateWeights(double[][] weights, double[] input, int y, int t, int neuronIndex, double eta) {
        for (int i = 0; i < weights.length; i++) {
            weights[i][neuronIndex] = weights[i][neuronIndex] - (eta * (y - t) * input[i]);
        }
        return weights;
    }

    public static int checkCorrect(int numCorrectTest, int[] output, int[] target) {
        if (Arrays.equals(output, target)) {
            return ++numCorrectTest;
        } else {
            return numCorrectTest;
        }
    }

    public static int[][] updateConfusionMatrix(int[][] matrix, int[] output, int target) {
        try {
            int outputVal = outputVectorToValue(output);
            matrix[outputVal][target]++;
        } catch (NoOutputException e) {
            //return matrix;
        }
        return matrix;
//        System.out.println("Output: " + outputVal + "Target: " + target);
    }

    public static int outputVectorToValue(int[] outputVector) throws NoOutputException {
        for (int i = 0; i < 10; i++) {
            if (outputVector[i] == 1) {
                return i;
            }
        }
        throw new NoOutputException("No neurons fired");
    }


    public static void main(String[] args) throws IOException {
        double[][] weights = new double[785][10];
        int[][] confusionMaxtrix = new int[10][10];
        int[] labelsTrain = new int[TRAIN_NUM_IMAGES];
        int[] labelsTest = new int[TEST_NUM_IMAGES];
        double[] data = new double[784];        // one "image" = 28x28 matrix = 784 cells
        double[] input = new double[785]; // 1 bias node + data
        double[] neuronSum = new double[10];
        int[] neuronOutput = new int[10];
        int[] target = new int[10];
        int[][] confusionMatrix = new int[10][10];


        // Initialize weights with values between -0.5 and 0.5
        weights = initializeWeights(weights);
        //printDoubleMatrix(weights);

        // Initialize labels from mnist
        labelsTrain = initializeLabels(labelsTrain, "mnist_train-labels.csv");
        labelsTest = initializeLabels(labelsTest, "mnist_test-labels.csv");


        System.out.println("Ep\tTrImg\tRec\t\tAcc\t\tTeImg\tRec\t\tAcc");
        // Enter epoch
        for (int e = 0; e < 30; e++) {
            File file = new File("mnist_train-images.csv");
            FileReader fr = new FileReader(file);
            BufferedReader br = new BufferedReader(fr);

            File file2 = new File("mnist_test-images.csv");
            FileReader fr2 = new FileReader(file2);
            BufferedReader br2 = new BufferedReader(fr2);

            IMAGES_SEEN_TRAIN = 0;
            IMAGES_RECOGNIZED_TRAIN = 0;
            IMAGES_SEEN_TEST = 0;
            IMAGES_RECOGNIZED_TEST = 0;

            // Enter image. For each image, i.e. one line from mnist_train-images.csv
            for (int i = 0; i < TRAIN_NUM_IMAGES; i++) {
                IMAGES_SEEN_TRAIN++;
                neuronSum = clearDoubleArray(neuronSum);
                neuronOutput = clearIntArray(neuronOutput);
                target = clearIntArray(target);

                data = getAndNormizeLine(br, data);
//                   printDoubleArray(data);
                input = initializeInputVector(input, data); // same as data, just with bias node
//                printDoubleArray(input);
                neuronSum = calculateWeightedSum(weights, neuronSum, input);
                neuronOutput = calculateOutput(neuronSum, neuronOutput);
//                printDoubleArray(neuronSum);
//                printIntArray(neuronOutput);
                target = returnTargetVector(target, labelsTrain[i]);
//                printIntArray(target);
                weights = updateWeightsIfNecessary(weights, input, neuronOutput, target, ETA);
//                System.out.println("Images recognized: " + IMAGES_RECOGNIZED_TRAIN);
            }
            for (int i = 0; i < TEST_NUM_IMAGES; i++) {
                IMAGES_SEEN_TEST++;
                neuronSum = clearDoubleArray(neuronSum);
                neuronOutput = clearIntArray(neuronOutput);
                target = clearIntArray(target);

                data = getAndNormizeLine(br2, data);
                input = initializeInputVector(input, data);
                neuronSum = calculateWeightedSum(weights, neuronSum, input);
                neuronOutput = calculateOutput(neuronSum, neuronOutput);
                target = returnTargetVector(target, labelsTest[i]);
                IMAGES_RECOGNIZED_TEST = checkCorrect(IMAGES_RECOGNIZED_TEST, neuronOutput, target);

                confusionMatrix = updateConfusionMatrix(confusionMatrix, neuronOutput, labelsTest[i]);


            }
            System.out.println(e + "\t" + IMAGES_SEEN_TRAIN + "\t" + IMAGES_RECOGNIZED_TRAIN + "\t" + df.format(IMAGES_RECOGNIZED_TRAIN / (double) IMAGES_SEEN_TRAIN) +
                    "\t" + IMAGES_SEEN_TEST + "\t" + IMAGES_RECOGNIZED_TEST + "\t" + df.format(IMAGES_RECOGNIZED_TEST / (double) IMAGES_SEEN_TEST));
        }
        printConfusionMatrix(confusionMatrix);
    }
    public static void printConfusionMatrix(int[][] matrix) {
        System.out.println("Confusion Matrix");
        for (int[] row : matrix) {
            for (int val : row) {
                System.out.print(val + "\t");
            }
            System.out.println();
        }
        System.out.println();
    }
    public static void printIntArray(int[] array) {
        for(int i : array) {
            System.out.print(i + "\t\t");
        }
        System.out.println();
    }
    public static void printDoubleArray(double[] array) {
        int i=0;
        for(double d : array) {
            System.out.print(i++ + "\t\t");
        }
        System.out.println();

        for(double d : array) {
            System.out.print(df.format(d) + "\t");
        }
        System.out.println();;
    }
    public static void printDoubleMatrix(double[][] matrix) {
        DecimalFormat df = new DecimalFormat("#.000");
        int col = 0;
        for (double[] da : matrix) {
            for (double d : da) {
                col++;
                System.out.print(df.format(d) + "\t");
            }
            System.out.println();
        }
        System.out.println("Total row: " + matrix.length + " Total cells: " + col);
    }
}
