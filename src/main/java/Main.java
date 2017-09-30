import com.sun.org.apache.xpath.internal.SourceTree;
import weka.classifiers.C45;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.Id3TomMitchell;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;
import java.util.Scanner;

public class Main {


    public static Instances ReadArff(String filename) throws Exception {
        BufferedReader reader = new BufferedReader(
                new FileReader(filename));
        Instances data = new Instances(reader);
        data.setClassIndex(data.numAttributes() - 1);
        reader.close();

        return data;
    }

    public static void printAtr(Instances data) throws Exception{
        for(int i=0; i<data.numAttributes(); i++){
            System.out.println((i+1)+" "+data.attribute(i));
        }
    }

    public static void saveModel(String name, Classifier cls) throws Exception{

        System.out.print("Do you want to save model? (y/n)");
        Scanner s = new Scanner(System.in);
        String answer = s.next().toLowerCase();
        if (answer.equals("y") || answer.equals("yes")) {
            System.out.print("Insert save location: ");
            String filesave = s.next();
            weka.core.SerializationHelper.write(filesave + "/"+name+".model", cls);
            System.out.println("Saved");
        }
    }

    public static void printResult(Evaluation eval) throws Exception{
        System.out.println();
        System.out.println("=== Summary ===");
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());
    }

    public static void main(String[] args) throws Exception {
        boolean quit = false;
        while(!quit) {
            Thread.sleep(2000);
            System.out.println();
            System.out.println("===DTL Classification Program===");
            System.out.println("Menu Learning:");
            System.out.println("    1. from new dataset");
            System.out.println("    2. from existing model");
            System.out.println("    0. Quit");
            System.out.print("Your option: ");
            Scanner s = new Scanner(System.in);
            int option = s.nextInt();
            switch (option) {
                case 1:
                    System.out.println();
                    System.out.print("Insert dataset location: ");
                    s = new Scanner(System.in);
                    String filename = s.nextLine();

                    Instances data = ReadArff(String.valueOf(filename));
                    printAtr(data);

                    boolean stop = false;
                    while (!stop) {
                        Thread.sleep(2000);
                        System.out.println();
                        System.out.println("Dataset has been loaded");
                        System.out.println("Menu:");
                        System.out.println("    1. Remove attribute?");
                        System.out.println("    2. Resample?");
                        System.out.println("    3. ID3 - 10 Cross Validation");
                        System.out.println("    4. ID3 - Percentage Split");
                        System.out.println("    5. ID3 - Training Test");
                        System.out.println("    6. C4.5 - 10-fold Cross Validation - No Prune");
                        System.out.println("    7. C4.5 - 10-fold Cross Validation - Prune");
                        System.out.println("    8. C4.5 - Percentage Split - No Prune");
                        System.out.println("    9. C4.5 - Percentage Split - Prune");
                        System.out.println("    10. C4.5 - Training Test - No Prune");
                        System.out.println("    11. C4.5 - Training Test - Prune");
                        System.out.println("    0. Quit");
                        System.out.print("Your option: ");
                        s = new Scanner(System.in);
                        int option1 = s.nextInt();
                        Id3TomMitchell id3 =  null;
                        C45 c45 = null;
                        Evaluation eval = null;
                        Instances test = null;
                        Instances train = null;
                        Random rand = null;
                        int trainSize;
                        int testSize;

                        switch (option1) {
                            case 1: //remove attribute
                                System.out.println("Insert index of the attribute you want to delete: ");
                                s = new Scanner(System.in);
                                String deleteAtr = s.next();
                                try {
                                    Remove remove = new Remove();
                                    remove.setAttributeIndices(deleteAtr);
                                    remove.setInputFormat(data);
                                    data = Filter.useFilter(data, remove);
                                    printAtr(data);
                                } catch (Exception e) {
                                    System.out.println("Remove attribute error");
                                    e.printStackTrace();
                                }
                                break;
                            case 2: //resampling
                                System.out.println("Resampling...");
                                final Resample filter = new Resample();
                                filter.setBiasToUniformClass(1.0);
                                try {
                                    filter.setInputFormat(data);
                                    filter.setNoReplacement(false);
                                    filter.setSampleSizePercent(100);
                                    data = Filter.useFilter(data, filter);
                                    System.out.println(data);
                                } catch (Exception e) {
                                    System.out.println("Resampling error");
                                    e.printStackTrace();
                                }
                                break;
                            case 3: //id3 - 10folds
                                System.out.println("ID3 10-folds Cross Validation choosen");

                                try {
                                    data.randomize(new java.util.Random(0));
                                    id3 = new Id3TomMitchell();
                                    id3.buildClassifier(data);

                                    eval = new Evaluation(data);
                                    rand = new Random(1);
                                    eval.crossValidateModel(id3, data, 10, rand);

                                    printResult(eval);
                                    saveModel("id3_10fold", id3);
                                } catch (Exception e){
                                    System.out.println("Error while learning");
                                    e.printStackTrace();
                                }
                                break;
                            case 4: //id3 - percentage split
                                System.out.println("Percentage Split");
                                System.out.print("Insert % data training (out of 100%): ");

                                s = new Scanner(System.in);
                                double trainPercent = s.nextInt();
                                try {

                                    data.randomize(new java.util.Random(0));
                                    trainSize = (int) Math.round(data.numInstances() * trainPercent / 100);
                                    testSize = data.numInstances() - trainSize;
                                    System.out.println(data.numInstances() + " " + trainSize + " " + testSize);
                                    train = new Instances(data, 0, trainSize);
                                    test = new Instances(data, trainSize, testSize);

                                    id3 = new Id3TomMitchell();
                                    id3.buildClassifier(train);

                                    eval = new Evaluation(test);
                                    eval.evaluateModel(id3, test);

                                    printResult(eval);
                                    saveModel("id3_percentage", id3);
                                } catch (Exception e){
                                    System.out.println("Error while learning");
                                    e.printStackTrace();
                                }
                                break;
                            case 5:
                                System.out.print("Insert file test location: ");
                                s = new Scanner(System.in);
                                filename = s.nextLine();

                                try {
                                    test = ReadArff(String.valueOf(filename));
                                    printAtr(data);
                                    data.randomize(new java.util.Random(0));
                                    id3 = new Id3TomMitchell();
                                    id3.buildClassifier(data);

                                    eval = new Evaluation(data);
                                    eval.evaluateModel(id3, test);

                                    printResult(eval);
                                    saveModel("id3_training_test", id3);
                                } catch (Exception e){
                                    System.out.println("Error while learning");
                                    e.printStackTrace();
                                }
                                break;
                            case 6: //c45 - 10folds - no prune
                                System.out.println("C4.5 10-folds Cross Validation No Prune choosen");

                                try {
                                    data.randomize(new java.util.Random(0));
                                    c45 = new C45();
                                    c45.buildClassifier(data);
                                    eval = new Evaluation(data);
                                    rand = new Random(1);
                                    eval.crossValidateModel(c45, data, 10, rand);

                                    printResult(eval);
                                    saveModel("c45_10fold_noprune", c45);
                                } catch (Exception e){
                                    System.out.println("Error while learning");
                                    e.printStackTrace();
                                }
                                break;
                            case 7: //c45 - 10folds prune
                                System.out.println("C4.5 10-folds Cross Validation Prune choosen");

                                try {
                                    data.randomize(new java.util.Random(0));
                                    c45 = new C45();
                                    c45.buildClassifier(data);

                                    trainSize = (int) Math.round(data.numInstances() * 0.9);
                                    testSize = data.numInstances() - trainSize;
                                    test = new Instances(data, trainSize, testSize);
                                    c45.prune(test, c45);

                                    eval = new Evaluation(data);
                                    rand = new Random(1);
                                    eval.crossValidateModel(c45, data, 10, rand);

                                    printResult(eval);
                                    saveModel("c45_10fold_prune", c45);
                                } catch (Exception e){
                                    System.out.println("Error while learning");
                                    e.printStackTrace();
                                }
                                break;
                            case 8://c4.5 percent split no prune
                                System.out.println("Percentage Split No Prune");
                                System.out.print("Insert % data training (out of 100%): ");

                                s = new Scanner(System.in);
                                trainPercent = s.nextInt();

                                try {
                                    data.randomize(new java.util.Random(0));
                                    trainSize = (int) Math.round(data.numInstances() * trainPercent / 100);
                                    testSize = data.numInstances() - trainSize;
                                    train = new Instances(data, 0, trainSize);
                                    test = new Instances(data, trainSize, testSize);

                                    c45 = new C45();
                                    c45.buildClassifier(train);

                                    eval = new Evaluation(test);
                                    eval.evaluateModel(c45, test);

                                    printResult(eval);
                                    saveModel("c45_percentage_noprune", c45);
                                } catch (Exception e){
                                    System.out.println("Error while learning");
                                    e.printStackTrace();
                                }
                                break;
                            case 9://c4.5 percent split prune
                                System.out.println("Percentage Split Prune");
                                System.out.print("Insert % data training (out of 100%): ");

                                try {
                                    s = new Scanner(System.in);
                                    trainPercent = s.nextInt();

                                    data.randomize(new java.util.Random(0));
                                    trainSize = (int) Math.round(data.numInstances() * trainPercent / 100);
                                    testSize = data.numInstances() - trainSize;
                                    train = new Instances(data, 0, trainSize);
                                    test = new Instances(data, trainSize, testSize);

                                    c45 = new C45();
                                    c45.buildClassifier(train);
                                    c45.prune(test, c45);

                                    eval = new Evaluation(test);
                                    eval.evaluateModel(c45, test);

                                    printResult(eval);
                                    saveModel("c45_percentage_prune", c45);
                                } catch (Exception e){
                                    System.out.println("Error while learning");
                                    e.printStackTrace();
                                }
                                break;
                            case 10:
                                System.out.println("C4.5 Training Set No Prune");
                                System.out.print("Insert file test location: ");
                                s = new Scanner(System.in);
                                filename = s.nextLine();

                                try {
                                    test = ReadArff(String.valueOf(filename));
                                    printAtr(data);
                                    data.randomize(new java.util.Random(0));
                                    c45 = new C45();
                                    c45.buildClassifier(data);

                                    eval = new Evaluation(data);
                                    eval.evaluateModel(c45, test);

                                    printResult(eval);
                                    saveModel("c45_training_test_noprune", c45);
                                } catch (Exception e){
                                    System.out.println("Error while learning");
                                    e.printStackTrace();
                                }
                                break;
                            case 11:
                                System.out.println("C4.5 Training Set Prune");
                                System.out.print("Insert file test location: ");
                                s = new Scanner(System.in);
                                filename = s.nextLine();

                                try {
                                    test = ReadArff(String.valueOf(filename));
                                    printAtr(data);

                                    data.randomize(new java.util.Random(0));
                                    c45 = new C45();
                                    c45.buildClassifier(data);
                                    c45.prune(test, c45);

                                    eval = new Evaluation(data);
                                    eval.evaluateModel(c45, test);

                                    printResult(eval);
                                    saveModel("c45_training_test_prune", c45);
                                } catch (Exception e){
                                    System.out.println("Error while learning");
                                    e.printStackTrace();
                                }
                                break;
                            case 0:
                                stop = true;
                                break;
                            default:
                                System.out.println("Wrong option");
                                stop = true;
                        }
                    }
                    break;
                case 2:
                    System.out.print("Insert model location: ");
                    String modelload = s.next();
                    Classifier cls = (Classifier) weka.core.SerializationHelper.read(modelload);

                    System.out.print("Insert file test location: ");
                    s = new Scanner(System.in);
                    filename = s.nextLine();

                    Instances test = ReadArff(String.valueOf(filename));
                    Evaluation eval = new Evaluation(test);
                    eval.evaluateModel(cls, test);

                    System.out.println();
                    System.out.println("=== Summary ===");
                    System.out.println(eval.toSummaryString());
                    System.out.println(eval.toMatrixString());
                    break;
                case 0:
                    quit = true;
                    System.out.println("Quitting...");
                    break;
                default:
                    System.out.println("Wrong option");
                    break;
            }
        }
    }
}