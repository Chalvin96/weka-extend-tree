import weka.classifiers.C45;
import weka.classifiers.Evaluation;
import weka.classifiers.Id3TomMitchell;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;

public class Main {


    public static Instances ReadArff(String filename) throws Exception {
        BufferedReader reader = new BufferedReader(
                new FileReader(filename));
        Instances data = new Instances(reader);
        data.setClassIndex(data.numAttributes() - 1);
        reader.close();

        return data;
    }

    public static void main(String[] args) throws Exception {


        Instances data = ReadArff(String.valueOf("/root/Documents/weka-3-6-14/src/data/soybean.arff"));
        data.randomize(new java.util.Random(0));
        int trainSize = (int) Math.round(data.numInstances() * 0.8);
        int testSize = data.numInstances() - trainSize;
        Instances train = new Instances(data, 0, trainSize);
        Instances test = new Instances(data, trainSize, testSize);

        C45 c45 = new C45();

        c45.buildClassifier(train);

        Evaluation eval = new Evaluation(test);
        eval.evaluateModel(c45, test);

        System.out.println();
        System.out.println("=== Summary ===");
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());

        c45.prune(test,c45);

        eval = new Evaluation(test);

        eval.evaluateModel(c45, test);

        System.out.println();
        System.out.println("=== Summary ===");
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toMatrixString());
    }


}
