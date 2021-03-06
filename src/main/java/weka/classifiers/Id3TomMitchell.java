package weka.classifiers;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Enumeration;

public class Id3TomMitchell extends Classifier {


    private Id3TomMitchell[] son;
    private Attribute decision_attribute;
    private double ClassValue;
    private double eps = 10e-7;
    private boolean isLeaf = false;

    public void buildClassifier(Instances data) throws Exception {

        makeTree(data);
    }

    private void makeTree(Instances data) throws Exception{

        double max_info_gain = -1.0;
        Attribute att_with_max_info_gain = null;

        Enumeration attEnum = data.enumerateAttributes();
        while (attEnum.hasMoreElements()) {
            Attribute att = (Attribute) attEnum.nextElement();
            double infoGain = getInfoGain(data,att);
            if (infoGain > max_info_gain){
                max_info_gain = infoGain;
                att_with_max_info_gain = att;
            }
        }

        if(max_info_gain < eps){
            isLeaf = true;
            ClassValue = mostCommonClass(data);
        } else{
            decision_attribute = att_with_max_info_gain;
            son = new Id3TomMitchell[data.numDistinctValues(decision_attribute)];

            Instances[] splitedData = splitData(data,decision_attribute);


            System.out.println("Dictinct Value : ");
            for(int i=0; i< data.numDistinctValues(decision_attribute); i++){
                son[i] = new Id3TomMitchell();
                if( splitedData[i].numInstances() > 0) {
                    son[i].makeTree(splitedData[i]);
                }else{
                    son[i].isLeaf = true;
                    son[i].ClassValue = mostCommonClass(data);
                }
            }

        }
    }

    @Override
    public double classifyInstance(Instance record) {
        if(isLeaf == true){
            return ClassValue;
        }
        else{
            return son[ (int)record.value(decision_attribute) ].classifyInstance(record);
        }
    }



    private double mostCommonClass(Instances data){
        int classCount[] = new int[data.numClasses()];
        Enumeration instEnum = data.enumerateInstances();


        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();

            classCount[(int) inst.classValue()]++;
        }

        int idx = 0;
        int max = classCount[0];

        for (int i=1;i < data.numClasses(); i++){
            if (classCount[i] > max){
                max = classCount[i];
                idx = i;
            }
        }

        return (double) idx;

    }


    private Instances[] splitData(Instances data, Attribute att){
        Instances[] res = new Instances[att.numValues()];

        for (int i=0; i<att.numValues(); i++){
            res[i] = new Instances(data, data.numInstances());
        }

        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            res[(int) inst.value(att)].add(inst);
        }

        return res;

    }

    private double getGainRatio(Instances data, Attribute att) throws Exception {
        return getInfoGain(data,att) /SplitInformation(data,att) ;
    }

    private double getGainRatio(Instances data, Attribute att) throws Exception {

    }


    private double getInfoGain(Instances data, Attribute att)
            throws Exception {

        double infoGain = getEntropy(data);
        Instances[] splitData = splitData(data, att);
        for (int j = 0; j < att.numValues(); j++) {
            if (splitData[j].numInstances() > 0) {
                infoGain -= ((double) splitData[j].numInstances() /
                        (double) data.numInstances()) *
                        getEntropy(splitData[j]);
            }
        }
        return infoGain;
    }

    private double getEntropy(Instances data){
        double [] classCounts = new double[data.numClasses()];
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            classCounts[(int) inst.classValue()]++;
        }
        double entropy = 0;
        for (int j = 0; j < data.numClasses(); j++) {
            if (classCounts[j] > 0) {
                entropy -= classCounts[j] / data.numInstances() * Utils.log2(classCounts[j] / data.numInstances());
            }
        }
        return entropy;
    }

}
