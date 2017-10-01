package weka.classifiers;

import weka.core.*;

import java.util.Enumeration;
import java.util.Vector;

public class C45 extends Classifier {


    private C45[] son;
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
            //to be used on pruning
            ClassValue = mostCommonClass(data);

            son = new C45[data.numDistinctValues(decision_attribute)];



            Instances[] splitedData = splitData(data,decision_attribute);


            for(int i=0; i< data.numDistinctValues(decision_attribute); i++){
                son[i] = new C45();
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

    public void prune(Instances data, C45 root) throws Exception{
        if(isLeaf == true){
            return;
        }

        for(int i=0; i<son.length; i++){
            son[i].prune(data,root);
        }

        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(root, data);

        double before_pruning_error = eval.errorRate();

        isLeaf = true;
        eval.evaluateModel(root, data);
        double after_pruning_error  = eval.errorRate();

        if(after_pruning_error > before_pruning_error){
            isLeaf = false;
        }
    }

    private double mostCommonTargetValue(Instances data, Attribute att){
        int targetCount[] = new int[data.numDistinctValues(att)];
        Enumeration instEnum = data.enumerateInstances();


        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();

            targetCount[(int) inst.value(att)]++;
        }

        int idx = -1;
        int max = -1;

        for (int i=0;i < data.numDistinctValues(att); i++){
            if ( !(att.value(i) == "?") && targetCount[i] > max){
                max = targetCount[i];
                idx = i;
            }
        }

        return (double) idx;

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
            if(inst.value(att) < 0)
                System.out.println(inst.value(att));
            res[(int) inst.value(att)].add(inst);
        }

        return res;

    }

    private Instances[] splitDataWithMissingValueHandle(Instances data, Attribute att){

        Instances[] res = new Instances[att.numValues()];
        double mostCommonTargetValue = mostCommonTargetValue(data,att);

        for (int i=0; i<att.numValues(); i++){
            res[i] = new Instances(data, data.numInstances());
        }

        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            if(inst.stringValue(att) == "?")
                res[(int) mostCommonTargetValue].add(inst);
            else
                res[(int) inst.value(att)].add(inst);
        }

        return res;

    }




    private double getGainRatio(Instances data, Attribute att) throws Exception {
        return getInfoGain(data,att) /SplitInformation(data,att) ;
    }

    private double SplitInformation(Instances data, Attribute att) throws Exception {
        double attCount[] = new double[data.numAttributes()];
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            attCount[(int) inst.value(att)]++;
        }

        double result = 0.0;
        for (int i=0; i<data.numAttributes(); i++){
            result -= attCount[i]/data.numInstances() * Utils.log2(attCount[i]/data.numInstances());
        }

        return result;
    }


    private double getInfoGain(Instances data, Attribute att)
            throws Exception {

        double infoGain = getEntropy(data);
        Instances[] splitData = splitDataWithMissingValueHandle(data, att);

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

    private Vector<Integer> getNumericAttributeIndex(Instances inst) {
        Vector<Integer> v = new Vector<Integer>();
        for (int i = 0; i < inst.numAttributes(); i++) {
            if (inst.attribute(i).isNumeric()) {
                v.add(i);
            }
        }
        return v;
    }

    private Instances[] splitData(Instances data, int att, double threshold){
        Instances[] res = new Instances[2];

        for (int i=0; i<2; i++){
            res[i] = new Instances(data, data.numInstances());
        }

//        Enumeration instEnum = data.enumerateInstances();
//        while (instEnum.hasMoreElements()) {
//            Instance inst = (Instance) instEnum.nextElement();
//            res[(int) inst.value(att)].add(inst);
//        }
        for (int i = 0; i < data.numInstances(); i++) {
            if (data.instance(i).value(att) > threshold) {
                res[1].add(data.instance(i));
            }
            else {
                res[0].add(data.instance(i));
            }
        }

        return res;

    }

    private double getInfoGain(Instances data, int att, double threshold)
            throws Exception {

        double infoGain = getEntropy(data);
        Instances[] splitData = splitData(data, att, threshold);
        for (int j = 0; j < 2; j++) {
            if (splitData[j].numInstances() > 0) {
                infoGain -= ((double) splitData[j].numInstances() /
                        (double) data.numInstances()) *
                        getEntropy(splitData[j]);
            }
        }
        return infoGain;
    }

    public Instances handleContinuousAttribute(Instances inst) throws Exception{
        Vector<Integer> v = getNumericAttributeIndex(inst);
        for (int i = 0; i < v.size(); i++) {
            inst.sort(v.get(i));
            double currentClassValue = inst.instance(0).classValue();
            Vector<Double> d = new Vector<Double>(0);
            for (int j = 1; j < inst.numInstances(); j++) {
                if (inst.instance(j).classValue()!=currentClassValue) {
                    if (d.size()<10) {
                        d.add((inst.instance(j).value(v.get(i)) + inst.instance(j - 1).value(v.get(i))) / 2.0);
                    }
                    currentClassValue = inst.instance(j).classValue();
                }
            }

            double threshold = d.get(0);

            for (int j = 1; j < d.size(); j++) {
                //cari information gain max
                //dapet angka threshold
                if (getInfoGain(inst, v.get(i), d.get(j)) > getInfoGain(inst, v.get(i), d.get(j-1))) {
                    threshold = d.get(i);
                }

            }
//
//            List<String> l;
//            l.add("0");
//            l.add("1");

            FastVector values = new FastVector(); /* FastVector is now deprecated. Users can use any java.util.List */
            values.addElement("0");               /* implementation now */
            values.addElement("1");
            inst.insertAttributeAt(new Attribute(inst.attribute(v.get(i)).name()+"new", values), inst.numAttributes());
            for (int j = 0; j < inst.numInstances(); j++) {
                //ganti nilai atribut
                //pake instance.setValue
                if (inst.instance(j).value(v.get(i)) > threshold) {
                    inst.instance(j).setValue(inst.numAttributes()-1, "1");
                }
                else {
                    inst.instance(j).setValue(inst.numAttributes()-1, "0");
                }
            }
        }

        for (int i = 0; i < v.size(); i++) {
            inst.deleteAttributeAt(v.get(i)-i);
        }

        return inst;
    }

}
