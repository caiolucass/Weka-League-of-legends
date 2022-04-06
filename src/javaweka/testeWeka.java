/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package javaweka;

import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import weka.core.converters.ConverterUtils.DataSource;
import java.util.Random;

/**
 *
 * @author caiol
 */
public class testeWeka {
    private Instances dados;
    private String caminhoDados;
    
    public testeWeka(String caminho){
        caminhoDados = caminho;
    }

    /**
     * @leDados()
     * Metodo para ler os dados do arquivo de dados
     * @throws Exception
     */
    public void leDados() throws Exception {
        DataSource leitura = new DataSource(caminhoDados);
        dados = leitura.getDataSet();
        if (dados.classIndex() == -1)
        dados.setClassIndex(dados.numAttributes() - 1);
    }

    /**
     * @imprimeDados()
     * Metodo para imprimir os dados
     */
    public void imprimeDados() {
        for (int i = 0; i < dados.numInstances(); i++) {
        Instance instance = dados.instance(i);
        System.out.println((i + 1) + ": " + instance + "\n");
        }
    }

    /**
     * Metodo implementando o algoritmo de arvoreDeDecisaoJ48()
     * @arvoreDeDecisaoJ48()
     * @return new double
     * @throws Exception
     */
    public double[] arvoreDeDecisaoJ48() throws Exception {
        J48 tree = new J48();
        tree.buildClassifier(dados);
        System.out.println(tree);
        System.out.println("Avaliacao inicial: \n");

        Evaluation evaluation;
        evaluation = new Evaluation(dados);
        evaluation.evaluateModel(tree, dados);

        var inicial = evaluation.correct();
        System.out.println("--> Instancias corretas: " +
        inicial + "\n");
        System.out.println("Avaliacao cruzada: \n");

        Evaluation avalCruzada;
        avalCruzada = new Evaluation(dados);
        avalCruzada.crossValidateModel(tree, dados, 10, new
        Random(1));

        var cruzada = avalCruzada.correct();
        System.out.println("--> Instancias corretas CV: " +
        cruzada + "\n");

        return new double[]{Double.parseDouble(String.valueOf(inicial)), Double.parseDouble(String.valueOf(cruzada))};
    }
    
    public String InstanceBased(int GOLDEARNED, int TOTALMINIONSKILLED, int KILLS, int ASSISTS, int DEATHS,
                                String CHAMPION, int VISIONSCORE, int TOTALDAMAGEDEALTTOCHAMPIONS) throws Exception{
        IBk k3 = new IBk(3);
        k3.buildClassifier(dados);

        Instance instance = new DenseInstance(8);
        instance.setDataset(dados);
        instance.setValue(0, GOLDEARNED);
        instance.setValue(1, TOTALMINIONSKILLED);
        instance.setValue(2, KILLS);
        instance.setValue(3, ASSISTS);
        instance.setValue(4, DEATHS);
        instance.setValue(5, CHAMPION);
        instance.setValue(6, VISIONSCORE);
        instance.setValue(7, TOTALDAMAGEDEALTTOCHAMPIONS);

        double classifyInstance = k3.classifyInstance(instance);
        System.out.println("Predição: " + classifyInstance);
        Attribute a = dados.attribute(4);
        String predClass = a.value((int) classifyInstance);
        System.out.println("Predição: " + predClass);

        return predClass;
    }
}
