import org.deeplearning4j.datasets.iterator.impl.Cifar10DataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class CIFAR10Classifier {
    public static void main(String[] args) throws Exception {
        // -------------- Define Model Hyperparameters --------------
        int seed = 123; // Random seed for reproducibility
        int batchSize = 128; // Number of images processed in a batch
        int numClasses = 10; // CIFAR-10 has 10 categories
        int epochs = 50; // Number of training epochs
        double learningRate = 0.001; // Initial learning rate for training

        // -------------- Load and Preprocess CIFAR-10 Dataset --------------
        // The dataset contains 32x32 color images categorized into 10 classes.
        DataSetIterator trainIterator = new Cifar10DataSetIterator(batchSize, true);
        DataSetIterator testIterator = new Cifar10DataSetIterator(batchSize, false);

        // Normalizes pixel values to the range [0,1] for stable training.
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        trainIterator.setPreProcessor(scaler);
        testIterator.setPreProcessor(scaler);

        // -------------- Define Convolutional Neural Network (CNN) --------------
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed) // Set random seed
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(learningRate)) // Use Adam optimizer for efficient training
                .l2(0.0005) // Regularization to prevent overfitting
                .list()
                // First Convolutional Layer: Extracts basic image features
                .layer(0, new ConvolutionLayer.Builder(3, 3)
                        .nIn(3) // Number of input channels (RGB)
                        .stride(1, 1)
                        .nOut(32) // Number of filters (feature maps)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                // Pooling Layer: Reduces spatial dimensions while preserving important features
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                // Second Convolutional Layer: Detects more complex patterns
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                // Second Pooling Layer: Further reduces dimensions
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                // Fully Connected Layer: Learns high-level image representations
                .layer(4, new DenseLayer.Builder()
                        .nOut(512)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                // Output Layer: Predicts one of the 10 classes
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX) // Softmax for multi-class classification
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .build();

        // -------------- Initialize and Train the Model --------------
        org.deeplearning4j.nn.multilayer.MultiLayerNetwork model = 
                new org.deeplearning4j.nn.multilayer.MultiLayerNetwork(config);
        model.init();
        model.setListeners(new ScoreIterationListener(5)); // Prints loss every 5 iterations

        System.out.println("Starting Training...");
        for (int i = 0; i < epochs; i++) {
            model.fit(trainIterator); // Train the model with CIFAR-10 images
            System.out.println("Epoch " + (i + 1) + " completed");
        }

        // -------------- Evaluate the Model --------------
        System.out.println("Evaluating Model...");
        org.deeplearning4j.eval.Evaluation evaluation = model.evaluate(testIterator);
        System.out.println(evaluation.stats()); // Print accuracy, precision, recall

        // -------------- Save the Model --------------
        System.out.println("Saving Model...");
        model.save(new java.io.File("cifar10_model.zip"));
        System.out.println("Model Saved Successfully!");
    }
}
