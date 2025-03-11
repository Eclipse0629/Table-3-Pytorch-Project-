import ai.djl.Application;
import ai.djl.Model;
import ai.djl.ModelZoo;
import ai.djl.basicdataset.cv.classification.Cifar10;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.transform.RandomFlipLeftRight;
import ai.djl.modality.cv.transform.RandomResizedCrop;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolution.Conv2d;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.pooling.Pool;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.util.Utils;
import java.io.IOException;

public class Cifar10Trainer {
    public static void main(String[] args) throws IOException, TranslateException {
        // -------------- Device Configuration --------------
        String device = Engine.getInstance().getGpuCount() > 0 ? "GPU" : "CPU";
        System.out.println("Using device: " + device);

        // -------------- Define Neural Network Model --------------
        Block block = new SequentialBlock()
                .add(Conv2d.builder().setKernelShape(new int[]{3, 3}).setFilters(32).build())
                .add(BatchNorm.builder().build())
                .add(Pool.maxPool2dBlock(new int[]{2, 2}, new int[]{2, 2}))
                .add(Linear.builder().setUnits(10).build());
        
        // -------------- Data Preprocessing --------------
        Cifar10 dataset = Cifar10.builder()
                .setSampling(128, true)
                .optUsage(Dataset.Usage.TRAIN)
                .addTransform(new RandomResizedCrop(32))
                .addTransform(new RandomFlipLeftRight())
                .addTransform(new ToTensor())
                .build();
        dataset.prepare(new ProgressBar());
        
        RandomAccessDataset testDataset = Cifar10.builder()
                .setSampling(100, false)
                .optUsage(Dataset.Usage.TEST)
                .addTransform(new ToTensor())
                .build();
        testDataset.prepare(new ProgressBar());

        // -------------- Define Loss Function & Optimizer --------------
        DefaultTrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .optMetrics(new Metrics())
                .optDevices(Engine.getInstance().getDevices());

        Model model = Model.newInstance("CIFAR-10", block);
        Trainer trainer = model.newTrainer(config);

        // -------------- Training & Testing Loop --------------
        for (int epoch = 0; epoch < 200; epoch++) {
            System.out.println("Epoch: " + epoch);
            trainer.trainDataset(dataset);
            trainer.validateDataset(testDataset);
        }
        
        model.close();
    }
}
