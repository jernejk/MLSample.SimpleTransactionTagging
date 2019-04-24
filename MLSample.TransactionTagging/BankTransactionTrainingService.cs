using Microsoft.ML;
using System.Collections.Generic;
using System.Linq;

namespace MLSample.TransactionTagging
{
    public class BankTransactionTrainingService
    {
        public void Train(IEnumerable<TransactionData> trainingData, string modelSavePath)
        {
            var mlContext = new MLContext(seed: 0);

            // Configure ML pipeline
            var pipeline = LoadDataProcessPipeline(mlContext);
            var trainingPipeline = GetTrainingPipeline(mlContext, pipeline);
            var trainingDataView = mlContext.Data.LoadFromEnumerable(trainingData);

            // Generate training model.
            var trainingModel = trainingPipeline.Fit(trainingDataView);

            // Save training model to disk.
            mlContext.Model.Save(trainingModel, trainingDataView.Schema, modelSavePath);
        }

        private IEstimator<ITransformer> LoadDataProcessPipeline(MLContext mlContext)
        {
            // Configure data pipeline based on the features in TransactionData.
            // Description and TransactionType are the inputs and Category is the expected result.
            var dataProcessPipeline = mlContext
                .Transforms.Conversion.MapValueToKey(inputColumnName: nameof(TransactionData.Category), outputColumnName: "Label")
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(TransactionData.Description), outputColumnName: "TitleFeaturized"))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(TransactionData.TransactionType), outputColumnName: "DescriptionFeaturized"))
                // Merge two features into a single feature.
                .Append(mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                .AppendCacheCheckpoint(mlContext);

            return dataProcessPipeline;
        }

        private IEstimator<ITransformer> GetTrainingPipeline(MLContext mlContext, IEstimator<ITransformer> pipeline)
        {
            // Use the multi-class SDCA algorithm to predict the label using features.
            // For StochasticDualCoordinateAscent the KeyToValue needs to be PredictedLabel.
            return pipeline
                .Append(GetScadaTrainer(mlContext))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
        }

        private IEstimator<ITransformer> GetScadaTrainer(MLContext mlContext)
        {
            return mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features");
        }

        // Alternative trainer
        private IEstimator<ITransformer> GetAveragedPerceptronTrainer(MLContext mlContext)
        {
            // Create a binary classification trainer.
            var averagedPerceptronBinaryTrainer = mlContext.BinaryClassification.Trainers.AveragedPerceptron("Label", "Features", numberOfIterations: 10);

            // Compose an OVA (One-Versus-All) trainer with the BinaryTrainer.
            // In this strategy, a binary classification algorithm is used to train one classifier for each class, "
            // which distinguishes that class from all other classes. Prediction is then performed by running these binary classifiers, "
            // and choosing the prediction with the highest confidence score.
            return mlContext.MulticlassClassification.Trainers.OneVersusAll(averagedPerceptronBinaryTrainer);
        }
    }
}
