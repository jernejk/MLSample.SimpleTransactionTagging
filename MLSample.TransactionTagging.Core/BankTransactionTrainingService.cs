using Microsoft.ML;
using System.Collections.Generic;
using System.Linq;

namespace MLSample.TransactionTagging
{
    public class BankTransactionTrainingService
    {
        public void Train(IEnumerable<Transaction> trainingData, string modelSavePath)
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
                .Transforms.Conversion.MapValueToKey(inputColumnName: nameof(Transaction.Category), outputColumnName: "Label")
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(Transaction.Description), outputColumnName: "TitleFeaturized"))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(Transaction.TransactionType), outputColumnName: "DescriptionFeaturized"))
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

        private Microsoft.ML.Trainers.SdcaMaximumEntropyMulticlassTrainer GetScadaTrainer(MLContext mlContext)
        {
            return mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features");
        }
    }
}
