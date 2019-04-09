using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace MLSample.TransactionTagging
{
    public class BankTransactionTrainingService
    {
        public void Train(IEnumerable<TransactionData> trainingData, string modelSavePath)
        {
            var mlContext = new MLContext(seed: 0);
            var pipeline = LoadDataPipeline(mlContext);
            var trainingPipeline = GetTrainingPipeline(mlContext, pipeline);
            var trainingModel = BuildAndTrainModel(mlContext, trainingPipeline, trainingData);

            using (var fs = new FileStream(modelSavePath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(trainingModel, fs);
        }

        private IEstimator<ITransformer> LoadDataPipeline(MLContext mlContext)
        {
            // Configure data pipeline based on the features in TransactionData.
            // Description and TransactionType are the inputs and Category is the expected result.
            var pipeline = mlContext
                .Transforms.Conversion.MapValueToKey(inputColumnName: nameof(TransactionData.Category), outputColumnName: "Label")
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(TransactionData.Description), outputColumnName: "TitleFeaturized"))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(TransactionData.TransactionType), outputColumnName: "DescriptionFeaturized"))
                // Merge two features into a single feature.
                .Append(mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                .AppendCacheCheckpoint(mlContext);

            return pipeline;
        }

        private IEstimator<ITransformer> GetTrainingPipeline(MLContext mlContext, IEstimator<ITransformer> pipeline)
        {
            // Use the multi-class SDCA algorithm to predict the label using features.
            // For StochasticDualCoordinateAscent the KeyToValue needs to be PredictedLabel.
            return pipeline
                .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(DefaultColumnNames.Label, DefaultColumnNames.Features))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
        }

        private ITransformer BuildAndTrainModel(MLContext mlContext, IEstimator<ITransformer> trainingPipeline, IEnumerable<TransactionData> trainingData)
        {
            // Load training data and train the model based on training pipeline.
            var trainingDataView = mlContext.Data.LoadFromEnumerable(trainingData);
            return trainingPipeline.Fit(trainingDataView);
        }
    }
}
