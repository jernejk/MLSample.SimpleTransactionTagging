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
            var pipeline = LoadDataIntoPipeline(mlContext);
            var trainingPipeline = GetTrainingPipeline(mlContext, pipeline);
            var trainingModel = BuildAndTrainModel(mlContext, trainingPipeline, trainingData);

            using (var fs = new FileStream(modelSavePath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(trainingModel, fs);
        }

        private IEstimator<ITransformer> LoadDataIntoPipeline(MLContext mlContext)
        {
            // STEP 2: Common data process configuration with pipeline data transformations
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: nameof(TransactionData.Category), outputColumnName: "Label")
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(TransactionData.Description), outputColumnName: "TitleFeaturized"))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(TransactionData.TransactionType), outputColumnName: "DescriptionFeaturized"))
                .Append(mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                // Sample Caching the DataView so estimators iterating over the data multiple times, instead of always reading from file, using the cache might get better performance.
                .AppendCacheCheckpoint(mlContext);

            return pipeline;
        }

        private IEstimator<ITransformer> GetTrainingPipeline(MLContext mlContext, IEstimator<ITransformer> pipeline)
        {
            // STEP 3: Create the training algorithm/trainer
            // Use the multi-class SDCA algorithm to predict the label using features.
            // Set the trainer/algorithm and map label to value (original readable state)
            return pipeline
                .Append(mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(DefaultColumnNames.Label, DefaultColumnNames.Features))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
        }

        private ITransformer BuildAndTrainModel(MLContext mlContext, IEstimator<ITransformer> trainingPipeline, IEnumerable<TransactionData> trainingData)
        {
            //// We can now examine the records in the IDataView. We first create an enumerable of rows in the IDataView.
            var trainingDataView = mlContext.Data.LoadFromEnumerable(trainingData);
            return trainingPipeline.Fit(trainingDataView);
        }
    }
}
