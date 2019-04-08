using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace MLSample.TransactionTagging
{
    public class BankTransactionTrainingService
    {
        private readonly MLContext _mlContext;
        private readonly string _modelSavePath;

        public BankTransactionTrainingService(string modelSavePath)
        {
            _mlContext = new MLContext(seed: 0);
            _modelSavePath = modelSavePath;
        }

        public void Train(IEnumerable<TransactionData> trainingData)
        {
            var pipeline = ProcessData();
            BuildAndTrainModel(trainingData, pipeline);
        }

        private IEstimator<ITransformer> ProcessData()
        {
            // STEP 2: Common data process configuration with pipeline data transformations
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: nameof(TransactionData.Category), outputColumnName: "Label")
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(TransactionData.Description), outputColumnName: "TitleFeaturized"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(TransactionData.TransactionType), outputColumnName: "DescriptionFeaturized"))
                .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"))
                // Sample Caching the DataView so estimators iterating over the data multiple times, instead of always reading from file, using the cache might get better performance.
                .AppendCacheCheckpoint(_mlContext);

            return pipeline;
        }

        private IEstimator<ITransformer> BuildAndTrainModel(IEnumerable<TransactionData> trainingData, IEstimator<ITransformer> pipeline)
        {
            // STEP 3: Create the training algorithm/trainer
            // Use the multi-class SDCA algorithm to predict the label using features.
            // Set the trainer/algorithm and map label to value (original readable state)
            var trainingPipeline = pipeline
                .Append(_mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(DefaultColumnNames.Label, DefaultColumnNames.Features))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            //// We can now examine the records in the IDataView. We first create an enumerable of rows in the IDataView.
            var trainingDataView = _mlContext.Data.LoadFromEnumerable(trainingData);
            var trainingModel = trainingPipeline.Fit(trainingDataView);

            // Save the new model to .ZIP file
            using (var fs = new FileStream(_modelSavePath, FileMode.Create, FileAccess.Write, FileShare.Write))
                _mlContext.Model.Save(trainingModel, fs);

            return trainingPipeline;
        }
    }
}
