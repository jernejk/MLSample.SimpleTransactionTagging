using Microsoft.ML;
using MLSample.TransactionTagging.Core.Models;
using System.Collections.Generic;
using System.Linq;

namespace MLSample.TransactionTagging.Core
{
    public class BankTransactionTrainingService
    {
        private readonly MLContext _mlContext;
        private ITransformer _model;
        private IDataView _trainingDataView;

        public BankTransactionTrainingService(MLContext mlContext)
        {
            _mlContext = mlContext;
        }

        public ITransformer Train(IEnumerable<Transaction> trainingData)
        {
            // Configure ML pipeline
            var pipeline = LoadDataProcessPipeline(_mlContext);
            var trainingPipeline = GetTrainingPipeline(_mlContext, pipeline);
            _trainingDataView = _mlContext.Data.LoadFromEnumerable(trainingData);

            // Generate training model.
            _model = trainingPipeline.Fit(_trainingDataView);

            return _model;
        }

        public void SaveModel(string modelSavePath)
        {
            // Save training model to disk.
            _mlContext.Model.Save(_model, _trainingDataView.Schema, modelSavePath);
        }

        private IEstimator<ITransformer> LoadDataProcessPipeline(MLContext mlContext)
        {
            // Configure data pipeline based on the features in TransactionData.
            // Description and TransactionType are the inputs and Category is the expected result.
            var dataProcessPipeline = mlContext
                .Transforms.Conversion.MapValueToKey(inputColumnName: nameof(Transaction.Category), outputColumnName: "Label")
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: nameof(Transaction.Description), outputColumnName: "Features"))
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
