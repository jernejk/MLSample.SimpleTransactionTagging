using Microsoft.ML;
using MLSample.TransactionTagging.Core.Models;
using System.IO;

namespace MLSample.TransactionTagging.Core
{
    public class BankTransactionLabelService
    {
        private readonly MLContext _mlContext;
        private PredictionEngine<Transaction, TransactionPrediction> _predEngine;

        public BankTransactionLabelService(MLContext mlContext)
        {
            _mlContext = mlContext;
        }

        public void LoadModelFromFile(string modelPath)
        {
            // Load model from file.
            ITransformer loadedModel;
            using (var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
                loadedModel = _mlContext.Model.Load(stream, out var modelInputSchema);

            _predEngine = _mlContext.Model.CreatePredictionEngine<Transaction, TransactionPrediction>(loadedModel);
        }

        public void LoadModel(ITransformer mlModel)
        {
            // Load already loaded model.
            _predEngine = _mlContext.Model.CreatePredictionEngine<Transaction, TransactionPrediction>(mlModel);
        }

        public string PredictCategory(Transaction transaction)
        {
            var prediction = Predict(transaction);
            return prediction?.Category;
        }

        public TransactionPrediction Predict(Transaction transaction)
        {
            return _predEngine.Predict(transaction);
        }
    }
}
