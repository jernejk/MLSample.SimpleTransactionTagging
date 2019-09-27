using Microsoft.ML;
using System.IO;

namespace MLSample.TransactionTagging
{
    public class BankTransactionLabelService
    {
        private readonly MLContext _mlContext;
        private PredictionEngine<Transaction, TransactionPrediction> _predEngine;

        public BankTransactionLabelService()
        {
            _mlContext = new MLContext(seed: 0);
        }

        public void LoadModel(string modelPath)
        {
            ITransformer loadedModel;
            using (var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
                loadedModel = _mlContext.Model.Load(stream, out var modelInputSchema);
            _predEngine = _mlContext.Model.CreatePredictionEngine<Transaction, TransactionPrediction>(loadedModel);
        }

        public string PredictCategory(Transaction transaction)
        {
            var prediction = new TransactionPrediction();
            _predEngine.Predict(transaction, ref prediction);
            return prediction?.Category;
        }
    }
}
