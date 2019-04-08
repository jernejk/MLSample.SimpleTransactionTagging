using Microsoft.ML;
using Microsoft.ML.Data;
using System.IO;

namespace MLSample.TransactionTagging
{
    public class BankTransactionLabelService
    {
        private readonly MLContext _mlContext;
        private ITransformer _loadedModel;
        private PredictionEngine<TransactionData, TransactionPrediction> _predEngine;

        public BankTransactionLabelService()
        {
            _mlContext = new MLContext(seed: 0);
        }

        public void LoadModel(string modelPath)
        {
            using (var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                _loadedModel = _mlContext.Model.Load(stream);
            }

            _predEngine = _loadedModel.CreatePredictionEngine<TransactionData, TransactionPrediction>(_mlContext);
        }

        public string PredictTag(TransactionData transaction)
        {
            var prediction = new TransactionPrediction();
            _predEngine.Predict(transaction, ref prediction);

            return prediction?.Area;
        }

        public class TransactionPrediction
        {
            [ColumnName("PredictedLabel")]
            public string Area;
        }
    }
}
