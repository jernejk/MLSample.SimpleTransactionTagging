using Microsoft.ML.Data;

namespace MLSample.TransactionTagging
{
    public class TransactionPrediction
    {
        [ColumnName("PredictedLabel")]
        public string Category;
    }
}
