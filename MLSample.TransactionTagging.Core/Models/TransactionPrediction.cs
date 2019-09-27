using Microsoft.ML.Data;

namespace MLSample.TransactionTagging.Core.Models
{
    public class TransactionPrediction
    {
        [ColumnName("PredictedLabel")]
        public string Category { get; set; }

        public float[] Score { get; set; }
    }
}
