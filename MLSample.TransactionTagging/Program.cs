using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;

namespace MLSample.TransactionTagging
{
    public static class Program
    {
        public static void Main(string[] args)
        {
            // Some manually chosen 
            var trainingData = JsonConvert.DeserializeObject<List<TransactionData>>(File.ReadAllText("training.json"));

            var trainingService = new BankTransactionTrainingService();
            trainingService.Train(trainingData, "Model.zip");

            var labelService = new BankTransactionLabelService();
            labelService.LoadModel("Model.zip");

            MakePrediction(labelService, "AMERICAN CONCEPTS PT BRISBANE", "expense");
            MakePrediction(labelService, "ANZ M-BANKING PAYMENT TRANSFER 513542 TO SPIRE REALITY", "expense");
            MakePrediction(labelService, "SPIRE REALITY", "expense");
            MakePrediction(labelService, "SPIRE REALITY", "income");
        }

        private static void MakePrediction(BankTransactionLabelService labelService, string description, string transactionType)
        {
            string prediction = labelService.PredictTag(new TransactionData
            {
                Description = description,
                TransactionType = transactionType
            });

            Console.WriteLine($"{description} ({transactionType}) => {prediction}");
        }

        //private static string UpdateDescription(string originalDescription)
        //{
        //    originalDescription = originalDescription
        //        .Replace("POS AUTHORISATION ", string.Empty)
        //        .Replace("VISA DEBIT PURCHASE CARD 0082 ", string.Empty)
        //        .Replace("VISA DEBIT PURCHASE CARD 0068 ", string.Empty)
        //        .Replace(" Card Used 9926", string.Empty)
        //        .Replace(" Card Used 0082", string.Empty)
        //        //.Replace("ANZ M-BANKING PAYMENT ", string.Empty)
        //        .Replace("EFTPOS ", string.Empty)
        //        .Replace("PAYPAL *", string.Empty);

        //    if (originalDescription.EndsWith("AU"))
        //    {
        //        originalDescription = originalDescription
        //            .Substring(0, originalDescription.LastIndexOf("AU"))
        //            .Trim();
        //    }

        //    return originalDescription;
        //}
    }
}
