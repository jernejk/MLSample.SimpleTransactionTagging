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
            // Some manually chosen transactions with some modifications.
            Console.WriteLine("Loading training data...");
            List<TransactionData> trainingData = GetTrainingData();

            Console.WriteLine("Training the model...");
            var trainingService = new BankTransactionTrainingService();
            trainingService.Train(trainingData, "Model.zip");

            Console.WriteLine("Prepare transaction labeler...");
            var labelService = new BankTransactionLabelService();
            labelService.LoadModel("Model.zip");

            Console.WriteLine("Predict some transactions based on their description and type...");
            Console.WriteLine();

            // Should be "coffee & tea".
            MakePrediction(labelService, "AMERICAN CONCEPTS PT BRISBANE", "expense");

            // The number in the transaction is always random but it will work despite that. Result: rent
            MakePrediction(labelService, "ANZ M-BANKING PAYMENT TRANSFER 513542 TO SPIRE REALITY", "expense");

            // In fact, searching just for part of the transaction will give us the same result.
            MakePrediction(labelService, "SPIRE REALITY", "expense");

            // If we change the transaction type, we'll get a reimbursement instead.
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

        private static List<TransactionData> GetTrainingData()
        {
            return JsonConvert.DeserializeObject<List<TransactionData>>(File.ReadAllText("training.json"));
        }
    }
}
