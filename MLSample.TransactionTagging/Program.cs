using Microsoft.ML;
using MLSample.TransactionTagging.Core;
using MLSample.TransactionTagging.Core.Models;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

namespace MLSample.TransactionTagging
{
    public static class Program
    {
        public static void Main(string[] args)
        {
            var mlContex = new MLContext();

            // Training is optional as long it's done at least once.
            if (args.Length == 0 || !args[0].Equals("no-training", StringComparison.OrdinalIgnoreCase))
            {
                string trainingDataFile = Path.Combine(AppContext.BaseDirectory, "Data/training.json");

                // Some manually chosen transactions with some modifications.
                Console.WriteLine("Loading training data...");
                List<Transaction> trainingData = GetTrainingData(trainingDataFile);

                Console.WriteLine("Training the model...");
                var trainingService = new BankTransactionTrainingService(mlContex);

                var timer = Stopwatch.StartNew();
                trainingService.Train(trainingData);
                trainingService.SaveModel("Model.zip");
                timer.Stop();

                Console.WriteLine($"Training done in {Math.Round(timer.Elapsed.TotalSeconds, 2)} seconds");
                Console.WriteLine();
            }

            Console.WriteLine("Prepare transaction labeler...");
            string modelFile = Path.Combine(AppContext.BaseDirectory, "Model.zip");
            var labelService = new BankTransactionLabelService(mlContex);
            labelService.LoadModelFromFile(modelFile);

            Console.WriteLine("Predict some transactions based on their description and type...");
            Console.WriteLine();

            // Should be "coffee & tea".
            MakePrediction(labelService, "VISA DEBIT PURCHASE CARD 0012 AMERICAN CONCEPTS PT BRISBANE");

            // Should be "coffee & tea".
            MakePrediction(labelService, "AMERICAN CONCEPTS PT BRISBANE");

            // The number in the transaction is always random but it will work despite that. Result: rent
            MakePrediction(labelService, "ANZ M-BANKING PAYMENT TRANSFER 513542 TO SPIRE REALITY");

            // In fact, searching just for part of the transaction will give us the same result.
            MakePrediction(labelService, "SPIRE REALITY");

            // Should be "investment".
            MakePrediction(labelService, "VISA DEBIT PURCHASE CARD 0012 DOTNETFOUNDATION.ORG 42553885334 10.00 USD INC O/S FEE $0.42");

            // Should be "investment".
            MakePrediction(labelService, "VISA DEBIT PURCHASE CARD 0012 DOTNETFOUNDATION.ORG 334634543 10.00 USD INC O/S FEE $0.12");

            // Will likely fail.
            MakePrediction(labelService, "DOTNETFOUNDATION.ORG random text");
        }

        private static void MakePrediction(BankTransactionLabelService labelService, string description)
        {
            string prediction = labelService.PredictCategory(new Transaction
            {
                Description = description,
            });

            Console.WriteLine($"{description}\n => {prediction}\n");
        }

        private static List<Transaction> GetTrainingData(string trainingDataFile)
        {
            return JsonConvert.DeserializeObject<List<Transaction>>(File.ReadAllText(trainingDataFile));
        }
    }
}
