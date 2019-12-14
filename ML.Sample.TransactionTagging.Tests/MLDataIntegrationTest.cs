using FluentAssertions;
using Microsoft.ML;
using MLSample.TransactionTagging.Core;
using MLSample.TransactionTagging.Core.Models;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using Xunit;

namespace ML.Sample.TransactionTagging.Tests
{
    public class MLDataIntegrationTest
    {
        [Fact]
        public void TestCommonDemoCases()
        {
            var mlContext = new MLContext(0);
            var trainingService = new BankTransactionTrainingService(mlContext);

            string trainingDataFile = Path.Combine(AppContext.BaseDirectory, "Data/training.json");
            var trainingData = JsonConvert.DeserializeObject<List<Transaction>>(File.ReadAllText(trainingDataFile));
            var mlModel = trainingService.ManualTrain(trainingData);

            var labelService = new BankTransactionLabelService(mlContext);
            labelService.LoadModel(mlModel);

            TestModel(labelService);
        }

        [Fact]
        public void TestLoadingMLModel()
        {
            var mlContext = new MLContext(0);
            var trainingService = new BankTransactionTrainingService(mlContext);

            string modelFile = Path.Combine(AppContext.BaseDirectory, $"{Guid.NewGuid()}.zip");
            string trainingDataFile = Path.Combine(AppContext.BaseDirectory, "Data/training.json");
            var trainingData = JsonConvert.DeserializeObject<List<Transaction>>(File.ReadAllText(trainingDataFile));

            var model = trainingService.ManualTrain(trainingData);
            trainingService.SaveModel(modelFile, model);

            var labelService = new BankTransactionLabelService(mlContext);
            labelService.LoadModelFromFile(modelFile);

            TestModel(labelService);

            File.Delete(modelFile);
        }

        [Fact]
        public void TestAutoTrain()
        {
            var mlContext = new MLContext(0);
            var trainingService = new BankTransactionTrainingService(mlContext);

            string trainingDataFile = Path.Combine(AppContext.BaseDirectory, "Data/training.json");
            var trainingData = JsonConvert.DeserializeObject<List<Transaction>>(File.ReadAllText(trainingDataFile));
            var mlModel = trainingService.AutoTrain(trainingData, 5);

            var labelService = new BankTransactionLabelService(mlContext);
            labelService.LoadModel(mlModel);
            
            TestModel(labelService);
        }

        private static void TestModel(BankTransactionLabelService labelService)
        {
            // Exact matches from the training data.
            labelService.PredictCategory(new Transaction("VISA DEBIT PURCHASE CARD 0012 AMERICAN CONCEPTS PT BRISBANE")).Should().Be("coffee & tea");
            labelService.PredictCategory(new Transaction("VISA DEBIT PURCHASE CARD 0012 DOTNETFOUNDATION.ORG 42553885334 10.00 USD INC O/S FEE $0.42")).Should().Be("investment");

            // Not exact matches, ML Model needs to be to do "Fuzzy" search on them.
            labelService.PredictCategory(new Transaction("coffee")).Should().Be("coffee & tea");
            labelService.PredictCategory(new Transaction("DotNetFoundation.org")).Should().Be("investment");
            labelService.PredictCategory(new Transaction("Anytime Fitness")).Should().Be("health");
            
            // TODO: Doesn't work for AutoML. Data seems to be too unbalanced and biased toward conferences.
            //labelService.PredictCategory(new Transaction("UBER")).Should().Be("transport");
            labelService.PredictCategory(new Transaction("PubConf")).Should().Be("conference");
            labelService.PredictCategory(new Transaction("DDD")).Should().Be("conference");
            labelService.PredictCategory(new Transaction("DDD Perth")).Should().Be("conference");
        }
    }
}
