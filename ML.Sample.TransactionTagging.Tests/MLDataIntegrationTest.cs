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
            var mlModel = trainingService.Train(trainingData);

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
            trainingService.Train(trainingData);
            trainingService.SaveModel(modelFile);


            var labelService = new BankTransactionLabelService(mlContext);
            labelService.LoadModelFromFile(modelFile);

            TestModel(labelService);

            File.Delete(modelFile);
        }

        private static void TestModel(BankTransactionLabelService labelService)
        {
            labelService.PredictCategory(new Transaction("coffee")).Should().Be("coffee & tea");
            labelService.PredictCategory(new Transaction("DotNetFoundation.org")).Should().Be("investment");
            labelService.PredictCategory(new Transaction("Fitness")).Should().Be("health");
            labelService.PredictCategory(new Transaction("Uber")).Should().Be("transport");
            labelService.PredictCategory(new Transaction("PubConf")).Should().Be("conference");
        }
    }
}
