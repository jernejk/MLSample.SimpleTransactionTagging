using Microsoft.ML;
using Microsoft.ML.Data;
using MLSample.TransactionTagging.Core.Models;
using System;
using System.Linq;
using System.Collections.Generic;
using System.IO;

namespace MLSample.TransactionTagging.Core
{
    public class BankTransactionLabelService
    {
        private readonly MLContext _mlContext;
        private PredictionEngine<Transaction, TransactionPrediction> _predEngine;
        private List<string> _categories;

        public BankTransactionLabelService(MLContext mlContext)
        {
            _mlContext = mlContext;
        }

        public void LoadModelFromFile(string modelPath)
        {
            _categories = null;

            // Load model from file.
            ITransformer loadedModel;
            using (var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
                loadedModel = _mlContext.Model.Load(stream, out var modelInputSchema);

            _predEngine = _mlContext.Model.CreatePredictionEngine<Transaction, TransactionPrediction>(loadedModel);
        }

        public void LoadModelFromStream(Stream modelStream)
        {
            _categories = null;

            // Load model from file.
            ITransformer loadedModel = _mlContext.Model.Load(modelStream, out var modelInputSchema);
            _predEngine = _mlContext.Model.CreatePredictionEngine<Transaction, TransactionPrediction>(loadedModel);
        }

        public void LoadModel(ITransformer mlModel)
        {
            _categories = null;

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

        public List<string> GetCategories()
        {
            if (_categories != null)
            {
                return _categories;
            }

            // Based on https://github.com/dotnet/docs/issues/14265
            var column = _predEngine.OutputSchema.GetColumnOrNull("Score");

            var slotNames = new VBuffer<ReadOnlyMemory<char>>();
            column.Value.GetSlotNames(ref slotNames);
            var names = new string[slotNames.Length];

            _categories = slotNames
                .DenseValues()
                .Select(x => x.ToString())
                .ToList();

            return _categories;
        }

        public static Dictionary<string, float> GetScoresWithLabelsSorted(DataViewSchema schema, string name, float[] scores)
        {
            // Based on https://github.com/dotnet/docs/issues/14265
            Dictionary<string, float> result = new Dictionary<string, float>();

            var column = schema.GetColumnOrNull(name);

            var slotNames = new VBuffer<ReadOnlyMemory<char>>();
            column.Value.GetSlotNames(ref slotNames);
            var names = new string[slotNames.Length];
            var num = 0;
            foreach (var denseValue in slotNames.DenseValues())
            {
                result.Add(denseValue.ToString(), scores[num++]);
            }

            return result.OrderByDescending(c => c.Value).ToDictionary(i => i.Key, i => i.Value);
        }
    }
}
