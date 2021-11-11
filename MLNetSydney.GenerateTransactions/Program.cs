using Bogus;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace MLNetSydney.GenerateTransactions
{
    /// <summary>
    /// Generate fake financial data used by ML.NET Model Builder for experimentation.
    /// 
    /// You can control the number of unique companies, number of generated transactions and percentage of ambigious/conflicting transactions.
    /// 
    /// A company will have their own transaction format, some are static ("CREDIT CARD {{COMPANY-NAME}}") or dynamic ("{{COMPANY-NAME}} {{DYNAMIC-NUMBERS}}").
    /// </summary>
    internal class Program
    {
        private static Faker _faker = new Faker("en");
        private static Random _random = new Random();

        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            int numOfCompanies = 20;
            int numOfTransactions = 1000;
            double conflictPercentage = 0.15;
            var genTestData = GenerateInitialData(numOfCompanies, conflictPercentage);
            var trannsactions = GenerateTransactions(genTestData, numOfTransactions);

            foreach (var transaction in trannsactions)
            {
                Console.WriteLine(transaction);
            }

            var csvLines = trannsactions
                .Select(x => $"\"{x.TransactionName.Replace("\"","\\\"")}\",\"{x.Category}\"")
                .ToList();

            csvLines.Insert(0, "\"Transaction Name\", \"Category\"");

            string csvFileName = $"transactions-{numOfCompanies}-{numOfTransactions}-{conflictPercentage}.csv";
            File.WriteAllLines(csvFileName, csvLines);

            /*
             * To use ML.NET Model:
             * 1. Run this console application to generate CSV
             * 2. Open TransactionCategory.mbconfig
             * 3. Open Data tab, update file path
             * 4. Click "Advanced data options..."
             *    4.1 Data formatting
             *    4.2 Does your data contains headers?
             *        Yes
             * 5. Column to predict (Label):
             *    Category
             * 6. Next step (moving to Train tab)
             * 7. Train it for 10+ seconds
             * 8. Click Next Step and you can play around with the model
             * 9. Click next and you can copy code similar as below which you can use in your projects (you'll need to copy the code nested in TransactionCategory.mbconfig)
             * 
            var sampleData = new TransactionCategory.ModelInput()
            {
                Transaction_Name = @"CREDIT Koepp",
            };

            //Load model and predict output
            var result = TransactionCategory.Predict(sampleData);
            */
        }

        public static List<TransactionGenData> GenerateInitialData(int number, double chanceOfContradiction)
        {
            Dictionary<string, int> addedCompanies = new Dictionary<string, int>();

            List<TransactionGenData> list = new List<TransactionGenData>();
            for (int i = 0; i < number; ++i)
            {
                if (i > 0 && chanceOfContradiction > _random.NextDouble())
                {
                    var randomTransaction = _faker.PickRandom(list);

                    TransactionGenData conflictingData = new TransactionGenData
                    {
                        CompanyName = randomTransaction.CompanyName,
                        Category = _faker.Commerce.Categories(1).FirstOrDefault(),
                        TransactionFormatType = randomTransaction.TransactionFormatType,
                    };

                    list.Add(conflictingData);
                    continue;
                }

                string company = _faker.Company.CompanyName("{{name.lastName}}");
                if (addedCompanies.ContainsKey(company))
                {
                    company = _faker.Company.CompanyName("{{name.lastName}}");

                    if (addedCompanies.ContainsKey(company))
                    {
                        // In case it's still not unique
                        company = Guid.NewGuid().ToString();
                    }
                }

                TransactionGenData data = new TransactionGenData
                {
                    CompanyName = company,
                    Category = _faker.Commerce.Categories(1).FirstOrDefault(),
                    TransactionFormatType = _faker.PickRandom<TransactionFormatType>()
                };

                list.Add(data);
            }

            return list;
        }

        public static List<TransactionData> GenerateTransactions(List<TransactionGenData> testData, int number)
        {
            List<TransactionData> list = new List<TransactionData>();
            for (int i = 0; i < number; ++i)
            {
                var genData = _faker.PickRandom(testData);
                list.Add(new TransactionData
                {
                    TransactionName = GenerateTransactionName(genData),
                    Category = genData.Category
                });
            }

            return list;
        }

        public static string GenerateTransactionName(TransactionGenData data)
        {
            switch (data.TransactionFormatType)
            {
                case TransactionFormatType.CompanyName:
                    return data.CompanyName;
                case TransactionFormatType.CompanyNameWithTransferNumber:
                    return $"{data.CompanyName} {_faker.Finance.RoutingNumber()}";
                case TransactionFormatType.CardTypeCompanyName:
                    return $"{_faker.PickRandom("CREDIT", "DEBIT", "TRANSFER")} {data.CompanyName}";
                case TransactionFormatType.CardTypeCompanyNameWithTransferNumber:
                    return $"{_faker.PickRandom("CREDIT", "DEBIT", "TRANSFER")} {data.CompanyName} {_faker.Finance.RoutingNumber()}";
                case TransactionFormatType.PayPal:
                    return $"PAYPAL *{data.CompanyName} {_faker.Finance.RoutingNumber()}";
                case TransactionFormatType.Payment:
                    return $"Payment to {data.CompanyName} Ref {_faker.Finance.RoutingNumber()}";
                case TransactionFormatType.PosAuthorization:
                default:
                    return $"{data.CompanyName} Ca Pos Authorisation";
                       
            }
        }
    }
}
