/// <summary>
/// Generate fake financial data used by ML.NET Model Builder for experimentation.
/// 
/// You can control the number of unique companies, number of generated transactions and percentage of ambigious/conflicting transactions.
/// 
/// A company will have their own transaction format, some are static ("CREDIT CARD {{COMPANY-NAME}}") or dynamic ("{{COMPANY-NAME}} {{DYNAMIC-NUMBERS}}").
/// </summary>
Faker faker = new("en");
Random random = new();

int numOfCompanies = 20;
int numOfTransactions = 1000;
double conflictPercentage = 0.15;
var genTestData = GenerateInitialData(numOfCompanies, conflictPercentage);
var trannsactions = GenerateTransactions(genTestData, numOfTransactions);

foreach (var transaction in trannsactions)
{
    Console.WriteLine(transaction);
}

// Manually generate CSV and escape " character.
var csvLines = trannsactions
    .Select(x => $"\"{x.TransactionName.Replace("\"", "\\\"")}\",\"{x.Category}\"")
    .ToList();

// Add headers for the CSV.
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

List<TransactionGenData> GenerateInitialData(int number, double chanceOfContradiction)
{
    Dictionary<string, int> addedCompanies = new();

    List<TransactionGenData> list = new();
    for (int i = 0; i < number; ++i)
    {
        if (i > 0 && chanceOfContradiction > random.NextDouble())
        {
            var randomTransaction = faker.PickRandom(list);

            TransactionGenData conflictingData = new()
            {
                CompanyName = randomTransaction.CompanyName,
                Category = faker.Commerce.Categories(1).FirstOrDefault(),
                TransactionFormatType = randomTransaction.TransactionFormatType,
            };

            list.Add(conflictingData);
            continue;
        }

        string company = faker.Company.CompanyName("{{name.lastName}}");
        if (addedCompanies.ContainsKey(company))
        {
            company = faker.Company.CompanyName("{{name.lastName}}");

            if (addedCompanies.ContainsKey(company))
            {
                // In case it's still not unique
                company = Guid.NewGuid().ToString();
            }
        }

        TransactionGenData data = new()
        {
            CompanyName = company,
            Category = faker.Commerce.Categories(1).FirstOrDefault(),
            TransactionFormatType = faker.PickRandom<TransactionFormatType>()
        };

        list.Add(data);
    }

    return list;
}

List<TransactionData> GenerateTransactions(List<TransactionGenData> testData, int number)
{
    List<TransactionData> list = new();
    for (int i = 0; i < number; ++i)
    {
        var genData = faker.PickRandom(testData);
        list.Add(new TransactionData
        {
            TransactionName = GenerateTransactionName(genData),
            Category = genData.Category
        });
    }

    return list;
}

string GenerateTransactionName(TransactionGenData data)
    => data.TransactionFormatType switch
    {
        TransactionFormatType.CompanyName
            => data.CompanyName,
        TransactionFormatType.CompanyNameWithTransferNumber
            => $"{data.CompanyName} {faker.Finance.RoutingNumber()}",
        TransactionFormatType.CardTypeCompanyName
            => $"{faker.PickRandom("CREDIT", "DEBIT", "TRANSFER")} {data.CompanyName}",
        TransactionFormatType.CardTypeCompanyNameWithTransferNumber
            => $"{faker.PickRandom("CREDIT", "DEBIT", "TRANSFER")} {data.CompanyName} {faker.Finance.RoutingNumber()}",
        TransactionFormatType.PayPal
            => $"PAYPAL *{data.CompanyName} {faker.Finance.RoutingNumber()}",
        TransactionFormatType.Payment
            => $"Payment to {data.CompanyName} Ref {faker.Finance.RoutingNumber()}",
        _ => $"{data.CompanyName} Ca Pos Authorisation",
    };
