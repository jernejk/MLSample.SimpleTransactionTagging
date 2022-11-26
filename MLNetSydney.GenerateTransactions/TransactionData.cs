namespace MLNetSydney.GenerateTransactions;

internal class TransactionData
{
    public string TransactionName { get; set; }
    public string Category { get; set; }

    public override string ToString() => $"{TransactionName} - {Category}";
}
