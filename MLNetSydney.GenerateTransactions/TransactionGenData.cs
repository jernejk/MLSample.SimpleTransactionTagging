namespace MLNetSydney.GenerateTransactions
{
    internal class TransactionGenData
    {
        public TransactionGenData() { }
        public TransactionGenData(string companyName, string category)
        {
            CompanyName = companyName;
            Category = category;
        }

        public string CompanyName { get; set; }
        public string Category { get; set; }
        public TransactionFormatType TransactionFormatType { get; set; }

        public override string ToString()
        {
            return $"{CompanyName} - {Category} - {TransactionFormatType}";
        }
    }

    public enum TransactionFormatType
    {
        CompanyName,
        CompanyNameWithTransferNumber,
        CardTypeCompanyName,
        CardTypeCompanyNameWithTransferNumber,
        PayPal,
        Payment,
        PosAuthorization
    }
}
