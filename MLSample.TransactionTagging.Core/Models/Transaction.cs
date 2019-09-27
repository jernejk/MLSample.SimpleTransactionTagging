using System.Runtime.Serialization;

namespace MLSample.TransactionTagging
{
    [DataContract]
    public class Transaction
    {
        [DataMember(Name = "id")]
        public string ID { get; set; }

        [DataMember(Name = "desc")]
        public string Description { get; set; }

        [DataMember(Name = "category")]
        public string Category { get; set; }

        [DataMember(Name = "transactionType")]
        public string TransactionType { get; set; }
    }
}
