using System.Runtime.Serialization;

namespace MLSample.TransactionTagging.Core.Models
{
    [DataContract]
    public class Transaction
    {
        [DataMember(Name = "desc")]
        public string Description { get; set; }

        [DataMember(Name = "category")]
        public string Category { get; set; }
    }
}
