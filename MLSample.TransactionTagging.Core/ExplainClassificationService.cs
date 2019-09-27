using MLSample.TransactionTagging.Core.Models;
using System.Collections.Generic;
using System.Linq;

namespace MLSample.TransactionTagging.Core
{
    public class ExplainClassificationService
    {
        public void LoadCategories(List<Transaction> transactions)
        {
            Categories = transactions.Select(t => t.Category).Distinct().ToList();
        }

        public IEnumerable<string> Categories { get; private set; }
    }
}
