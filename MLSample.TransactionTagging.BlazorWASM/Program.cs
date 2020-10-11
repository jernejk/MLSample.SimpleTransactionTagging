using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using MLSample.TransactionTagging.Core;
using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;

namespace MLSample.TransactionTagging.BlazorWASM
{
    public class Program
    {
        public static async Task Main(string[] args)
        {
            var builder = WebAssemblyHostBuilder.CreateDefault(args);
            builder.RootComponents.Add<App>("#app");

            builder.Services.AddScoped(sp => new HttpClient { BaseAddress = new Uri(builder.HostEnvironment.BaseAddress) });

            builder.Services.AddSingleton<MLContext>();
            builder.Services.AddTransient<BankTransactionLabelService>(
                ctx =>
                {
                    // Prediction engine in BankTransactionLabelService should be transient as it is not thread safe.
                    var mlContext = ctx.GetService<MLContext>();
                    return new BankTransactionLabelService(mlContext);
                });

            builder.Services.AddSingleton<ExplainClassificationService>(
                ctx =>
                {
                    // Load data and extract categories, so we can explain how certain ML was.
                    // We only use it to have distinct categories from training data in exact order they appear.
                    var explainClassificationService = new ExplainClassificationService();
                    return explainClassificationService;
                });

            await builder.Build().RunAsync();
        }
    }
}
