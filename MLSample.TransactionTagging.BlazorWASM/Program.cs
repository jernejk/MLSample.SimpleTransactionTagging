using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML;
using MLSample.TransactionTagging.Core;
using System;
using System.Net.Http;
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
                    // Also, PredictionEnginePool doesn't seem to be supported in Blazor WASM atm.
                    var mlContext = ctx.GetService<MLContext>();
                   return new BankTransactionLabelService(mlContext);
               });

            await builder.Build().RunAsync();
        }
    }
}
