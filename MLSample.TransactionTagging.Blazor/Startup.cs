using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.ML;
using MLSample.TransactionTagging.Core;
using MLSample.TransactionTagging.Core.Models;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;

namespace MLSample.TransactionTagging.Blazor
{
    public class Startup
    {
        public Startup(IConfiguration configuration)
        {
            Configuration = configuration;
        }

        public IConfiguration Configuration { get; }

        // This method gets called by the runtime. Use this method to add services to the container.
        // For more information on how to configure your application, visit https://go.microsoft.com/fwlink/?LinkID=398940
        public void ConfigureServices(IServiceCollection services)
        {
            services.AddRazorPages();
            services.AddServerSideBlazor();

            services.AddSingleton<MLContext>();
            services.AddTransient<BankTransactionTrainingService>();
            services.AddSingleton<ITransformer>((ctx) =>
                {
                    // Based on https://devblogs.microsoft.com/cesardelatorre/how-to-optimize-and-run-ml-net-models-on-scalable-asp-net-core-webapis-or-web-apps/
                    // Load data and train model once per lifetime of the application.
                    // MLContext and the ML Model should be singleton.
                    var mlContext = ctx.GetService<MLContext>();
                    var trainingService = new BankTransactionTrainingService(mlContext);

                    string path = Path.Combine(AppContext.BaseDirectory, "Data/training.json");
                    var data = JsonConvert.DeserializeObject<List<Transaction>>(File.ReadAllText(path));
                    var mlModel = trainingService.ManualTrain(data);

                    return mlModel;
                });
            services.AddTransient<BankTransactionLabelService>(
                ctx =>
                {
                    // Prediction engine in BankTransactionLabelService should be transient as it is not thread safe.
                    var mlContext = ctx.GetService<MLContext>();
                    var mlModel = ctx.GetService<ITransformer>();

                    var labelService = new BankTransactionLabelService(mlContext);
                    labelService.LoadModel(mlModel);
                    return labelService;
                });

            services.AddSingleton<ExplainClassificationService>(
                ctx =>
                {
                    // Load data and extract categories, so we can explain how certain ML was.
                    // We only use it to have distinct categories from training data in exact order they appear.
                    var explainClassificationService = new ExplainClassificationService();
                    string path = Path.Combine(AppContext.BaseDirectory, "Data/training.json");
                    var data = JsonConvert.DeserializeObject<List<Transaction>>(File.ReadAllText(path));
                    explainClassificationService.LoadCategories(data);

                    return explainClassificationService;
                });
        }

        // This method gets called by the runtime. Use this method to configure the HTTP request pipeline.
        public void Configure(
            IApplicationBuilder app,
            IWebHostEnvironment env,
            // We initialize it here, so that we can use as soon as the server is ready.
            BankTransactionLabelService bankTransactionLabelService,
            ExplainClassificationService explainClassificationService)
        {
            if (env.IsDevelopment())
            {
                app.UseDeveloperExceptionPage();
            }
            else
            {
                app.UseExceptionHandler("/Error");
                // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
                app.UseHsts();
            }

            app.UseHttpsRedirection();
            app.UseStaticFiles();

            app.UseRouting();

            app.UseEndpoints(endpoints =>
            {
                endpoints.MapBlazorHub();
                endpoints.MapFallbackToPage("/_Host");
            });
        }
    }
}
