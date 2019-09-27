FROM mcr.microsoft.com/dotnet/core/aspnet:3.0-buster-slim AS base
WORKDIR /app
EXPOSE 80
EXPOSE 443

FROM mcr.microsoft.com/dotnet/core/sdk:3.0-buster AS build
WORKDIR /src
COPY ["MLSample.TransactionTagging.Blazor/MLSample.TransactionTagging.Blazor.csproj", "MLSample.TransactionTagging.Blazor/"]
RUN dotnet restore "MLSample.TransactionTagging.Blazor/MLSample.TransactionTagging.Blazor.csproj"
COPY . .
WORKDIR "/src/MLSample.TransactionTagging.Blazor"
RUN dotnet build "MLSample.TransactionTagging.Blazor.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "MLSample.TransactionTagging.Blazor.csproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "MLSample.TransactionTagging.Blazor.dll"]