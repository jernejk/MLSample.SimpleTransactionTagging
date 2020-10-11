$pathToSolution = "./"
$projectName = "MLSample.TransactionTagging.BlazorWASM"
$repoName = "MLSample.SimpleTransactionTagging"

Write-Output "----==== Publish $pathToSolution"
dotnet publish $pathToSolution -c Release -o ./dist/
Write-Output ""

Write-Output "----==== Copy from ./dist/wwwroot"
Copy-Item -Path "./dist/wwwroot/*" -Destination "./" -Recurse -Force
Write-Output ""

$indexFile = "./index.html"
$originalBaseUrlText = "<base href=""/"">";
$targetBaseUrlText = "<base href=""/$repoName/"">";

Write-Output "----==== Replace base href in $indexFile to be /$repoName/"
((Get-Content -path $indexFile -Raw) -replace $originalBaseUrlText, $targetBaseUrlText) | Set-Content -Path $indexFile

Write-Output "----==== Delete dist folder"
Remove-Item ./dist/ -Recurse
