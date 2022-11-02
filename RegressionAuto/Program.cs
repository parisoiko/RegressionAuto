// See https://aka.ms/new-console-template for more information
using System.Data;
using CsvHelper;
using System.Formats.Asn1;
using System.Globalization;
using System;
using Microsoft.ML.Data;
using System.Xml.Schema;
using Microsoft.ML;
using System.IO;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.Data.Analysis;
using System;
using System.IO;
using Microsoft.ML.AutoML;
using Tensorflow.Contexts;
using Microsoft.ML.Trainers;
using RegressionAuto;



MLContext mlContext = new MLContext(seed: 1);
IDataView trainDataView = mlContext.Data.LoadFromTextFile<ModelInput>("C:/Users/HPLaptop/source/repos/ParkingRegression/ParkingRegression/Data/AllData2021Plus4.csv", hasHeader: true, separatorChar: ',');
IDataView testDataView = mlContext.Data.LoadFromTextFile<ModelInput>("C:/Users/HPLaptop/source/repos/ParkingRegression/ParkingRegression/Data/SingleDataTest.csv", hasHeader: true, separatorChar: ',');
ITransformer model;
Console.WriteLine("Set data? (Yes/No)");
if (Console.ReadLine().Equals("Yes")) {
    SetData();
}
Console.WriteLine("|-------------------------------------|");
Console.WriteLine("|-------------------------------------|");
Console.WriteLine("|-------------------------------------|");
Console.WriteLine("Build new model? (Yes/No)");
if (Console.ReadLine().Equals("Yes")) {
    model = BuildModel(trainDataView, testDataView, mlContext);
} else {
    model = LoadModel(mlContext);
}
Console.WriteLine("|-------------------------------------|");
Console.WriteLine("|-------------------------------------|");
Console.WriteLine("|-------------------------------------|");
Console.WriteLine("Make a prediction? (Yes/No)");
if (Console.ReadLine().Equals("Yes")) {
    var predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);
    TestEngine(predictionEngine);
}
Console.WriteLine("|-------------------------------------|");
Console.WriteLine("|-------------------------------------|");
Console.WriteLine("|-------------------------------------|");
Console.WriteLine("Build Poisson for weight check? (Yes/No)");
if (Console.ReadLine().Equals("Yes")) {
    BuildPoissonModel(trainDataView, testDataView, mlContext);
}



ITransformer BuildModel(IDataView trainDataView, IDataView testDataView, MLContext mlContext) {

    var experimentSettings = new RegressionExperimentSettings();
    var progressHandler = new Progress<RunDetail<RegressionMetrics>>();
    experimentSettings.MaxExperimentTimeInSeconds = 500;
    //experimentSettings.Trainers.Remove(RegressionTrainer.LbfgsPoissonRegression);
    //experimentSettings.Trainers.Remove(RegressionTrainer.OnlineGradientDescent);
    //experimentSettings.Trainers.Remove(RegressionTrainer.StochasticDualCoordinateAscent);
    var columnInfo = new ColumnInformation();
    columnInfo.NumericColumnNames.Remove("Latest");
    columnInfo.NumericColumnNames.Remove("Previous");
    columnInfo.NumericColumnNames.Remove("CheckIns");
    columnInfo.NumericColumnNames.Remove("CheckOuts");
    columnInfo.TextColumnNames.Remove("Date");
    columnInfo.IgnoredColumnNames.Add("Date");
    columnInfo.IgnoredColumnNames.Add("Latest");
    columnInfo.IgnoredColumnNames.Add("Previous");
    columnInfo.IgnoredColumnNames.Add("CheckIns");
    columnInfo.IgnoredColumnNames.Add("CheckOuts");
    columnInfo.LabelColumnName = "Parking";
    RegressionExperiment experiment = mlContext.Auto().CreateRegressionExperiment(experimentSettings);
    ExperimentResult<RegressionMetrics> experimentResult = experiment.Execute(trainDataView, testDataView, columnInformation: columnInfo, progressHandler: progressHandler);

    RunDetail<RegressionMetrics> bestRun = experimentResult.BestRun;
    ITransformer model = bestRun.Model;
    mlContext.Model.Save(model, trainDataView.Schema, "C:/Users/HPLaptop/source/repos/RegressionAuto/RegressionAuto/Model.zip");
    
    return model;
}

ITransformer BuildPoissonModel(IDataView trainDataView, IDataView testDataView, MLContext mlContext) {
    var pipeline = mlContext.Transforms.ReplaceMissingValues(new[] { new InputOutputColumnPair(@"Year", @"Year"), new InputOutputColumnPair(@"Month", @"Month"), new InputOutputColumnPair(@"Day", @"Day"), new InputOutputColumnPair(@"DayOfWeek", @"DayOfWeek"), new InputOutputColumnPair(@"Hour", @"Hour"), new InputOutputColumnPair(@"TimeOfDay", @"TimeOfDay"), new InputOutputColumnPair(@"Holidays", @"Holidays"), new InputOutputColumnPair(@"Season", @"Season"), new InputOutputColumnPair(@"WeekEnd", @"WeekEnd") })
                                    .Append(mlContext.Transforms.Concatenate(@"Features", new[] { @"Year", @"Month", @"Day", @"DayOfWeek", @"Hour", @"TimeOfDay", @"Holidays", @"Season", @"WeekEnd" }))
                                    .Append(mlContext.Regression.Trainers.LbfgsPoissonRegression(
                                                    labelColumnName: @"Parking",
                                                    featureColumnName: @"Features"));
    var model = pipeline.Fit(trainDataView);
    CheckWeights(model);
    return model;
}

ITransformer LoadModel(MLContext mlContext) {
    string MLNetModelPath = Path.GetFullPath("C:/Users/HPLaptop/source/repos/RegressionAuto/RegressionAuto/Model.zip");
    var mlModel = mlContext.Model.Load(MLNetModelPath, out var _);
    return mlModel;
}

void TestEngine(PredictionEngine<ModelInput, ModelOutput> predictionEngine) {
    DateTime date = new DateTime(2022, 8, 1, 0, 0, 0);
    date = date.AddHours(-1);
    List<ModelInput> test = new List<ModelInput>();
    for (int x = 0; x < (24 * 31) + 1; x++) {
        float year = date.Year;
        float month = date.Month;
        float day = date.Day;
        float dayOfWeek = (float)date.DayOfWeek;
        float hour = date.Hour;
        float timeOfDay = 0;
        if (hour > 2 && hour <= 6) {
            timeOfDay = 1;
        }
        else if (hour > 6 && hour <= 10) {
            timeOfDay = 2;
        }
        else if (hour > 10 && hour <= 14) {
            timeOfDay = 3;
        }
        else if (hour > 14 && hour <= 18) {
            timeOfDay = 4;
        }
        else if (hour > 18 && hour <= 22) {
            timeOfDay = 5;
        }
        float season = 0;
        if (month > 2 && month <= 5) {
            season = 1;
        }
        else if (month > 5 && month <= 8) {
            season = 2;
        }
        else if (month > 8 && month <= 11) {
            season = 3;
        }
        float holidays = 0;
        if (month == 7 || month == 8 || (month == 9 && day <= 5)) {
            holidays = 1;
        }
        else if (month == 4 && day >= 17 && day <= 30) {
            holidays = 1;
        }
        else if ((month == 12 && day >= 23) || (month == 1 && day <= 7)) {
            holidays = 1;
        }
        else if (month == 3 && day >= 24 && day <= 26) {
            holidays = 1;
        }
        else if (month == 10 && day >= 27 && day <= 29) {
            holidays = 1;
        }
        float weekend = 0;
        if (dayOfWeek == 0 || dayOfWeek == 6) {
            weekend = 1;
        }
        test.Add(new ModelInput(year, month, day, dayOfWeek, hour, timeOfDay, holidays, season, weekend));
        date = date.AddHours(1);
    }
    List<int[]> actual = new List<int[]>();
    using (var reader = new StreamReader("C:/Users/HPLaptop/source/repos/RegressionAuto/RegressionAuto/Data/SingleDataTest.csv")) {
        //int j;
        //for (j = 0; j < 2873; j++) {
        //    reader.ReadLine();
        //}
        reader.ReadLine();
        do {
            string line = reader.ReadLine();
            var values = line.Split(',');
            actual.Add(new int[] { Int32.Parse(values[9]), Int32.Parse(values[10]), Int32.Parse(values[11]), Int32.Parse(values[13]), Int32.Parse(values[14]) });
        } while (!reader.EndOfStream);
    }
    float totalDeviationParis = 0;
    int y = 0;
    foreach (ModelInput parking in test) {
        var predictionR = predictionEngine.Predict(parking);
        predictionR.Score = predictionR.Score < 0 ? 0 : predictionR.Score;
        if (y != 0) {
            Console.WriteLine($"Date: {parking.Day}-{parking.Month}-{parking.Year} {parking.Hour}:00:00"); /* */
            Console.WriteLine($"Forecast demand: {(int)predictionR.Score:0.####} \t\t\t Actual demand: {actual[y - 1][0]}");
            totalDeviationParis += Math.Abs((int)predictionR.Score - actual[y - 1][0]);
        }
        y++;
    }
    Console.WriteLine($"Total Paris Deviation: " + totalDeviationParis);
    Console.WriteLine($"Total Mean Paris Deviation: " + totalDeviationParis/(test.Count-1));
}

void SetData() {
    var inputs = new SortedDictionary<DateTime, int>();
    var inputs2 = new SortedDictionary<DateTime, int>();
    var outputs = new SortedDictionary<DateTime, int>();
    using (var reader = new StreamReader("C:/Users/HPLaptop/source/repos/RegressionAuto/RegressionAuto/Data/OG.csv")) {

        do {
            string line = reader.ReadLine();
            var values = line.Split(',');
            var entrance = DateTime.Parse(values[0]); //MINUTES PRECISSION
            var exit = DateTime.Parse(values[1]);//MINUTES PRECISSION
            entrance = entrance.AddSeconds(-1 * entrance.Second).AddMilliseconds(-1 * entrance.Millisecond);
            exit = exit.AddSeconds(-1 * exit.Second).AddMilliseconds(-1 * exit.Millisecond);
            if (!inputs.ContainsKey(entrance))
                inputs.Add(entrance, 0);
            inputs[entrance]++;

            if (!outputs.ContainsKey(exit))
                outputs.Add(exit, 0);
            outputs[exit]++;
        }
        while (!reader.EndOfStream);



        var minDate = inputs.Keys.Min(); //NA MPEI KAI TO MIN EXIT KAI NA PARW TO MEGALYTERO
        var minDateOutputs = outputs.Keys.Min();
        if (minDateOutputs < minDate) {
            minDate = minDateOutputs;
        }
        var maxDate = inputs.Keys.Max();
        var maxDateOutputs = outputs.Keys.Max();
        if (maxDateOutputs > maxDate) {
            maxDate = maxDateOutputs;
        }
        var hourValue = new List<ModelInput>();
        var stayedInside = 0;

        //ENTRANCES
        //EXITS
        //MAXIMUM OCCUPANCY
        var hourEntrances = 0;
        var hourExits = 0;
        var hourData = new List<int>();
        var firstData = new List<ModelInput>();
        var moreData = new List<ModelInput>();
        var garbageData = new List<ModelInput>();
        var singleDataTest = new List<ModelInput>();
        int prev = stayedInside;
        for (int y = 0; y <= (maxDate - minDate).TotalMinutes; y++) {
            var entrances = 0;
            var exits = 0;


            //var hourData = new List<int>();

            var currentDate = minDate.AddMinutes(y);


            //GET HOUR DATA

            if (inputs.ContainsKey(currentDate)) {
                entrances = inputs[currentDate];
            }
            if (outputs.ContainsKey(currentDate)) {
                exits = outputs[currentDate];
            }

            hourEntrances += entrances;
            hourExits += exits;
            //

            stayedInside += entrances - exits;
            hourData.Add(stayedInside);

            if (currentDate.Minute == 59) {
                var hourstayedInside = hourData.Max();
                float year = currentDate.Year;
                float month = currentDate.Month;
                float day = currentDate.Day;
                float dayOfWeek = (float)currentDate.DayOfWeek;
                float hour = currentDate.Hour;
                float timeOfDay = 0;
                if (hour > 2 && hour <= 6) {
                    timeOfDay = 1;
                }
                else if (hour > 6 && hour <= 10) {
                    timeOfDay = 2;
                }
                else if (hour > 10 && hour <= 14) {
                    timeOfDay = 3;
                }
                else if (hour > 14 && hour <= 18) {
                    timeOfDay = 4;
                }
                else if (hour > 18 && hour <= 22) {
                    timeOfDay = 5;
                }
                float season = 0;
                if (month > 2 && month <= 5) {
                    season = 1;
                }
                else if (month > 5 && month <= 8) {
                    season = 2;
                }
                else if (month > 8 && month <= 11) {
                    season = 3;
                }
                float holidays = 0;
                if (month == 7 || month == 8 || (month == 9 && day <= 5)) {
                    holidays = 1;
                }
                else if (month == 4 && day >= 17 && day <= 30) {
                    holidays = 1;
                }
                else if ((month == 12 && day >= 23) || (month == 1 && day <= 7)) {
                    holidays = 1;
                }
                else if (month == 3 && day >= 24 && day <= 26) {
                    holidays = 1;
                }
                else if (month == 10 && day >= 27 && day <= 29) {
                    holidays = 1;
                }
                float weekend = 0;
                if (dayOfWeek == 0 || dayOfWeek == 6) {
                    weekend = 1;
                }
                if (year == 2022 && month == 8) {
                    singleDataTest.Add(new ModelInput(year, month, day, dayOfWeek, hour, timeOfDay, holidays, season, weekend, hourstayedInside, hourEntrances, hourExits, currentDate.AddMinutes(-1 * currentDate.Minute).ToString("dd/MM/yyyy HH:mm:ss"), prev, stayedInside));
                }
                if (year == 2021 || (year == 2022 && month < 8)) {
                    moreData.Add(new ModelInput(year, month, day, dayOfWeek, hour, timeOfDay, holidays, season, weekend, hourstayedInside, hourEntrances, hourExits, currentDate.AddMinutes(-1 * currentDate.Minute).ToString("dd/MM/yyyy HH:mm:ss"), prev, stayedInside));
                }
                if (year == 2021) {
                    firstData.Add(new ModelInput(year, month, day, dayOfWeek, hour, timeOfDay, holidays, season, weekend, hourstayedInside, hourEntrances, hourExits, currentDate.AddMinutes(-1 * currentDate.Minute).ToString("dd/MM/yyyy HH:mm:ss"), prev, stayedInside));
                }
                if(year == 2022 && month == 9) {
                    garbageData.Add(new ModelInput(year, month, day, dayOfWeek, hour, timeOfDay, holidays, season, weekend, hourstayedInside, hourEntrances, hourExits, currentDate.AddMinutes(-1 * currentDate.Minute).ToString("dd/MM/yyyy HH:mm:ss"), prev, stayedInside));
                }
                hourEntrances = 0;
                hourExits = 0;
                hourData.Clear();
                prev = hourstayedInside;
            }

        }
        var path = "C:/Users/HPLaptop/source/repos/RegressionAuto/RegressionAuto/Data/AllData2021.csv";
        var path2 = "C:/Users/HPLaptop/source/repos/RegressionAuto/RegressionAuto/Data/Garbage.csv";
        var path3 = "C:/Users/HPLaptop/source/repos/RegressionAuto/RegressionAuto/Data/SingleDataTest.csv";
        var path4 = "C:/Users/HPLaptop/source/repos/RegressionAuto/RegressionAuto/Data/AllData2021Plus4.csv";
        using (var writer = new StreamWriter(path)) {
            using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture)) {
                csv.WriteRecords(firstData);
            }
        }
        using (var writer = new StreamWriter(path2)) {
            using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture)) {
                csv.WriteRecords(garbageData);
            }
        }
        using (var writer = new StreamWriter(path3)) {
            using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture)) {
                csv.WriteRecords(singleDataTest);
            }
        }
        using (var writer = new StreamWriter(path4)) {
            using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture)) {
                csv.WriteRecords(moreData);
            }
        }
    }
}

void CheckWeights(ITransformer model) {
    IEnumerable<ITransformer> chain = model as IEnumerable<ITransformer>;
    ISingleFeaturePredictionTransformer<object> predictionTransformer = chain.Last() as ISingleFeaturePredictionTransformer<object>;
    object modelParameters = predictionTransformer.Model;

    //This is for FAST TREE
    //VBuffer<float> buffer = new VBuffer<float>();
    //((FastTreeRegressionModelParameters)modelParameters).GetFeatureWeights(ref buffer);
    //Console.WriteLine($"{buffer.GetValues}");


    //This is for POISSON
    for (int p = 0; p <= ((PoissonRegressionModelParameters)modelParameters).Weights.Count - 1; p++) {
        Console.WriteLine($"{((PoissonRegressionModelParameters)modelParameters).Weights[p]}");
    }
}