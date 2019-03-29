using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace DoneDoneML
{
    class Program
    {
        static void Main(string[] args)
        {
            var issues = Issue.GetAllIssues("issues.csv").Select(i => new Input {
                Title = i.Title,
                Description = i.Description,
                Fixer = i.FixerName
            });

            string modelPath = "dd.ml";

            if(!File.Exists(modelPath))
                CreateModel(modelPath, issues);
            
            var mlContext = new MLContext();
            var model = LoadModel(modelPath, mlContext);

            var predEngine = model.CreatePredictionEngine<Input, Prediction>(mlContext);

            do
            {
                var input = new Input();
                Console.WriteLine("Title (or enter to exit):");
                input.Title = Console.ReadLine();

                if(string.IsNullOrWhiteSpace(input.Title))
                    break;

                Console.WriteLine("Description:");
                input.Description = Console.ReadLine();

                var pred = predEngine.Predict(input);

                foreach(var p in pred.GetPredictions(predEngine))
                {
                    Console.WriteLine("Prediction: {0} ({1})", p.label, p.score);
                }

            } while(true);         
        }

        static ITransformer LoadModel(string path, MLContext mlContext)
        {
            using(var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
                return mlContext.Model.Load(stream);
        }

        static void CreateModel(string path, IEnumerable<Input> input)
        {
            var mlContext = new MLContext();
            var trainingDataView = mlContext.Data.LoadFromEnumerable(input);

            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: DefaultColumnNames.Label, inputColumnName: nameof(Input.Fixer))
                .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "TitleFeaturized", inputColumnName: nameof(Input.Title)))
                .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "DescriptionFeaturized", inputColumnName: nameof(Input.Description)))
                .Append(mlContext.Transforms.Concatenate(outputColumnName: DefaultColumnNames.Features, "TitleFeaturized", "DescriptionFeaturized"))
                .AppendCacheCheckpoint(mlContext);

            var trainer = mlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(
                DefaultColumnNames.Label, DefaultColumnNames.Features);
            
            var trainingPipeline = dataProcessPipeline
                .Append(trainer)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(DefaultColumnNames.PredictedLabel));

            mlContext.MulticlassClassification.CrossValidate(data: trainingDataView, estimator: trainingPipeline, numFolds: 6, labelColumn: DefaultColumnNames.Label);

            var trainedModel = trainingPipeline.Fit(trainingDataView);

            using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(trainedModel, fs);
        }

        public class Input
        {
            [LoadColumn(0)]
            public string Title;

            [LoadColumn(1)]
            public string Description;

            [LoadColumn(2)]
            public string Fixer;
        }

        public class Prediction
        {
            public string PredictedLabel;

            public float[] Score;

            public IEnumerable<(string label, float score)> GetPredictions(PredictionEngine<Input, Prediction> predEngine, int top = 3)
            {
                VBuffer<ReadOnlyMemory<char>> slotNames = default;
                predEngine.OutputSchema[DefaultColumnNames.Score].GetSlotNames(ref slotNames);

                return Score.Select((score, i) => (slotNames.GetItemOrDefault(i).ToString(), score))
                            .OrderByDescending(x => x.score)
                            .Take(3)
                            .ToList();
            }
        }
    }
}
