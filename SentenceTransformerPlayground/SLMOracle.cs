#nullable enable

using System;
using Microsoft.ML.OnnxRuntimeGenAI;
using System.IO;
using System.Collections.Generic;
using System.Text;

namespace SentenceTransformerPlayground
{
    public class SLMOracle : IDisposable
    {
        private string ModelDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "model", "phi2_int4_cpu_2");

        private Model model;
        private Tokenizer tokenizer;

        public SLMOracle()
        {
            model = new Model(ModelDir);
            tokenizer = new Tokenizer(model);
        }

        public void Dispose()
        {
            model.Dispose();
            tokenizer.Dispose();
        }

        public string Infer(string prompt)
        {
            var generatorParams = new GeneratorParams(model);

            var sequences = tokenizer.Encode(prompt);

            generatorParams.SetSearchOption("max_length", 2048);
            generatorParams.SetInputSequences(sequences);

            var outputSequences = model.Generate(generatorParams);
            var outputString = tokenizer.Decode(outputSequences[0]);

            return outputString;
        }

        public async IAsyncEnumerable<string> InferStreaming(string prompt)
        {
            var generatorParams = new GeneratorParams(model);

            var sequences = tokenizer.Encode(prompt);

            generatorParams.SetSearchOption("max_length", 2048);
            generatorParams.SetInputSequences(sequences);

            using var tokenizerStream = tokenizer.CreateStream();
            using var generator = new Generator(model, generatorParams);
            StringBuilder stringBuilder = new StringBuilder();
            while (!generator.IsDone())
            {
                generator.ComputeLogits();
                generator.GenerateNextToken();
                var part = tokenizerStream.Decode(generator.GetSequence(0)[^1]);
                stringBuilder.Append(part);
                if (stringBuilder.ToString().Contains("<|end|>")
                    || stringBuilder.ToString().Contains("<|user|>")
                    || stringBuilder.ToString().Contains("<|system|>"))
                {
                    break;
                }
                yield return part;
            }
        }
    }
}