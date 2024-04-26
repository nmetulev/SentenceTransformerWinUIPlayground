using System;
using Microsoft.ML.OnnxRuntimeGenAI;
using System.IO;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;

namespace SentenceTransformerPlayground
{
    public class SLMRunner : IDisposable
    {
        private readonly string ModelDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "model", "Phi-3-mini-4k-instruct-onnx\\directml-int4-awq-block-128");

        private Model? model = null;
        private Tokenizer? tokenizer = null;
        public event EventHandler? ModelLoaded = null;

        [MemberNotNullWhen(true, nameof(model), nameof(tokenizer))]
        public bool IsReady => model != null && tokenizer != null;

        public void Dispose()
        {
            model?.Dispose();
            tokenizer?.Dispose();
        }

        public string Infer(string prompt)
        {
            if (!IsReady)
            {
                throw new InvalidOperationException("Model is not ready");
            }

            var generatorParams = new GeneratorParams(model);

            var sequences = tokenizer.Encode(prompt);

            generatorParams.SetSearchOption("max_length", 2048);
            generatorParams.SetInputSequences(sequences);

            var outputSequences = model.Generate(generatorParams);
            var outputString = tokenizer.Decode(outputSequences[0]);

            return outputString;
        }

        public IEnumerable<string> InferStreaming(string prompt)
        {
            if (!IsReady)
            {
                throw new InvalidOperationException("Model is not ready");
            }

            var generatorParams = new GeneratorParams(model);

            var sequences = tokenizer.Encode(prompt);

            generatorParams.SetSearchOption("max_length", 4096);
            generatorParams.SetInputSequences(sequences);
            generatorParams.TryGraphCaptureWithMaxBatchSize(1);

            using var tokenizerStream = tokenizer.CreateStream();
            using var generator = new Generator(model, generatorParams);
            StringBuilder stringBuilder = new();
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

        public Task InitializeAsync()
        {
            return Task.Run(() =>
            {
                var sw = Stopwatch.StartNew();
                model = new Model(ModelDir);
                tokenizer = new Tokenizer(model);
                sw.Stop();
                Debug.WriteLine($"Model loading took {sw.ElapsedMilliseconds} ms");
                ModelLoaded?.Invoke(this, EventArgs.Empty);
            });
        }
    }
}