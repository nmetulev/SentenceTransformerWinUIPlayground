using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Diagnostics;
using System.Threading.Tasks;
using System;
using TorchSharp;
using System.Linq;
using System.IO;
using VectorDB;
using BERTTokenizers.Base;
using SharpDX.DXGI;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Threading;
using System.Text.RegularExpressions;
using Microsoft.ML.OnnxRuntimeGenAI;

namespace SentenceTransformerPlayground
{
    public class RAGService : IDisposable
    {
        // model from https://huggingface.co/optimum/all-MiniLM-L6-v2
        private readonly string modelDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "model");
        private InferenceSession? _inferenceSession;
        private MyTokenizer? tokenizer = null;
        private List<TextChunk>? _content;
        private VectorCollection<TextChunk>? _embeddings = null;

        public event EventHandler? ResourcesLoaded = null;

        [MemberNotNullWhen(true, nameof(_inferenceSession))]
        public bool IsModelReady => _inferenceSession != null;

        [MemberNotNullWhen(true, nameof(_embeddings), nameof(_content))]
        public bool IsEmbeddingsReady => _embeddings != null && _content != null;

        [MemberNotNullWhen(true, nameof(_inferenceSession), nameof(_embeddings), nameof(_content))]
        public bool IsReady => IsModelReady && IsEmbeddingsReady;

        [MemberNotNull(nameof(_inferenceSession))]
        private void InitModel()
        {
            if (_inferenceSession != null)
            {
                return;
            }

            var sessionOptions = new SessionOptions
            {
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO
            };

            int deviceId = GetBestDeviceId();

            sessionOptions.AppendExecutionProvider_DML(deviceId);

            _inferenceSession = new InferenceSession($@"{modelDir}\model.onnx", sessionOptions);

            ResourcesLoaded?.Invoke(this, EventArgs.Empty);
        }

        public static List<Adapter1> GetAdapters()
        {
            var factory1 = new Factory1();
            var adapters = new List<Adapter1>();
            for (int i = 0; i < factory1.GetAdapterCount1(); i++)
            {
                adapters.Add(factory1.GetAdapter1(i));
            }

            return adapters;
        }

        private static int GetBestDeviceId()
        {
            int deviceId = 0;
            Adapter1? selectedAdapter = null;
            List<Adapter1> list = GetAdapters();
            for (int i = 0; i < list.Count; i++)
            {
                Adapter1? adapter = list[i];
                Debug.WriteLine($"Adapter {i}:");
                Debug.WriteLine($"\tDescription: {adapter.Description1.Description}");
                Debug.WriteLine($"\tDedicatedVideoMemory: {(long)adapter.Description1.DedicatedVideoMemory / 1000000000}GB");
                Debug.WriteLine($"\tSharedSystemMemory: {(long)adapter.Description1.SharedSystemMemory / 1000000000}GB");
                if (selectedAdapter == null || (long)adapter.Description1.DedicatedVideoMemory > (long)selectedAdapter.Description1.DedicatedVideoMemory)
                {
                    selectedAdapter = adapter;
                    deviceId = i;
                }
            }

            return deviceId;
        }

        // TODO: run multiple sentences at once
        private async Task<float[][]> GetEmbeddingsAsync(params string[] sentences)
        {
            if (!IsModelReady)
            {
                return Array.Empty<float[]>();
            }

            tokenizer ??= new MyTokenizer($@"{modelDir}\vocab.txt");

            sentences = ["abc", "abcdef", "def"];

            var encoded = tokenizer.CustomEncode(sentences);

            var input = new ModelInput
            {
                InputIds = encoded.SelectMany(t => t.Select(t2 => t2.InputIds)).ToArray(),
                TokenTypeIds = encoded.SelectMany(t => t.Select(t2 => t2.TokenTypeIds)).ToArray(),
                AttentionMask = encoded.SelectMany(t => t.Select(t2 => t2.AttentionMask)).ToArray(),
            };

            var runOptions = new RunOptions();

            // Create input tensors over the input data.
            using var inputIdsOrtValue = OrtValue.CreateTensorValueFromMemory(input.InputIds,
                  [sentences.Length, input.InputIds.Length]);

            using var attMaskOrtValue = OrtValue.CreateTensorValueFromMemory(input.AttentionMask,
                  [sentences.Length, input.AttentionMask.Length]);

            using var typeIdsOrtValue = OrtValue.CreateTensorValueFromMemory(input.TokenTypeIds,
                  [sentences.Length, input.TokenTypeIds.Length]);

            var inputNames = new List<string>
            {
                "input_ids",
                "attention_mask",
                "token_type_ids"
            };

            var inputs = new List<OrtValue>
            {
                { inputIdsOrtValue },
                { attMaskOrtValue },
                { typeIdsOrtValue }
            };

            var outputValues = new List<OrtValue> { 
                    OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance,
                        TensorElementType.Float, [sentences.Length, input.InputIds.Length, 768])};

            try
            {
                var output = await _inferenceSession.RunAsync(runOptions, inputNames, inputs, _inferenceSession.OutputNames, outputValues);

                var firstElement = output.ToList()[0];
                var data = firstElement.GetTensorDataAsSpan<float>().ToArray();
                var typeAndShape = firstElement.GetTensorTypeAndShape();

                var sentence_embeddings = MeanPooling(data, input.AttentionMask, typeAndShape.Shape);
                var denom = sentence_embeddings.norm(1, true, 2).clamp_min(1e-12).expand_as(sentence_embeddings);
                var results = sentence_embeddings / denom;
                return [results.data<float>().ToArray()];
            }
            catch (Exception e)
            {
                Debug.WriteLine(e.Message);
                return Array.Empty<float[]>();
            }
        }

        public static torch.Tensor MeanPooling(float[] embeddings, long[] attentionMask, long[] shape)
        {
            var tokenEmbeddings = torch.tensor(embeddings, shape);
            var attentionMaskExpanded = torch.tensor(attentionMask, [shape[0], shape[1]]).unsqueeze(-1).expand(tokenEmbeddings.shape).@float();

            var sumEmbeddings = (tokenEmbeddings * attentionMaskExpanded).sum(new[] { 1L });
            var sumMask = attentionMaskExpanded.sum(new[] { 1L }).clamp(1e-9, float.MaxValue);

            return sumEmbeddings / sumMask;
        }

        public async Task<List<TextChunk>> Search(string searchTerm, int top = 5, int range = 3)
        {
            List<TextChunk> contents = [];
            if (!IsReady)
            {
                return contents;
            }

            var searchVector = await GetEmbeddingsAsync(searchTerm, searchTerm, searchTerm, searchTerm).ConfigureAwait(false);
            var ranking = _embeddings.CalculateRanking(searchVector[0]);

            for (int i = 0; i < top; i++)
            {
                var indexMin = Math.Max(0, ranking[i] - range);
                var indexMax = Math.Min(indexMin + range, _content.Count);
                contents.AddRange(_content.Skip(indexMin).Take(indexMax - indexMin).ToList());
            }

            return contents;
        }

        public async Task InitializeAsync(List<TextChunk>? contents = null, EventHandler<float>? progress = null, CancellationToken ct = default)
        {
            if (contents == null)
            {
                _embeddings = await VectorCollection<TextChunk>.LoadFromDiskAsync("vectors.vec", ct).ConfigureAwait(false);
                if (_embeddings != null)
                {
                    _content = _embeddings.Objects.ToList();
                }

                ResourcesLoaded?.Invoke(this, EventArgs.Empty);

                await Task.Run(InitModel, ct).ConfigureAwait(false);

                return;
            }

            _content = contents.ToList();

            if (_content.Count == 0)
            {
                await Task.Run(InitModel, ct).ConfigureAwait(false);

                return;
            }

            await Task.Run(InitModel, ct).ConfigureAwait(false);

            Stopwatch stopwatch = Stopwatch.StartNew();

            await Task.Run(async () =>
            {
                for (int i = 0; i < _content.Count; i++)
                {
                    if (ct.IsCancellationRequested)
                    {
                        return;
                    }
                    
                    _content[i].Vectors = (await GetEmbeddingsAsync(_content[i].Text!).ConfigureAwait(false))[0];

                    progress?.Invoke(this, (float)i / _content.Count);
                }
            }, ct).ConfigureAwait(false);

            if (ct.IsCancellationRequested)
            {
                return;
            }

            try
            {
                _embeddings = new VectorCollection<TextChunk>(_content.Count, _content);
                await _embeddings.SaveToDiskAsync("vectors.vec", ct).ConfigureAwait(false);

                stopwatch.Stop();
                Debug.WriteLine($"Indexing took {stopwatch.ElapsedMilliseconds} ms");

                ResourcesLoaded?.Invoke(this, EventArgs.Empty);
            }
            catch (Exception e)
            {
                Debug.WriteLine(e.Message);
            }
        }

        public void Dispose()
        {
            _inferenceSession?.Dispose();
        }
    }

    public class ModelInput
    {
        public required long[] InputIds { get; init; }

        public required long[] AttentionMask { get; init; }

        public required long[] TokenTypeIds { get; init; }
    }

    public class MyTokenizer(string vocabPath) : UncasedTokenizer(vocabPath)
    {
        public IEnumerable<List<(long InputIds, long TokenTypeIds, long AttentionMask)>> CustomEncode(params string[] texts)
        {
            int maxLength = texts.Max(s => s.Length);

            foreach (string text in texts)
            {
                IEnumerable<string> enumerable = [];
                enumerable = enumerable.Concat(["[CLS]"]);
                enumerable = enumerable.Concat(TokenizeSentence(text));
                enumerable = enumerable.Concat(["[SEP]"]);
                if (maxLength - text.Length - 1 > 0)
                {
                    enumerable = enumerable.Concat(Enumerable.Repeat("[PAD]", maxLength - text.Length - 1));
                }
                List<(string, int)> list2 = enumerable.SelectMany(TokenizeSubwords).ToList();
                IEnumerable<long> second3 = SegmentIndex(list2);
                List<(string, int, long)> list = list2.Zip<(string, int), long, (string, int, long)>(second3, ((string Token, int VocabularyIndex) tokenindex, long segmentindex) => (tokenindex.Token, tokenindex.VocabularyIndex, segmentindex)).ToList();

                List<long> second = Enumerable.Repeat(0L, list.Count).ToList();
                long[] first = list.Select<(string, int, long), long>(((string Token, int VocabularyIndex, long SegmentIndex) token) => token.VocabularyIndex).Concat(second).ToArray();
                long[] second2 = list.Select(((string Token, int VocabularyIndex, long SegmentIndex) token) => token.SegmentIndex).Concat(second).ToArray();
                yield return (
                    from x in Enumerable.Zip(second: list.Select(((string Token, int VocabularyIndex, long SegmentIndex) o) => 1L).Concat(second).ToArray(), first: first.Zip(second2, Tuple.Create), resultSelector: (Tuple<long, long> t, long z) => Tuple.Create(t.Item1, t.Item2, z))
                        select (x.Item1, x.Item2, x.Item3)).ToList();
            }

            /*
{'input_ids': tensor(
[
[  101,  5925,   102,     0,     0],
[  101,  5925,  3207,  2546,   102],
[  101, 13366,   102,     0,     0]]),
'token_type_ids': tensor(
[
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0]]),
'attention_mask': tensor(
[
[1, 1, 1, 0, 0],
[1, 1, 1, 1, 1],
[1, 1, 1, 0, 0]])}
*/
        }

        private IEnumerable<(string Token, int VocabularyIndex)> TokenizeSubwords(string word)
        {
            if (_vocabularyDict.ContainsKey(word))
            {
                return new (string, int)[1] { (word, _vocabularyDict[word]) };
            }

            List<(string, int)> list = new List<(string, int)>();
            string text = word;
            while (!string.IsNullOrEmpty(text) && text.Length > 2)
            {
                string text2 = null;
                int num = text.Length;
                while (num >= 1)
                {
                    string text3 = text.Substring(0, num);
                    if (!_vocabularyDict.ContainsKey(text3))
                    {
                        num--;
                        continue;
                    }

                    text2 = text3;
                    break;
                }

                if (text2 == null)
                {
                    list.Add(("[UNK]", _vocabularyDict["[UNK]"]));
                    return list;
                }

                text = new Regex(text2).Replace(text, "##", 1);
                list.Add((text2, _vocabularyDict[text2]));
            }

            if (!string.IsNullOrWhiteSpace(word) && !list.Any())
            {
                list.Add(("[UNK]", _vocabularyDict["[UNK]"]));
            }

            return list;
        }

        private IEnumerable<long> SegmentIndex(List<(string token, int index)> tokens)
        {
            int num = 0;
            List<long> list = new List<long>();
            foreach (var token in tokens)
            {
                string item = token.token;
                list.Add(num);
                if (item == "[SEP]")
                {
                    num++;
                }
            }

            return list;
        }
    }
}
