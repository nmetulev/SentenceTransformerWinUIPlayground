using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Diagnostics;
using System.Threading.Tasks;
using System;
using System.Linq;
using System.IO;
using VectorDB;
using BERTTokenizers.Base;
using SharpDX.DXGI;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Threading;
using System.Text.RegularExpressions;
using System.Numerics;

namespace SentenceTransformerPlayground
{
    public class RAGService : IDisposable
    {
        // model from https://huggingface.co/optimum/all-MiniLM-L6-v2
        private readonly string modelDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "model\\all-MiniLM-L6-v2");
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

        private async Task<float[][]> GetEmbeddingsAsync(params string[] sentences)
        {
            if (!IsModelReady)
            {
                return Array.Empty<float[]>();
            }

            tokenizer ??= new MyTokenizer($@"{modelDir}\vocab.txt");

            var encoded = tokenizer.CustomEncode(sentences);

            var input = new ModelInput
            {
                InputIds = encoded.Select(t => t.InputIds).ToArray(),
                TokenTypeIds = encoded.Select(t => t.TokenTypeIds).ToArray(),
                AttentionMask = encoded.Select(t => t.AttentionMask).ToArray(),
            };

            var runOptions = new RunOptions();

            // round up
            int sequenceLength = input.InputIds.Length / sentences.Length;

            // Create input tensors over the input data.
            using var inputIdsOrtValue = OrtValue.CreateTensorValueFromMemory(input.InputIds,
                  [sentences.Length, sequenceLength]);

            using var attMaskOrtValue = OrtValue.CreateTensorValueFromMemory(input.AttentionMask,
                  [sentences.Length, sequenceLength]);

            using var typeIdsOrtValue = OrtValue.CreateTensorValueFromMemory(input.TokenTypeIds,
                  [sentences.Length, sequenceLength]);

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

            List<OrtValue> outputValues = [
                OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance,
                    TensorElementType.Float, [sentences.Length, sequenceLength, 384]),
                OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance,
                    TensorElementType.Float, [sentences.Length, 384])];

            try
            {
                var output = await _inferenceSession.RunAsync(runOptions, inputNames, inputs, _inferenceSession.OutputNames, outputValues);

                var firstElement = output.ToList()[0];
                var data = firstElement.GetTensorDataAsSpan<float>().ToArray();
                var typeAndShape = firstElement.GetTensorTypeAndShape();

                var sentence_embeddings = MeanPooling(data, input.AttentionMask, typeAndShape.Shape);

                var resultArray = NormalizeAndDivide(sentence_embeddings, typeAndShape.Shape);

                return Enumerable.Chunk(resultArray, resultArray.Length / sentences.Length).ToArray();
            }
            catch (Exception e)
            {
                Debug.WriteLine(e.Message);
                return Array.Empty<float[]>();
            }
        }

        public static float[] MeanPooling(float[] embeddings, long[] attentionMask, long[] shape)
        {
            long batchSize = shape[0];
            int sequenceLength = (int)shape[1];
            int embeddingSize = (int)shape[2];
            float[] result = new float[batchSize * embeddingSize];

            for (int batch = 0; batch < batchSize; batch++)
            {
                Vector<float> sumMask = Vector<float>.Zero;
                Vector<float>[] sumEmbeddings = new Vector<float>[embeddingSize];

                for (int i = 0; i < embeddingSize; i++)
                    sumEmbeddings[i] = Vector<float>.Zero;

                for (int seq = 0; seq < sequenceLength; seq++)
                {
                    long mask = attentionMask[batch * sequenceLength + seq];
                    if (mask == 0)
                        continue;

                    for (int emb = 0; emb < embeddingSize; emb++)
                    {
                        int index = batch * sequenceLength * embeddingSize + seq * embeddingSize + emb;
                        float value = embeddings[index];
                        sumEmbeddings[emb] += new Vector<float>(value);
                    }
                    sumMask += new Vector<float>(1);
                }

                for (int emb = 0; emb < embeddingSize; emb++)
                {
                    float sum = Vector.Dot(sumEmbeddings[emb], Vector<float>.One);
                    float maskSum = Vector.Dot(sumMask, Vector<float>.One);
                    result[batch * embeddingSize + emb] = sum / maskSum;
                }
            }

            return result;
        }

        public static float[] NormalizeAndDivide(float[] sentenceEmbeddings, long[] shape)
        {
            long numSentences = shape[0];
            int embeddingSize = (int)shape[2];

            float[] result = new float[sentenceEmbeddings.Length];
            int vectorSize = Vector<float>.Count;

            // Compute Frobenius (L2) norms
            float[] norms = new float[numSentences];

            for (int i = 0; i < numSentences; i++)
            {
                Vector<float> sumSquares = Vector<float>.Zero;
                for (int j = 0; j < embeddingSize; j += vectorSize)
                {
                    int index = i * embeddingSize + j;
                    Vector<float> vec = new Vector<float>(sentenceEmbeddings, index);
                    sumSquares += vec * vec; // Element-wise squaring and summing
                }
                norms[i] = (float)Math.Sqrt(Vector.Dot(sumSquares, Vector<float>.One)); // Take square root of sum of squares
                norms[i] = Math.Max(norms[i], 1e-12f); // Clamping to avoid division by zero
            }

            // Normalize and divide
            for (int i = 0; i < numSentences; i++)
            {
                float norm = norms[i];
                for (int j = 0; j < embeddingSize; j += vectorSize)
                {
                    int index = i * embeddingSize + j;
                    Vector<float> vec = new Vector<float>(sentenceEmbeddings, index);
                    Vector<float> normalizedVec = vec / new Vector<float>(norm);
                    normalizedVec.CopyTo(result, index);
                }
            }

            return result;
        }

        public async Task<List<TextChunk>> Search(string searchTerm, int top = 5, int range = 3)
        {
            List<TextChunk> contents = [];
            if (!IsReady)
            {
                return contents;
            }

            var searchVector = await GetEmbeddingsAsync(searchTerm).ConfigureAwait(false);
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
                int chunkSize = 32;
                for (int i = 0; i < _content.Count; i += chunkSize)
                {
                    if (ct.IsCancellationRequested)
                    {
                        return;
                    }
                    var chunk = _content.Skip(i).Take(chunkSize).ToList();
                    var vectors = await GetEmbeddingsAsync(chunk.Select(c => c.Text!).ToArray()).ConfigureAwait(false);
                    for (int j = 0; j < chunk.Count; j++)
                    {
                        chunk[j].Vectors = vectors[j];
                    }

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
        public List<(long InputIds, long TokenTypeIds, long AttentionMask)> CustomEncode(params string[] texts)
        {
            List<List<int>> list = [];
            foreach (string text in texts)
            {
                List<string> innerList = ["[CLS]"];
                innerList.AddRange(TokenizeSentence(text));
                innerList.Add("[SEP]");
                list.Add(innerList.SelectMany(TokenizeSubwords).Select(s => s.VocabularyIndex).ToList());
            }

            int maxLength = list.Max(s => s.Count());

            for (int i = 0; i < list.Count; i++)
            {
                var innerList = list[i];
                if (maxLength - innerList.Count() > 0)
                {
                    list[i] = innerList.Concat(Enumerable.Repeat(0, maxLength - innerList.Count())).ToList();
                }
            }
            List<int> flatList = list.SelectMany(s => s).ToList();

            List<long> second = Enumerable.Repeat(0L, flatList.Count).ToList();
            List<long> third = AttentionMask(flatList);

            return flatList.Select((t, i) => ((long)t, second[i], third[i])).ToList();
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
                string? text2 = null;
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

        private List<long> AttentionMask(List<int> tokens)
        {
            return tokens.Select(index => index == 0 ? (long)0 : 1).ToList();
        }
    }
}
