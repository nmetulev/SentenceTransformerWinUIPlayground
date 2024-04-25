#nullable enable

using BERTTokenizers.Base;
using Microsoft.ML.OnnxRuntime;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using SharpDX.DXGI;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Timers;
using TorchSharp;
using UglyToad.PdfPig;
using VectorDB;
using Windows.Storage;
using Windows.Storage.Pickers;
using WinRT.Interop;
using static TorchSharp.torch.nn;

// To learn more about WinUI, the WinUI project structure,
// and more about our project templates, see: http://aka.ms/winui-project-info.

namespace SentenceTransformerPlayground
{
    /// <summary>
    /// An empty window that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainWindow : Window
    {
        // model from https://huggingface.co/optimum/all-MiniLM-L6-v2
        private string modelDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "model");
        private InferenceSession? _inferenceSession;
        private Timer _timer;
        private List<TextChunk>? _content;
        private VectorCollection<TextChunk>? _embeddings = null;
        private MyTokenizer? tokenizer = null;

        [GeneratedRegex(@"[\u0000-\u001F\u007F-\uFFFF]")]
        private static partial Regex MyRegex();

        public MainWindow()
        {
            this.InitializeComponent();

            _timer = new Timer
            {
                Interval = 200
            };
            _timer.Elapsed += new ElapsedEventHandler(OnTimedEvent);
        }

        [MemberNotNull(nameof(_inferenceSession))]
        private void InitModel()
        {
            if (_inferenceSession != null)
            {
                return;
            }

            var factory1 = new Factory1();
            int deviceId = 0;
            Adapter1? selectedAdapter = null;
            for (int i = 0; i < factory1.GetAdapterCount1(); i++)
            {
                Adapter1 adapter = factory1.GetAdapter1(i);
                Debug.WriteLine($"Adapter {i}:");
                Debug.WriteLine($"\tDescription: {adapter.Description1.Description}");
                Debug.WriteLine($"\tDedicatedVideoMemory: {(long)adapter.Description1.DedicatedVideoMemory / 1000000000}GB");
                if (selectedAdapter == null || (long)adapter.Description1.DedicatedVideoMemory > (long)selectedAdapter.Description1.DedicatedVideoMemory)
                {
                    selectedAdapter = adapter;
                    deviceId = i;
                }
            }

            var sessionOptions = new SessionOptions
            {
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO
            };
            sessionOptions.AppendExecutionProvider_DML(deviceId);
            _inferenceSession = new InferenceSession($@"{modelDir}\model.onnx", sessionOptions);
        }

        // TODO: run multiple sentences at once
        private ValueTask<float[]> GetEmbeddingsAsync(params string[] sentences)
        {
            InitModel();
            tokenizer ??= new MyTokenizer($@"{modelDir}\vocab.txt");

            var tokensCount = tokenizer.Tokenize(sentences).Count();

            var encoded = tokenizer.Encode(tokensCount, sentences);

            var input = new ModelInput
            {
                InputIds = encoded.Select(t => t.InputIds).ToArray(),
                AttentionMask = encoded.Select(t => t.AttentionMask).ToArray(),
                TokenTypeIds = encoded.Select(t => t.TokenTypeIds).ToArray(),
            };

            var runOptions = new RunOptions();

            // Create input tensors over the input data.
            using var inputIdsOrtValue = OrtValue.CreateTensorValueFromMemory(input.InputIds,
                  new long[] { sentences.Length, input.InputIds.Length });

            using var attMaskOrtValue = OrtValue.CreateTensorValueFromMemory(input.AttentionMask,
                  new long[] { sentences.Length, input.AttentionMask.Length });

            using var typeIdsOrtValue = OrtValue.CreateTensorValueFromMemory(input.TokenTypeIds,
                  new long[] { sentences.Length, input.TokenTypeIds.Length });

            var inputs = new Dictionary<string, OrtValue>
            {
                { "input_ids", inputIdsOrtValue },
                { "attention_mask", attMaskOrtValue },
                { "token_type_ids", typeIdsOrtValue }
            };

            try
            {
                using var output = _inferenceSession.Run(runOptions, inputs, _inferenceSession.OutputNames);
                var data = output.ToList()[0].GetTensorDataAsSpan<float>().ToArray();

                var sentence_embeddings = MeanPooling(data, input.AttentionMask, sentences.Length, input.AttentionMask.Length, 384);
                var denom = sentence_embeddings.norm(1, true, 2).clamp_min(1e-12).expand_as(sentence_embeddings);
                var results = sentence_embeddings / denom;
                return ValueTask.FromResult(results.data<float>().ToArray());
            }
            catch (Exception e)
            {
                Debug.WriteLine(e.Message);
                return ValueTask.FromResult(new float[0]);
            }
        }

        public static torch.Tensor MeanPooling(float[] embeddings, long[] attentionMask, long batchSize, long sequence, int hiddenSize)
        {
            var tokenEmbeddings = torch.tensor(embeddings, new[] { batchSize, sequence, hiddenSize });
            var attentionMaskExpanded = torch.tensor(attentionMask, new[] { batchSize, sequence }).unsqueeze(-1).expand(tokenEmbeddings.shape).@float();

            var sumEmbeddings = (tokenEmbeddings * attentionMaskExpanded).sum(new[] { 1L });
            var sumMask = attentionMaskExpanded.sum(new[] { 1L }).clamp(1e-9, float.MaxValue);

            return sumEmbeddings / sumMask;
        }

        public static float CheckOverflow(double x)
        {
            if (x >= double.MaxValue)
            {
                throw new OverflowException("operation caused overflow");
            }
            return (float)x;
        }
        public static float DotProduct(float[] a, float[] b)
        {
            float result = 0.0f;
            for (int i = 0; i < a.Length; i++)
            {
                result = CheckOverflow(result + CheckOverflow(a[i] * b[i]));
            }
            return result;
        }
        public static float Magnitude(float[] v)
        {
            float result = 0.0f;
            for (int i = 0; i < v.Length; i++)
            {
                result = CheckOverflow(result + CheckOverflow(v[i] * v[i]));
            }
            return (float)Math.Sqrt(result);
        }
        public static float CosineSimilarity(float[] v1, float[] v2)
        {
            if (v1.Length != v2.Length)
            {
                throw new ArgumentException("Vectors must have the same length.");
            }
            int size = v1.Length;
            float m1 = Magnitude(v1);
            float m2 = Magnitude(v2);
            /*                        var normalizedList1 = raw1.Select(o => o / m1).ToArray();
                                    var normalizedList2 = raw2.Select(o => o / m2).ToArray();
            */
            /*// Vectors should already be normalized.
            if (Math.Abs(m1 - m2) > 0.4f || Math.Abs(m1 - 1.0f) > 0.4f)
            {
                throw new InvalidOperationException("Vectors are not normalized.");
            }*/
            return DotProduct(v1, v2);
        }

        private async void IndexPDFButton_Click(object sender, RoutedEventArgs e)
        {
            var picker = new FileOpenPicker
            {
                ViewMode = PickerViewMode.Thumbnail
            };

            InitializeWithWindow.Initialize(picker, WindowNative.GetWindowHandle(this));

            picker.FileTypeFilter.Add(".pdf");

            StorageFile file = await picker.PickSingleFileAsync();
            IndexPDFProgressBar.Visibility = Visibility.Visible;
            if (file != null)
            {
                var contents = new List<TextChunk>();
                using (PdfDocument document = PdfDocument.Open(file.Path))
                {
                    foreach (var page in document.GetPages())
                    {
                        var range = page.Text
                                .Split('\r', '\n', StringSplitOptions.RemoveEmptyEntries)
                                .Select(x => MyRegex().Replace(x, ""))
                                .Where(x => !string.IsNullOrWhiteSpace(x))
                                .Select(x => new TextChunk
                                {
                                    Text = x,
                                    Page = page.Number,
                                });

                        contents.AddRange(range);
                    }
                }

                var maxLength = 1024;
                for (int i = 0; i < contents.Count; i++)
                {
                    var content = contents[i];
                    int index = 0;
                    var contentChunks = new List<TextChunk>();
                    while (index < content.Text!.Length)
                    {
                        if (index + maxLength >= content.Text.Length)
                        {
                            contentChunks.Add(new TextChunk(content)
                            {
                                Text = content.Text.Substring(index).Trim()
                            });
                            break;
                        }

                        int lastIndexOfBreak = content.Text.LastIndexOf(' ', index + maxLength, maxLength);
                        if (lastIndexOfBreak <= index)
                        {
                            lastIndexOfBreak = index + maxLength;
                        }

                        contentChunks.Add(new TextChunk(content)
                        {
                            Text = content.Text.Substring(index, lastIndexOfBreak - index).Trim()
                        });

                        index = lastIndexOfBreak + 1;
                    }

                    contents.RemoveAt(i);
                    contents.InsertRange(i, contentChunks);
                    i+= contentChunks.Count - 1;
                }

                _content = contents.ToList();

                if (_content.Count == 0)
                {
                    return;
                }

                IndexPDFProgressBar.Minimum = 0;
                IndexPDFProgressBar.Maximum = _content.Count;

                Stopwatch stopwatch = Stopwatch.StartNew();


                await Task.Run(async () =>
                {
                    for (int i = 0; i < _content.Count; i++)
                    {
                        _content[i].TextVectors = await GetEmbeddingsAsync(_content[i].Text!).ConfigureAwait(false);

                        DispatcherQueue.TryEnqueue(() => IndexPDFProgressBar.Value = i);
                    }
                });

                _embeddings = new VectorCollection<TextChunk>(_content.Count, _content);
                await _embeddings.SaveToDiskAsync("vectors.vec");

                stopwatch.Stop();
                Debug.WriteLine($"Indexing took {stopwatch.ElapsedMilliseconds} ms");
            }
            else
            {
                _embeddings = await VectorCollection<TextChunk>.LoadFromDiskAsync("vectors.vec");
                if (_embeddings != null)
                {
                    _content = _embeddings.Objects.ToList();
                }
            }

            IndexPDFProgressBar.Visibility = Visibility.Collapsed;

        }

        public int[] CalculateRanking(float[] searchVector)
        {
            if (_embeddings == null || _content == null)
            {
                throw new InvalidOperationException("Embeddings or content is not loaded");
            }

            float[] scores = new float[_embeddings.Dimensions];
            int[] indexranks = new int[_embeddings.Dimensions];

            for (int i = 0; i < _embeddings.Dimensions; i++)
            {
                var score = CosineSimilarity(_embeddings.GetItem(i).TextVectors!, searchVector);
                scores[i] = score;
            }

            var indexedFloats = scores.Select((value, index) => new { Value = value, Index = index })
              .ToArray();

            // Sort the indexed floats by value in descending order
            Array.Sort(indexedFloats, (a, b) => b.Value.CompareTo(a.Value));

            // Extract the top k indices
            indexranks = indexedFloats.Select(item => item.Index).ToArray();

            return indexranks;
        }

        private void SearchTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            _timer.Stop();
            _timer.Start();
        }

        private void OnTimedEvent(object? sender, ElapsedEventArgs e)
        {
            _timer.Stop();
            if (_embeddings == null || _content == null)
            {
                return;
            }

            this.DispatcherQueue.TryEnqueue(async () =>
            {
                var searchVector = await GetEmbeddingsAsync(SearchTextBox.Text);
                var ranking = CalculateRanking(searchVector);
                List<TextChunk> contents = new List<TextChunk>();
                var top = 5;
                var range = 3;
                for (int i = 0; i < top; i++)
                {
                    var indexMin = Math.Max(0, ranking[i] - range);
                    var indexMax = Math.Min(indexMin + range, _content.Count);
                    contents.AddRange(_content.Skip(indexMin).Take(indexMax - indexMin).ToList());
                }

                FoundSentenceTextBlock.Text = "Relevant Sentences: " + string.Join(Environment.NewLine, contents.Distinct().Select(x => x.Text));
                foreach (var content in contents)
                {
                    Debug.WriteLine($"Page: {content.Page}");
                    Debug.WriteLine($"Text: {content.Text}");
                }
            });

            // handle search
        }
    }

    public class ModelInput
    {
        public required long[] InputIds { get; init; }

        public required long[] AttentionMask { get; init; }

        public required long[] TokenTypeIds { get; init; }
    }

    public class MyTokenizer : UncasedTokenizer
    {
        public MyTokenizer(string vocabPath) : base(vocabPath) { }
    }
}
