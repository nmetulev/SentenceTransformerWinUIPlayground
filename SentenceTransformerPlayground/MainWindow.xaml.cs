using BERTTokenizers.Base;
using Microsoft.ML.OnnxRuntime;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Documents;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Text.RegularExpressions;
using System.Timers;
using TorchSharp;

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
        private InferenceSession _inferenceSession;
        private Timer _timer;
        private Vector[] _embeddings;
        private string[] _content;

        public MainWindow()
        {
            this.InitializeComponent();
            {
                _timer = new Timer();
                _timer.Interval = 200;
                _timer.Elapsed += new ElapsedEventHandler(OnTimedEvent);
            }

        }


        private void InitModel()
        {
            if (_inferenceSession != null)
            {
                return;
            }


            var sessionOptions = new SessionOptions();
            sessionOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO;
            sessionOptions.AppendExecutionProvider_DML(1); //hardcoded to my machine - how do I get the device id?
            _inferenceSession = new InferenceSession($@"{modelDir}\model.onnx", sessionOptions);
        }

        // TODO: run multiple sentences at once
        private float[] GetEmbeddings(params string[] sentences)
        {
            InitModel();
            var tokenizer = new MyTokenizer($@"{modelDir}\vocab.txt");
            var tokens = tokenizer.Tokenize(sentences);
            var encoded = tokenizer.Encode(tokens.Count(), sentences);

            var input = new ModelInput()
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

            using var output = _inferenceSession.Run(runOptions, inputs, _inferenceSession.OutputNames);
            var data = output.ToList()[0].GetTensorDataAsSpan<float>().ToArray();


            var sentence_embeddings = MeanPooling(data, input.AttentionMask, sentences.Length, input.AttentionMask.Length, 384);
            var denom = sentence_embeddings.norm(1, true, 2).clamp_min(1e-12).expand_as(sentence_embeddings);
            var results = sentence_embeddings / denom;
            return results.data<float>().ToArray();
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

        private void myButton_Click(object sender, RoutedEventArgs e)
        {
            var sentence = "this is a test";

            var embeddings = GetEmbeddings(sentence);
        }

        private void ContentTextBox_TextChanged(object sender, RoutedEventArgs e)
        {
            ContentTextBox.Document.GetText(Microsoft.UI.Text.TextGetOptions.None, out string text);
            text = text.Trim();
            _content = text.Split('.', '\r', '\n');
            _content = _content.Where(x => !string.IsNullOrWhiteSpace(x)).ToArray();
            
            if (_content.Length == 0)
            {
                return;
            }

            var vectors = new Vector[_content.Length];
            for (int i = 0; i < _content.Length; i++)
            {
                var content = Regex.Replace(_content[i], @"[^\u0000-\u007F]", "");
                vectors[i] = new Vector { data = GetEmbeddings(content) };
            }
            _embeddings = vectors;
        }

        public int[] CalculateRanking(Vector searchVector, Vector[] vectors)
        {
            float[] scores = new float[vectors.Length];
            int[] indexranks = new int[vectors.Length];

            for (int i = 0; i < vectors.Length; i++)
            {
                var score = CosineSimilarity(vectors[i].data, searchVector.data);
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
        private void OnTimedEvent(object sender, ElapsedEventArgs e)
        {
            _timer.Stop();
            this.DispatcherQueue.TryEnqueue(() =>
            {
                var searchVector = GetEmbeddings(SearchTextBox.Text);
                var ranking = CalculateRanking(new Vector { data = searchVector }, _embeddings);

                FoundSentenceTextBlock.Text = "Relevant Sentence: " + _content[ranking[0]];
            });
           
            // handle search
        }
    }

    public struct Vector
    {
        public float[] data;
    }

    public class ModelInput
    {
        public long[] InputIds { get; set; }

        public long[] AttentionMask { get; set; }

        public long[] TokenTypeIds { get; set; }
    }

    public class MyTokenizer : UncasedTokenizer
    {
        public MyTokenizer(string vocabPath) : base(vocabPath){ }
    }
}
