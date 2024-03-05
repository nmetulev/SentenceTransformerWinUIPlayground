using BERTTokenizers.Base;
using Microsoft.ML.OnnxRuntime;
using Microsoft.UI.Xaml;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
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
        public MainWindow()
        {
            this.InitializeComponent();
        }

        private void myButton_Click(object sender, RoutedEventArgs e)
        {
            // model from https://huggingface.co/optimum/all-MiniLM-L6-v2
            var modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "model");

            var sentence = "this is a test";
            var tokenizer = new MyTokenizer($@"{modelPath}\vocab.txt");
            var tokens = tokenizer.Tokenize(sentence);
            var encoded = tokenizer.Encode(tokens.Count(), sentence);

            var input = new ModelInput()
            {
                InputIds = encoded.Select(t => t.InputIds).ToArray(),
                AttentionMask = encoded.Select(t => t.AttentionMask).ToArray(),
                TokenTypeIds = encoded.Select(t => t.TokenTypeIds).ToArray(),
            };

            var runOptions = new RunOptions();
            var session = new InferenceSession($@"{modelPath}\model.onnx");

            // Create input tensors over the input data.
            using var inputIdsOrtValue = OrtValue.CreateTensorValueFromMemory(input.InputIds,
                  new long[] { 1, input.InputIds.Length });

            using var attMaskOrtValue = OrtValue.CreateTensorValueFromMemory(input.AttentionMask,
                  new long[] { 1, input.AttentionMask.Length });

            using var typeIdsOrtValue = OrtValue.CreateTensorValueFromMemory(input.TokenTypeIds,
                  new long[] { 1, input.TokenTypeIds.Length });

            var inputs = new Dictionary<string, OrtValue>
            {
                { "input_ids", inputIdsOrtValue },
                { "attention_mask", attMaskOrtValue },
                { "token_type_ids", typeIdsOrtValue }
            };

            using var output = session.Run(runOptions, inputs, session.OutputNames);
            var data = output.ToList()[0].GetTensorDataAsSpan<float>().ToArray();


            var sentence_embeddings = MeanPooling(data, input.AttentionMask, 1, input.AttentionMask.Length, 384);
            var denom = sentence_embeddings.norm(1, true, 2).clamp_min(1e-12).expand_as(sentence_embeddings);
            var results = sentence_embeddings / denom;
            var array = results.data<float>();
        }
 
        public static torch.Tensor MeanPooling(float[] embeddings, long[] attentionMask, long batchSize, long sequence, int hiddenSize)
        {
            var tokenEmbeddings = torch.tensor(embeddings, new[] { batchSize, sequence, hiddenSize });
            var attentionMaskExpanded = torch.tensor(attentionMask, new[] { batchSize, sequence }).unsqueeze(-1).expand(tokenEmbeddings.shape).@float();

            var sumEmbeddings = (tokenEmbeddings * attentionMaskExpanded).sum(new[] { 1L });
            var sumMask = attentionMaskExpanded.sum(new[] { 1L }).clamp(1e-9, float.MaxValue);

            return sumEmbeddings / sumMask;
        }

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
