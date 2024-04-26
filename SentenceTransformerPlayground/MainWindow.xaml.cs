using Microsoft.UI.Xaml;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using UglyToad.PdfPig;
using VectorDB;
using Windows.Storage;
using Windows.Storage.Pickers;
using WinRT.Interop;

namespace SentenceTransformerPlayground
{
    public sealed partial class MainWindow : Window
    {
        private readonly SLMRunnerLlamaSharp SLMRunner;
        private readonly RAGService RAGService;

        [GeneratedRegex(@"[\u0000-\u001F\u007F-\uFFFF]")]
        private static partial Regex MyRegex();

        public MainWindow()
        {
            SLMRunner = new SLMRunnerLlamaSharp();
            SLMRunner.ModelLoaded += (sender, e) => DispatcherQueue.TryEnqueue(() => CheckIfReady());

            RAGService = new RAGService();
            RAGService.ResourcesLoaded += (sender, e) => DispatcherQueue.TryEnqueue(() => CheckIfReady());

            InitializeComponent();
        }

        private void CheckIfReady()
        {
            IndexPDFButton.IsEnabled = RAGService.IsModelReady;
            AskSLMButton.IsEnabled = SLMRunner.IsReady && RAGService.IsReady;
        }

        private async void Grid_Loaded(object sender, RoutedEventArgs e)
        {
            await Task.Run(() => Task.WhenAll(SLMRunner.InitializeAsync(), RAGService.InitializeAsync()));
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
            if (file == null)
            {
                return;
            }

            IndexPDFProgressBar.Visibility = Visibility.Visible;
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
                            Text = content.Text[index..].Trim()
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
                        Text = content.Text[index..lastIndexOfBreak].Trim()
                    });

                    index = lastIndexOfBreak + 1;
                }

                contents.RemoveAt(i);
                contents.InsertRange(i, contentChunks);
                i += contentChunks.Count - 1;
            }

            IndexPDFProgressBar.Minimum = 0;
            IndexPDFProgressBar.Maximum = contents.Count;

            await RAGService.InitializeAsync(contents, (sender, progress) =>
            {
                DispatcherQueue.TryEnqueue(() => IndexPDFProgressBar.Value = progress * contents.Count);
            });

            IndexPDFProgressBar.Visibility = Visibility.Collapsed;
        }

        private async void AskSLMButton_Click(object sender, RoutedEventArgs e)
        {
            AskSLMButton.IsEnabled = false;

            var prompt = """
<|system|>
You are a helpful assistant helping answer questions about this information:
""";
            
            List<TextChunk> contents = await RAGService.Search(SearchTextBox.Text, 2, 3);
            prompt += string.Join(Environment.NewLine, contents.Distinct().Select(c => $"Page {c.Page}: {c.Text}" + Environment.NewLine));

            prompt += $"""
<|end|>
<|user|>
{SearchTextBox.Text}
<|end|>
<|assistant|>
""";

            FoundSentenceTextBlock.Text = "";
            var fullResult = "";
            await Task.Run(async () =>
            {
                await foreach (var partialResult in SLMRunner.InferStreaming(prompt))
                {
                    fullResult += partialResult;
                    DispatcherQueue.TryEnqueue(() => FoundSentenceTextBlock.Text = fullResult);
                }
            });

            AskSLMButton.IsEnabled = true;
        }
    }
}
