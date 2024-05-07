using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Media.Imaging;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using UglyToad.PdfPig;
using VectorDB;
using Windows.Storage;
using Windows.Storage.Pickers;
using Windows.Storage.Streams;
using WinRT.Interop;

namespace SentenceTransformerPlayground
{
    public sealed partial class MainWindow : Window
    {
        private readonly SLMRunner SLMRunner;
        private readonly RAGService RAGService;
        private CancellationTokenSource? cts;
        private List<uint>? selectedPages = null;
        private StorageFile? pdfFile;

        [GeneratedRegex(@"[\u0000-\u001F\u007F-\uFFFF]")]
        private static partial Regex MyRegex();

        public MainWindow()
        {
            SLMRunner = new SLMRunner();
            SLMRunner.ModelLoaded += (sender, e) => DispatcherQueue.TryEnqueue(() => CheckIfReady());

            RAGService = new RAGService();
            RAGService.ResourcesLoaded += (sender, e) => DispatcherQueue.TryEnqueue(() => CheckIfReady());

            Closed += (sender, e) =>
            {
                cts?.Cancel();
                cts = null;
                SLMRunner?.Dispose();
                RAGService?.Dispose();
            };

            InitializeComponent();
        }

        private void CheckIfReady()
        {
            IndexPDFButton.IsEnabled = RAGService.IsModelReady;
            AskSLMButton.IsEnabled = SLMRunner.IsReady && RAGService.IsReady;
            if (RAGService.IsReady)
            {
                IndexPDFButton.Content = "Model Ready";
            }
        }

        private async void Grid_Loaded(object sender, RoutedEventArgs e)
        {
            var localFolder = ApplicationData.Current.LocalFolder;

            pdfFile = (await localFolder.TryGetItemAsync("CurrentPDF.pdf")) as StorageFile;
            if (pdfFile is not null)
            {
                ShowPDFPage.IsEnabled = true;
            }

            await Task.Run(() => Task.WhenAll(SLMRunner.InitializeAsync(), RAGService.InitializeAsync()));
        }

        private async void IndexPDFButton_Click(object sender, RoutedEventArgs e)
        {
            IndexPDFButton.IsEnabled = false;

            var picker = new FileOpenPicker
            {
                CommitButtonText = "Index PDF",
                ViewMode = PickerViewMode.Thumbnail
            };

            InitializeWithWindow.Initialize(picker, WindowNative.GetWindowHandle(this));

            picker.FileTypeFilter.Add(".pdf");

            var newPdfFile = await picker.PickSingleFileAsync();
            if (newPdfFile == null)
            {
                IndexPDFButton.IsEnabled = RAGService.IsModelReady;
                return;
            }

            IndexPDFProgressStackPanel.Visibility = Visibility.Visible;
            IndexPDFProgressBar.Minimum = 0;
            IndexPDFProgressBar.Maximum = 1;
            IndexPDFProgressBar.Value = 0;
            IndexPDFProgressTextBlock.Text = "Reading PDF...";

            var localFolder = ApplicationData.Current.LocalFolder;
            pdfFile = await newPdfFile.CopyAsync(localFolder, "CurrentPDF.pdf", NameCollisionOption.ReplaceExisting);
            ShowPDFPage.IsEnabled = true;

            await Task.Delay(1).ConfigureAwait(false);

            var contents = new List<TextChunk>();
            using (PdfDocument document = PdfDocument.Open(pdfFile.Path))
            {
                foreach (var page in document.GetPages())
                {
                    var words = page.GetWords();
                    var builder = string.Join(" ", words);

                    var range = builder
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

            var maxLength = 1024 / 4;
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
                            Text = Regex.Replace(content.Text[index..].Trim(), @"(\.){2,}", ".")
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
                        Text = Regex.Replace(content.Text[index..lastIndexOfBreak].Trim(), @"(\.){2,}", ".")
                    }); ;

                    index = lastIndexOfBreak + 1;
                }

                contents.RemoveAt(i);
                contents.InsertRange(i, contentChunks);
                i += contentChunks.Count - 1;
            }

            DispatcherQueue.TryEnqueue(() =>
            {
                IndexPDFProgressBar.Minimum = 0;
                IndexPDFProgressBar.Maximum = contents.Count;
                IndexPDFProgressBar.Value = 0;
            });

            Stopwatch sw = Stopwatch.StartNew();

            void UpdateProgress(float progress)
            {
                var elapsed = sw.Elapsed;
                if (progress == 0)
                {
                    progress = 0.0001f;
                }

                var remaining = TimeSpan.FromSeconds((long)(elapsed.TotalSeconds / progress * (1 - progress) / 5) * 5);

                IndexPDFProgressBar.Value = progress * contents.Count;
                IndexPDFProgressTextBlock.Text = $"Indexing PDF... {progress:P0} ({remaining})";
            }

            if (cts != null)
            {
                cts.Cancel();
                cts = null;
                AskSLMButton.Content = "Answer";
                return;
            }

            cts = new CancellationTokenSource();

            await RAGService.InitializeAsync(contents, (sender, progress) =>
            {
                DispatcherQueue.TryEnqueue(() =>
                {
                    UpdateProgress(progress);
                });
            }, cts.Token);

            cts = null;

            DispatcherQueue.TryEnqueue(() =>
            {
                IndexPDFProgressTextBlock.Text = "Indexing PDF... Done!";
            });

            await Task.Delay(1000);

            DispatcherQueue.TryEnqueue(() =>
            {
                IndexPDFProgressStackPanel.Visibility = Visibility.Collapsed;
                IndexPDFButton.IsEnabled = RAGService.IsModelReady;
            });
        }

        private async void AskSLMButton_Click(object sender, RoutedEventArgs e)
        {
            if (cts != null)
            {
                cts.Cancel();
                cts = null;
                AskSLMButton.Content = "Answer";
                return;
            }

            var prompt = """
<|system|>
You are a helpful assistant helping answer questions about this information:
""";

            cts = new CancellationTokenSource();
            AskSLMButton.Content = "Cancel";

            SLMRunner.SearchMaxLength = Math.Min(4096, Math.Max(1024, (int)(RAGService.MaxDedicatedVideoMemory / (1024 * 1024))));

            List<TextChunk> contents = await RAGService.Search(SearchTextBox.Text, 3, 1);

            selectedPages = contents.Select(c => (uint)c.Page).Distinct().ToList();

            PagesUsedRun.Text = $"Using page(s) : {string.Join(", ", selectedPages)}";

            var pagesChunks = contents.GroupBy(c => c.Page).Select(g => new { Page = g.Key, Text = string.Join(Environment.NewLine, g.OrderBy(g => g.TextChunkId).Select(c => c.Text)) }).ToList();

            prompt += string.Join(Environment.NewLine, pagesChunks.Select(c => $"{Environment.NewLine}Page {c.Page}: {c.Text}"));

            prompt += $"""
<|end|>
<|user|>
{SearchTextBox.Text}<|end|>
<|assistant|>
""";

            AnswerRun.Text = "";
            var fullResult = "";

            await Task.Run(async () =>
            {
                await foreach (var partialResult in SLMRunner.InferStreamingAsync(prompt).WithCancellation(cts.Token))
                {
                    fullResult += partialResult;
                    DispatcherQueue.TryEnqueue(() => AnswerRun.Text = fullResult);
                }
            }, cts.Token);

            cts = null;

            AskSLMButton.Content = "Answer";
        }

        private async void ShowPDFPage_Click(object sender, RoutedEventArgs e)
        {
            if (pdfFile == null || selectedPages == null || selectedPages.Count() == 0)
            {
                return;
            }

            var pdfDocument = await Windows.Data.Pdf.PdfDocument.LoadFromFileAsync(pdfFile).AsTask().ConfigureAwait(false);
            var pageId = selectedPages.First();
            if (pageId < 0 || pdfDocument.PageCount < pageId)
            {
                return;
            }
            var page = pdfDocument.GetPage(pageId);
            InMemoryRandomAccessStream inMemoryRandomAccessStream = new();
            var rect = page.Dimensions.TrimBox;
            await page.RenderToStreamAsync(inMemoryRandomAccessStream).AsTask().ConfigureAwait(false);

            DispatcherQueue.TryEnqueue(async () =>
            {
                BitmapImage bitmapImage = new();

                await bitmapImage.SetSourceAsync(inMemoryRandomAccessStream);

                PdfImage.Source = bitmapImage;

                PdfImage.Visibility = Visibility.Visible;
            });
        }

        private void PdfImage_Tapped(object sender, Microsoft.UI.Xaml.Input.TappedRoutedEventArgs e)
        {
            PdfImage.Visibility = Visibility.Collapsed;
        }
    }
}
