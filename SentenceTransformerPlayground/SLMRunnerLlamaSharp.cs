using System;
using System.IO;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Diagnostics;
using LLama;
using LLama.Common;
using LLama.Native;

namespace SentenceTransformerPlayground
{
    public class SLMRunnerLlamaSharp : IDisposable
    {
        private readonly string ModelDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "model", "Phi-3-mini-4k-instruct-gguf\\Phi-3-mini-4k-instruct-q4.gguf");

        private InteractiveExecutor? _executor;

        public event EventHandler? ModelLoaded = null;

        public bool IsReady => _executor != null ;

        public void Dispose()
        {
            
        }

        public IAsyncEnumerable<string> InferStreaming(string prompt)
        {
            if (!IsReady)
            {
                throw new InvalidOperationException("Model is not ready");
            }

            var session = new ChatSession(_executor);

            InferenceParams inferenceParams = new InferenceParams()
            {
                AntiPrompts = new List<string> { "<|end|>", "<|user|>", "<|system|>" } // Stop generation once antiprompts appear.
            };

            return session.ChatAsync(new ChatHistory.Message(AuthorRole.User, prompt), inferenceParams);
        }


        public Task InitializeAsync()
        {
            return Task.Run(() =>
            {
                var sw = Stopwatch.StartNew();

                NativeLibraryConfig.Instance.WithLogCallback((level, message) =>
                {
                    Debug.WriteLine($"[LLama] {level}: {message}");
                });
                var parameters = new ModelParams(ModelDir)
                {
                    ContextSize = 1024 * 2, // The longest length of chat as memory.
                    GpuLayerCount = 20 // How many layers to offload to GPU. Please adjust it according to your GPU memory.
                };

                var model = LLamaWeights.LoadFromFile(parameters);
                var context = model.CreateContext(parameters);
                _executor = new InteractiveExecutor(context);

                sw.Stop();
                Debug.WriteLine($"Model loading took {sw.ElapsedMilliseconds} ms");
                ModelLoaded?.Invoke(this, EventArgs.Empty);
            });
        }
    }
}