#nullable enable

using Microsoft.EntityFrameworkCore;
using System;

namespace VectorDB
{
    /// <summary>
    /// A chunk of text from a pdf document. Will also contain the page number and the source file.
    /// </summary>
    public class TextChunk : IVectorObject
    {
        public TextChunk()
        {
        }

        public TextChunk(TextChunk textChunk)
        {
            SourceFile = textChunk.SourceFile;
            Page = textChunk.Page;
            OnPageIndex = textChunk.OnPageIndex;
            Text = textChunk.Text;
            TextVectors = textChunk.TextVectors;
        }

        public int TextChunkId { get; set; }
        public string? SourceFile { get; set; }
        public int Page { get; set; }
        public int OnPageIndex { get; set; }
        public string? Text { get; set; }
        public float[]? TextVectors { get; set; }

        public float[] GetVector()
        {
            return TextVectors ?? throw new Exception("TextVectors not set");
        }
    }
}
