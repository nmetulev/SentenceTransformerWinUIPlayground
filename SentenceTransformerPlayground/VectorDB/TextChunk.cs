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
            Page = 0;
            Text = null;
            Vectors = Array.Empty<float>();
        }

        public TextChunk(TextChunk textChunk)
        {
            Page = textChunk.Page;
            Text = textChunk.Text;
            Vectors = textChunk.Vectors;
        }

        public int TextChunkId { get; set; }
        public int Page { get; set; }
        public string? Text { get; set; }
        public float[] Vectors { get; set; }
    }
}
