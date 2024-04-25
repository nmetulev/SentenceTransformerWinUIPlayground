#nullable enable

namespace VectorDB
{
    /// <summary>
    /// Data objects that should be stored insied a collection
    /// </summary>
    public interface IVectorObject
    {
        float[] GetVector();
    }
}
