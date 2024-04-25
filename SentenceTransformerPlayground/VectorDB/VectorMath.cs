#nullable enable

namespace VectorDB
{
    public static class VectorMath
    {
        public static float DotProduct(float[] a, float[] b)
        {
            float sum = 0;
            for (int i = 0; i < a.Length; i++)
            {
                sum += a[i] * b[i];
            }

            return sum;
        }
    }
}