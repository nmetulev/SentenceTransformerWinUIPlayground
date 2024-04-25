#nullable enable

using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace VectorDB
{
    /// <summary>
    /// A collection of <see cref="IVectorObject"/>
    /// </summary>
    public class VectorCollection<T> where T : IVectorObject
    {
        public VectorCollection()
        {
            Dimensions = 0;
            Objects = [];
        }

        public VectorCollection(int dimensions, List<T>? objects = null)
        {
            Dimensions = dimensions;
            Objects = objects ?? [];
        }

        public int VectorCollectionId { get; set; }

        public int Dimensions { get; set; }

        public List<T> Objects { get; set; }

        public T GetItem(int index)
        {
            return Objects[index];
        }

        /// <summary>
        /// Find nearest vector by comparing all vectors to the query
        /// </summary>
        /// <param name="query"></param>
        /// <returns></returns>
        /// <exception cref="NotImplementedException"></exception>
        public T FindNearest(float[] query)
        {
            return Objects[FindNearestIndex(query)];
        }

        /// <summary>
        /// Find the index of the nearest vector by comparing all vectors to the query
        /// </summary>
        /// <param name="query"></param>
        /// <returns></returns>
        /// <exception cref="NotImplementedException"></exception>
        public int FindNearestIndex(float[] query)
        {
            float maxDotProduct = 0;
            int bestIndex = 0;

            for (int i = 0; i < Objects.Count; i++)
            {
                float dotProd = VectorMath.DotProduct(Objects[i].GetVector(), query);
                if (dotProd > maxDotProduct)
                {
                    maxDotProduct = dotProd;
                    bestIndex = i;
                }
            }
            return bestIndex;
        }

        public static float DotProduct(float[] a, float[] b)
        {
            float sum = 0;
            for (int i = 0; i < a.Length; i++)
            {
                sum += a[i] * b[i];
            }

            return sum;
        }

        public async Task SaveToDiskAsync(string fileName)
        {
            using var db = new VectorDBContext<T>(fileName);
            if (File.Exists(db.DbPath))
            {
                File.Delete(db.DbPath);
            }

            await db.Database.EnsureCreatedAsync();

            await db.AddAsync(this);

            await db.SaveChangesAsync();
        }

        public static async Task<VectorCollection<T>?> LoadFromDiskAsync(string fileName)
        {
            using var db = new VectorDBContext<T>(fileName);

            try
            {
                return await db.VectorCollections
                    .Include(v => v.Objects).FirstAsync();
            }
            catch (Exception)
            {
                return null;
            }
        }
    }

    public class VectorDBContext<T> : DbContext where T : IVectorObject
    {
        public DbSet<VectorCollection<T>> VectorCollections { get; set; }

        public string DbPath { get; }

        public VectorDBContext(string fileName)
        {
            DbPath = Path.Join(Windows.Storage.ApplicationData.Current.LocalFolder.Path, fileName);
        }

        protected override void OnConfiguring(DbContextOptionsBuilder options) => options.UseSqlite($"Data Source={DbPath}");
    }
}
