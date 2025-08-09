import asyncio
import chromadb


async def main():
    print("Hello from chroma-db!")
    client = await chromadb.AsyncHttpClient()
    collection = await client.create_collection(name="my_collection")
    await collection.add(
        documents=["hello world"],
        ids=["id1"]
    )


if __name__ == "__main__":
    asyncio.run(main())
