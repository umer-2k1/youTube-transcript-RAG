from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from rag_pipeline import store_in_qdrant, get_context_from_store

ollama_model = "llama3.2:latest"

model = OllamaLLM(model=ollama_model, streaming=True)


template = """
You are a helpful YouTube assistant that answers questions using the video's transcript.
Use only the context provided. If the answer isn't in the context, say you don't know.

Context:
{context}

Question:
{question}
"""


prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model


def main():
    url = "https://www.youtube.com/watch?v=reUZRyXxUs4"
    print("🔹 Storing in Qdrant...")
    vectordb = store_in_qdrant(video_url=url)
    print("✅ Setup complete! You can now ask questions.", vectordb)

    while True:
        print("\n\n ----------------------------------")
        print("\n\n ----------------------------------")
        print("\n\n ----------------------------------")

        question = input("Ask a question about the video (or type 'q' to quit):\n")

        print("\n\n")
        if question.lower() == "q":
            break
        context = get_context_from_store(question)
        print(context)

        print("🔹 Answering...")
        print("context....", context)
        for chunk in chain.stream({"context": context, "question": question}):
            print(chunk, end="")
        print("\n\n ----------------------------------")


if __name__ == "__main__":
    main()
