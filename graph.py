import os
import typer
from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception_type,
    wait_random_exponential,
)
from lightrag import LightRAG, QueryParam
from lightrag.llm.bedrock import (
    bedrock_complete_if_cache,
    locate_json_string_body_from_string,
)
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
import textract
import asyncio
import nest_asyncio
import dotenv

dotenv.load_dotenv()
nest_asyncio.apply()


class BedrockError(Exception):
    pass


@retry(
    stop=stop_after_attempt(10),
    wait=wait_random_exponential(multiplier=1, max=60),
    retry=retry_if_exception_type(BedrockError),
)
async def bedrock_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
):
    """
    use bedrock model for prompt"""
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    llm_model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    kwargs.pop("stream", None)
    try:
        result = await bedrock_complete_if_cache(
            llm_model_name,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )
    except BedrockError:
        return "{}"
    if keyword_extraction:
        return locate_json_string_body_from_string(result)
    return result


async def initialize_rag(working_dir, llm_model_name="amazon.nova-micro-v1:0"):
    """
    init lightrag / nomic-embed-text (ollama)
    """
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=bedrock_complete,
        llm_model_name=llm_model_name,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model="nomic-embed-text", host="http://localhost:11434"
            ),
        ),
        llm_model_max_async=32,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main(working_dir, llm_model_name):
    """ask questions against your graph"""
    rag = asyncio.run(initialize_rag(working_dir, llm_model_name))
    print("LightRAG CLI. Type your question below (type 'exit' to quit):")
    while True:
        question = input(">> ")
        if question.lower() == "exit":
            print("Exiting LightRAG CLI.")
            break

        resp = rag.query(
            question,
            param=QueryParam(mode="mix"),
        )
        print("Response:", resp)


def populate(path, working_dir, llm_model_name):
    """
    populate knowledge graph from dataset dir
    use textract for csv data
    """
    rag = asyncio.run(initialize_rag(working_dir, llm_model_name))
    if os.path.isfile(path):
        _, ext = os.path.splitext(path)
        if ext.lower() == ".csv":
            text = textract.process(path, encoding="utf-8")
            rag.insert(
                text.decode("utf-8"),
                ids=[os.path.basename(path)],
                file_paths=[path],
            )
        else:
            with open(path, "r", encoding="utf-8") as f:
                print("Inserting data from file:", path)
                rag.insert(f.read(), ids=[os.path.basename(path)], file_paths=[path])
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file_path)
                if ext.lower() == ".csv":
                    text = textract.process(file_path, encoding="utf-8")
                    rag.insert(
                        text.decode("utf-8"),
                        ids=[os.path.basename(file_path)],
                        file_paths=[file_path],
                    )
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        print("Inserting data from file:", file_path)
                        rag.insert(
                            f.read(),
                            ids=[os.path.basename(file_path)],
                            file_paths=[file_path],
                        )
    else:
        print(f"Error: The path '{path}' is neither a file nor a directory.")


app = typer.Typer()


@app.command("populate")
def populate_command(
    working_dir: str = typer.Option(..., help="Working directory for LightRAG"),
    path: str = typer.Option(..., help="File or directory to populate"),
    llm_model_name: str = typer.Option(
        "amazon.nova-micro-v1:0", help="Bedrock LLM model name"
    ),
):
    populate(path, working_dir, llm_model_name)


@app.command("cli")
def cli_command(
    working_dir: str = typer.Option(..., help="Working directory for LightRAG"),
    llm_model_name: str = typer.Option(
        "amazon.nova-micro-v1:0", help="Bedrock LLM model name"
    ),
):
    main(working_dir, llm_model_name)


if __name__ == "__main__":
    app()
