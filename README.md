<h1 align="center">
  <br>
  <img src="assets/bigfoot.jpg" alt="logo" width="400"/>
  <br>
  Knowledge Graph with RAG Builder
  <br>
</h1>

<h4 align="center">This repo uses a stack of AI tools to create a RAG system with a knowledge graph that you can ask questions.The repo has a bigfoot sighting dataset to have fun with.
</h4>

<p align="center">
  <a href="#tech-stack">Tech Stack</a> •
  <a href="#how-it-works">How it works</a> •
  <a href="#setup">Setup</a> •
</p>

### Tech Stack

- [LightRag](https://github.com/HKUDS/LightRAG) - local graph creation
- [Ollama](https://ollama.com/) - local embeddings
- [AWS Bedrock](https://aws.amazon.com/bedrock/) - using the micro model for ultra cost effective model

### How it works

Basically there are 2 steps -

1. creating the graph
2. asking it questions

`graph.py` is a typer CLI with 2 commands

- populate - to populate your knowledge graph
- cli - to ask questions against your graph

### Setup

1. Any files you want to ingest into your graph need to be in the `dataset` dir. Currently `txt` and `csv` are supported. You could support `pdf`, `pptx`, etc by adding logic for `textract` to handle in the logic in the `populate` function in `graph.py`.
2. Define a .env file in the root with AWS credentials. `.env_example` can be used as the template.
3. The AWS account needs access to `amazon.nova-micro-v1:0` model.
4. Run `uv sync` to install deps
5. Pull embedding model with `ollama pull nomic-embed-text`
6. Ingest your data with the following command

```bash
uv run graph.py populate --path ./dataset/reports_last_5_yrs.csv --working-dir ./data_graph --llm-model-name amazon.nova-lite-v1:0
```

7. The knowledge graph will be in `data_graph` dir
8. Ask the knowledge graph questions

```bash
uv run graph.py cli --working-dir ./data_graph --llm-model-name amazon.nova-lite-v1:0
```

Some fun insights I got about Bigfoot were -

- they might be migrating
- witnesses are often credible
- easiest to find in olympic nat park at night time

You can change up the files in the `dataset` dir and build a knowledge graph on anything to talk to. I ran through multiple graph creations using the `micro` nova model and only managed to rack up 20 cents.
