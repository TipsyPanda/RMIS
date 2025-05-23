#!/usr/bin/env python3
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from tqdm import tqdm

from openai import AzureOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_openai.chat_models.azure import AzureChatOpenAI  
from langchain.prompts import PromptTemplate
from sqlalchemy.exc import OperationalError
from openai import RateLimitError, OpenAIError

from prompt import generate_combined_prompts_one
from prompt import generate_instruction_prompt

# Azure OpenAI settings
API_VERSION = "2024-12-01-preview"
API_BASE    = "https://rmisai.openai.azure.com/"


def new_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def post_process_response(content: str, db_path: str) -> str:
    db_id = os.path.basename(db_path).replace(".sqlite", "")
    return f"{content}\t----- bird -----\t{db_id}"


def langchain_prompt_template() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["input", "table_info"],
        template="""
You are an expert SQL generator. Given the database schema and the user’s question,
output *only* the SQL query in plain text. Do **not** include any markdown formatting,
code fences, or explanations.

Schema:
{table_info}

Question:
{input}
"""
    )


def connect_gpt(
    engine: str,
    client,
    db_path: str,
    question: str,
    prompt: str,
    use_langchain: bool,
    idx: int
) -> str:
    """
    Consolidated LLM call: LangChain or direct API, with retries and error handling.
    Returns processed SQL string.
    """
    if use_langchain:
        # LangChain path
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        llm = AzureChatOpenAI(
            azure_endpoint=API_BASE,
            azure_deployment=engine,
            openai_api_key=args.api_key,
            openai_api_version=API_VERSION,
            temperature=0.0,
        )
        chain = SQLDatabaseChain.from_llm(
            llm,
            db,
            prompt=langchain_prompt_template(),
            verbose=False,
            return_intermediate_steps=True,
        )
        attempts = 3
        for attempt in range(1, attempts + 1):
            try:
                result = chain.invoke(question, return_only_outputs=True)
                sql_query = result["intermediate_steps"][1]
                content = sql_query
                print(f"[LangChain]   #{idx} → {db_path} : {question}")
                return post_process_response(content, db_path)

            except OperationalError as e:
                print(f"[LangChain][SQL Error] #{idx} on {db_path}: {e}")
                return post_process_response(f"ERROR: {e}", db_path)

            except (RateLimitError, OpenAIError) as e:
                if attempt < attempts:
                    print(f"[LangChain][API Error] #{idx} attempt {attempt}/{attempts}, retrying in 5s: {e}")
                    time.sleep(5)
                    continue
                print(f"[LangChain][API Error] #{idx} on {db_path} after {attempts} attempts: {e}")
                return post_process_response(f"ERROR: {e}", db_path)

            except Exception as e:
                print(f"[LangChain][Unexpected Error] #{idx} on {db_path}: {e}")
                return post_process_response(f"ERROR: {e}", db_path)

    else:
        # direct Azure OpenAI chat completion
        MAX_API_RETRY = 10
        for attempt in range(1, MAX_API_RETRY + 1):
            try:
                if engine == "gpt-35-turbo-instruct":
                    resp = client.completions.create(
                        model=engine,
                        prompt=prompt,
                        max_tokens=512,
                        temperature=0.0,
                        stop=["--", "\n\n", ";", "#"],
                    )
                else:
                    resp = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=512,
                        temperature=0.0,
                        stop=["--", "\n\n", ";", "#"],
                    )
                content = resp.choices[0].message.content
                print(f"[Prompt]     #{idx} → {db_path} : {question}")
                return post_process_response(content, db_path)

            except Exception as e:
                print(f"[Prompt][API Error] #{idx} attempt {attempt}/{MAX_API_RETRY}: {e}")
                time.sleep(2)
                continue
        return post_process_response(f"ERROR: Failed after retries", db_path)


def worker_function(task):
    prompt, engine, client, db_path, question, idx = task
    use_langchain = args.use_knowledge == "Langchain"
    sql = connect_gpt(
        engine,
        client,
        db_path,
        question,
        prompt,
        use_langchain,
        idx,
    )
    return sql, idx


def init_client(api_key, api_version, engine):
    return AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        base_url=f"{API_BASE}/openai/deployments/{engine}",
    )


def collect_response_from_gpt(
    db_paths, questions, api_key, engine,
    sql_dialect, num_threads=3, knowledge_list=None, output_path=None
):
    client = init_client(api_key, API_VERSION, engine)
    if knowledge_list is None:
        knowledge_list = [None] * len(questions)

    tasks = []
    for i, (dbp, q, kn) in enumerate(zip(db_paths, questions, knowledge_list)):
        prompt = langchain_prompt_template() if args.use_knowledge == "Langchain" else generate_combined_prompts_one(
            db_path=dbp,
            question=q,
            sql_dialect=sql_dialect,
            knowledge=kn,
        )
        tasks.append((prompt, engine, client, dbp, q, i))

    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as exe:
        futures = {exe.submit(worker_function, t): t for t in tasks}
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            res = f.result()
            results.append(res)
            if output_path:
                generate_sql_file(results, output_path)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eval_path",        type=str, required=True)
    p.add_argument("--mode",             type=str, default="dev")
    p.add_argument("--use_knowledge",    type=str, default="False")
    p.add_argument("--db_root_path",     type=str, required=True)
    p.add_argument("--api_key",          type=str, required=True)
    p.add_argument("--engine",           type=str, required=True)
    p.add_argument("--data_output_path", type=str, required=True)
    p.add_argument("--chain_of_thought", type=str, default="False")
    p.add_argument("--num_processes",    type=int, default=3)
    p.add_argument("--sql_dialect",      type=str, default="SQLite")
    args = p.parse_args()

    data = json.load(open(args.eval_path))
    questions, db_paths, evidence = decouple_question_schema(data, args.db_root_path)

    cot_tag = "_cot" if args.chain_of_thought == "True" else ""
    fname = (
        f"{args.data_output_path}"
        f"predict_{args.mode}_{args.engine}{cot_tag}_{args.sql_dialect}.json"
    )

    responses = collect_response_from_gpt(
        db_paths,
        questions,
        args.api_key,
        args.engine,
        args.sql_dialect,
        args.num_processes,
        knowledge_list=evidence,
        output_path=fname,
    )

    print(
        f"Done: engine={args.engine}, mode={args.mode}, "
        f"knowledge={args.use_knowledge}, cot={args.chain_of_thought}"
    )