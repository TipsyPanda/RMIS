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


def connect_gpt(engine, prompt, max_tokens, temperature, stop, client):
    """Fallback GPT prompt path (no LangChain)."""
    MAX_API_RETRY = 10
    for attempt in range(MAX_API_RETRY):
        time.sleep(2)
        try:
            if engine == "gpt-35-turbo-instruct":
                return client.completions.create(
                    model=engine,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                )
            else:
                return client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                )
        except Exception as e:
            print(f"API error (attempt {attempt+1}/{MAX_API_RETRY}): {e}")
    raise RuntimeError("Failed to get response from API after retries")


def post_process_response(response, db_path):
    """Extract SQL text and tag with DB ID."""
    content = response if isinstance(response, str) else response.choices[0].message.content
    db_id = os.path.basename(db_path).replace(".sqlite", "")
    return f"{content}\t----- bird -----\t{db_id}"


def decouple_question_schema(datasets, db_root_path):
    qs, dbs, evidence = [], [], []
    for entry in datasets:
        qs.append(entry["question"])
        dbs.append(os.path.join(db_root_path, entry["db_id"], f"{entry['db_id']}.sqlite"))
        evidence.append(entry.get("evidence"))
    return qs, dbs, evidence


def generate_sql_file(sql_results, output_path=None):
    """Write out SQL dict to JSON."""
    sql_results.sort(key=lambda x: x[1])
    out = {f"{i}": sql for i, (sql, _) in enumerate(sql_results)}
    if output_path:
        new_directory(os.path.dirname(output_path))
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
    return out


def worker_function(task):
    """
    task = (prompt, engine, client, db_path, question, idx)
    """
    prompt, engine, client, db_path, question, idx = task

    if args.use_knowledge == "Langchain":
        # 1. load DB
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        # 2. Azure Chat LLM
        llm = AzureChatOpenAI(
            azure_endpoint=API_BASE,
            azure_deployment=engine,
            openai_api_key=args.api_key,
            openai_api_version=API_VERSION,
            temperature=0.0,
        )

        sql_prompt = PromptTemplate(
            input_variables=["input", "table_info"],
            template="""
                You are an expert SQL generator.  Given the database schema and the user’s question,
                output *only* the SQL query in plain text.  Do **not** include any markdown formatting,
                code fences, or explanations.

                Schema:
                {table_info}

                Question:
                {input}
                """
        )

        chain = SQLDatabaseChain.from_llm(
            llm,
            db,
            prompt=sql_prompt,
            verbose=False,
            return_intermediate_steps=True,
        )
         
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                result = chain.invoke(question, return_only_outputs=True)
                # success, pull out everything
                sql_query  = result["intermediate_steps"][1]
                sql_result = result["intermediate_steps"][3]
                answer     = result["result"]
                sql        = post_process_response(sql_query, db_path)
                print(f"[LangChain]   #{idx} → {db_path} : {question}")
                break

            except OperationalError as e:
                # non-LLM SQL error; no retry
                print(f"[LangChain][SQL Error] #{idx} on {db_path}: {e}")
                sql = f"ERROR: {e}\t----- bird -----\t{os.path.basename(db_path).split('.')[0]}"
                break

            except (RateLimitError, OpenAIError) as e:
                # retry on rate-limit or other OpenAI errors
                if attempt < max_attempts:
                    print(f"[LangChain][API Error] #{idx} attempt {attempt}/{max_attempts}, retrying in 5s: {e}")
                    time.sleep(5)
                    continue
                else:
                    print(f"[LangChain][API Error] #{idx} on {db_path} after {max_attempts} attempts: {e}")
                    sql = f"ERROR: {e}\t----- bird -----\t{os.path.basename(db_path).split('.')[0]}"
                    break

            except Exception as e:
                # catch-all for anything else
                print(f"[LangChain][Unexpected Error] #{idx} on {db_path}: {e}")
                sql = f"ERROR: {e}\t----- bird -----\t{os.path.basename(db_path).split('.')[0]}"
                break

    else:
        resp = connect_gpt(
            engine,
            prompt,
            max_tokens=512,
            temperature=0.0,
            stop=["--", "\n\n", ";", "#"],
            client=client,
        )
        sql = post_process_response(resp, db_path)
        print(f"[Prompt]     #{idx} → {db_path} : {question}")

    return sql, idx


def init_client(api_key, api_version, engine):
    """Initialize low-level AzureOpenAI client."""
    return AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        base_url=f"{API_BASE}/openai/deployments/{engine}",
    )


def collect_response_from_gpt(
    db_paths, questions, api_key, engine,
    sql_dialect, num_threads=3, knowledge_list=None, output_path=None
):
    """Threaded execution; flush partial results to disk."""
    client = init_client(api_key, API_VERSION, engine)
    if knowledge_list is None:
        knowledge_list = [None] * len(questions)

    tasks = []
    for i, (dbp, q, kn) in enumerate(zip(db_paths, questions, knowledge_list)):
        prompt = None if args.use_knowledge == "Langchain" else generate_combined_prompts_one(
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
            # flush current partial results
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
        output_path=fname,  # enable incremental flush
    )

    print(
        f"Done: engine={args.engine}, mode={args.mode}, "
        f"knowledge={args.use_knowledge}, cot={args.chain_of_thought}"
    )
