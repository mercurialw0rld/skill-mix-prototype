import os
import json
import random
import argparse
from pathlib import Path
import wikipedia
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage


def response_to_text(response) -> str:
    """Normalize LangChain model response content into plain text."""
    content = response.content
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text_attr = getattr(item, "text", None)
                if isinstance(text_attr, str):
                    parts.append(text_attr)
        return "\n".join(parts)

    return str(content)


def load_api_key_from_dotenv() -> str | None:
    """Reads gemini api"""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in {"GOOGLE_API_KEY", "GEMINI_API_KEY"}:
            continue
        parsed_value = value.strip().strip("\"'")
        if parsed_value:
            return parsed_value
    return None


def get_llm(api_key: str | None = None):
    api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or load_api_key_from_dotenv()
    if not api_key:
        raise ValueError("The API did not load.")
    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=api_key,
        temperature=0.7,
    )


def fetch_wikipedia(topic: str, sentences: int = 10) -> tuple[str, str]:
    """Fetch a Wikipedia summary for a given topic."""
    print(f"\n[1/4] Fetching Wikipedia article: '{topic}'...")
    results = wikipedia.search(topic)
    if not results:
        raise ValueError(f"No Wikipedia results found for '{topic}'")
    page = wikipedia.summary(results[0], sentences=sentences)
    title = results[0]
    print(f"      Found: '{title}' ({len(page.split())} words)")
    return title, page


def extract_skills(llm, title: str, excerpt: str) -> list[dict]:
    """Use LLM to extract named language/reasoning skills from the Wikipedia excerpt."""
    print("\n[2/4] Extracting named skills from the article...")

    prompt = f"""You are given a Wikipedia excerpt about "{title}".

Extract 6 to 8 distinct named language or reasoning skills that are either demonstrated
in this text or could naturally be applied when discussing this topic.

Use well-known skills that have Wikipedia entries, such as:
metaphor, statistical syllogism, modus ponens, red herring, self-serving bias,
spatial reasoning, folk physics, emotional self-regulation, loaded question,
analogy, appeal to authority, hasty generalization, etc.

Excerpt:
\"\"\"
{excerpt}
\"\"\"

Return ONLY a valid JSON array. No markdown, no explanation. Example format:
[
  {{"name": "metaphor", "definition": "a figure of speech that refers to one thing by mentioning another"}},
  {{"name": "spatial reasoning", "definition": "the capacity to reason about spatial relationships between objects"}}
]"""

    response = llm.invoke([
        SystemMessage(content="You extract structured data. Return only valid JSON arrays."),
        HumanMessage(content=prompt)
    ])

    raw = response_to_text(response).strip().replace("```json", "").replace("```", "").strip()
    skills = json.loads(raw)
    print(f"      Extracted {len(skills)} skills: {', '.join(s['name'] for s in skills)}")
    return skills


def generate_text(llm, topic: str, selected_skills: list[dict], k: int) -> str:
    """Generate text that combines k skills in the context of the topic."""
    print(f"\n[3/4] Generating text combining {k} skills in context of '{topic}'...")

    skill_defs = "\n".join(
        f"- {s['name']}: {s['definition']}" for s in selected_skills
    )

    prompt = f"""Write a short piece of text (at most {k + 1} sentences) in the context of "{topic}"
that naturally illustrates ALL of the following skills:

{skill_defs}

Important:
- Do NOT explicitly name the skills in your answer.
- The text should feel natural, not forced.
- Stay on topic.

Start your response with "Answer:" followed by the text.
Then write "Explanation:" followed by a brief explanation of how each skill appears."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response_to_text(response).strip()


def grade_output(llm, topic: str, selected_skills: list[dict], answer: str, k: int) -> dict:
    """Auto-grade the generated text using an LLM grader."""
    print("\n[4/4] Grading the output...")

    skill_defs = "\n".join(
        f"- {s['name']}: {s['definition']}" for s in selected_skills
    )
    skill_keys = [s['name'] for s in selected_skills]

    prompt = f"""Grade the following student answer. The student was asked to write at most {k + 1} sentences
in the context of "{topic}", illustrating these skills:

{skill_defs}

Student answer: "{answer}"

For each rubric item below, assign 1 (satisfied) or 0 (not satisfied).
Return ONLY valid JSON with integer values. No explanation.

{{
{chr(10).join(f'  "{name}": 0 or 1,' for name in skill_keys)}
  "on_topic": 0 or 1,
  "makes_sense": 0 or 1,
  "sentence_limit": 0 or 1
}}"""

    response = llm.invoke([
        SystemMessage(content="You are a strict grader. Return only valid JSON with 0 or 1 integer values."),
        HumanMessage(content=prompt)
    ])

    raw = response_to_text(response).strip().replace("```json", "").replace("```", "").strip()
    grades = json.loads(raw)
    return grades


def print_results(topic: str, selected_skills: list[dict], generation: str, grades: dict, k: int):
    """Console print the full pipeline results."""
    answer_part = ""
    explanation_part = ""

    if "Answer:" in generation:
        parts = generation.split("Explanation:")
        answer_part = parts[0].replace("Answer:", "").strip()
        explanation_part = parts[1].strip() if len(parts) > 1 else ""
    else:
        answer_part = generation

    max_score = k + 3
    total = sum(int(v) for v in grades.values())

    print("\n" + "=" * 60)
    print("Results: ")
    print("=" * 60)
    print(f"Topic       : {topic}")
    print(f"Skills (k={k}): {', '.join(s['name'] for s in selected_skills)}")
    print()
    print("Generated text:")
    print(f"  {answer_part}")
    if explanation_part:
        print()
        print("Explanation:")
        for line in explanation_part.split("\n"):
            print(f"  {line}")
    print()
    print("Grading:")
    for key, val in grades.items():
        status = "Correct" if int(val) == 1 else "Wrong"
        print(f"  {status} {key}: {val}")
    print()
    print(f"Score: {total} / {max_score}")
    print("=" * 60)


def run_pipeline(topic: str, k: int, api_key: str | None = None) -> dict:
    llm = get_llm(api_key)

    title, excerpt = fetch_wikipedia(topic)
    skills = extract_skills(llm, title, excerpt)

    if len(skills) < k:
        raise ValueError(f"Only {len(skills)} skills extracted, but k={k} was requested.")

    selected = random.sample(skills, k)
    print(f"      Selected: {', '.join(s['name'] for s in selected)}")

    generation = generate_text(llm, title, selected, k)
    answer = generation.split("Explanation:")[0].replace("Answer:", "").strip() if "Answer:" in generation else generation

    grades = grade_output(llm, title, selected, answer, k)
    return {
        "title": title,
        "topic": topic,
        "k": k,
        "selected_skills": selected,
        "generation": generation,
        "answer": answer,
        "grades": grades,
    }


def run(topic: str, k: int, api_key: str | None = None):
    result = run_pipeline(topic=topic, k=k, api_key=api_key)
    print_results(
        topic=result["title"],
        selected_skills=result["selected_skills"],
        generation=result["generation"],
        grades=result["grades"],
        k=result["k"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SkillMix: named skill assay from Wikipedia")
    parser.add_argument("--topic", type=str, default="beekeeping", help="Wikipedia topic to use")
    parser.add_argument("--k", type=int, default=2, help="Number of skills to combine (default: 2)")
    parser.add_argument("--api-key", type=str, default=None, help="Google API key (optional if GOOGLE_API_KEY is set)")
    args = parser.parse_args()

    run(topic=args.topic, k=args.k, api_key=args.api_key)
