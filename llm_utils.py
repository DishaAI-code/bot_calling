import os
import time
from openai import OpenAI
from langfuse import get_client
from dotenv import load_dotenv

load_dotenv()

# Initialize clients
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
langfuse = get_client()



def generate_response(prompt_text, trace_id=None, **metadata):
    # start a nested observation (child span)
    if trace_id:
        child_span = langfuse.start_span(
            trace_id=trace_id,
            name="llm_generation_child",
            input=prompt_text,
            metadata=metadata
        )
    else:
        child_span = None

    # start a separate generation (standalone)
    standalone_gen = langfuse.start_generation(
        name="llm_generation_standalone",
        model="gpt-3.5-turbo",
        input=prompt_text,
        metadata=metadata
    )

    try:
        start = time.time()
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.7,
            max_tokens=500
        )
        duration = time.time() - start
        response = completion.choices[0].message.content.strip()

        meta = {
            "usage_tokens": completion.usage.total_tokens if completion.usage else None,
            "response_length": len(response),
            "processing_time_seconds": duration,
            **metadata
        }

        if child_span:
            child_span.update(output=response, metadata=meta)
            child_span.end()

        standalone_gen.update(output=response, metadata=meta)
        standalone_gen.end()

        return response, duration

    except Exception as e:
        if child_span:
            child_span.update(output={"error": str(e)})
            child_span.end()
        standalone_gen.update(output={"error": str(e)})
        standalone_gen.end()
        raise