ZS_NAIVE_PROMPT_STR_FOR_MISTRAL = {
    "cnn": "<s>[INST]Here is a news article:\n<text>\n\n"
    "Please write a short summary for the article in 1-2 sentences.[/INST]",
    "arxiv": "<s>[INST]Here is a research paper:\n<text>\n\n"
    "Please write a short abstract in about 3 sentences.[/INST]",
    "pubmed": "<s>[INST]Here is a research paper:\n<text>\n\n"
    "Please write a short abstract in about 3 sentences.[/INST]",
    "samsum": "<s>[INST]Here is a conversation:\n<text>\n\n" "Please write a short 1 sentence summary. [/INST]",
    "meetingbank": "<s>[INST]Here is a conversation:\n<text>\n\n" "Please write a 2-3 sentence summary. [/INST]",
}

ZS_NAIVE_PROMPT_STR_FOR_CLAUDE = {
    "cnn": "Here is a news article:\n<text>\n\nPlease write a summary for the article in 2-3 sentences.",
    "arxiv": "Here is a research paper:\n<text>\n\nPlease write a comprehensive paper abstract section.",
    "pubmed": "Here is a research paper:\n<text>\n\nPlease write a paper abstract in around 5 sentences.",
    "samsum": "Here is a conversation:\n<text>\n\nPlease write a very short 1 sentence summary.",
    "meetingbank": "Here is a conversation:\n<text>\n\nPlease write a summary in about 5 sentences.",
}

ZS_KEYWORD_PROMPT_STR_FOR_MISTRAL = {
    "cnn": "<s>[INST]Here is a news article:\n<text>\n\n"
    "Please write a short summary for the article in 1-2 sentences. "
    "Consider include the following information: <keywords>[/INST]",
    "arxiv": "<s>[INST]Here is a research paper:\n<text>\n\n"
    "Please write a short abstract in about 3 sentences. "
    "Consider include the following information: <keywords>[/INST]",
    "pubmed": "<s>[INST]Here is a research paper:\n<text>\n\n"
    "Please write a short abstract in about 3 sentences. "
    "Consider include the following information: <keywords>[/INST]",
    "samsum": "<s>[INST]Here is a conversation:\n<text>\n\n"
    "Please write a short 1 sentence summary. "
    "Consider include the following information: <keywords>[/INST]",
    "meetingbank": "<s>[INST]Here is a conversation:\n<text>\n\n"
    "Please write a 2-3 sentence summary. "
    "Consider include the following information: <keywords>[/INST]",
}


ZS_KEYWORD_PROMPT_STR_FOR_CLAUDE = {
    "cnn": "Here is a news article:\n<text>\n\n"
    "Please write a summary for the article in 2-3 sentences. "
    "Consider include the following information: <keywords>.",
    "arxiv": "Here is a research paper:\n<text>\n\n"
    "Please write a comprehensive paper abstract section."
    "Consider include the following information: <keywords>.",
    "pubmed": "Here is a research paper:\n<text>\n\n"
    "Please write a paper abstract in around 5 sentences. "
    "Consider include the following information: <keywords>.",
    "samsum": "Here is a conversation:\n<text>\n\n"
    "Please write a very short 1 sentence summary. "
    "Consider include the following information: <keywords>",
    "meetingbank": "Here is a conversation:\n<text>\n\n"
    "Please write a summary in about 5 sentences. "
    "Consider include the following information: <keywords>",
}

ZS_COT_PROMPT = {
"cnn": """<s>[INST]
You are an expert news summarizer.

You will be given:
- a news article
- a list of key phrases extracted from the article

Your task is to write a short, faithful news summary.

STRICT RULES (you MUST follow ALL of them):
- The summary MUST be based ONLY on the article.
- The summary MUST be EXACTLY 2 sentences.
- The summary MUST be concise and no longer than 60 words in total.
- DO NOT explain your reasoning.
- DO NOT output analysis, steps, or bullet points.
- DO NOT add any information that is not present in the article.
- Try to naturally include some of the key phrases, but do NOT force all of them.

IMPORTANT:
If you produce more than 2 sentences, or any extra explanation, your answer will be considered WRONG.

Article:
<text>

Key Phrases:
<keywords>

Write ONLY the final summary (no titles, no labels, no commentary).
Summary:
[/INST]"""
}