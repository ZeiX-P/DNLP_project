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

ZS_NAIVE_PROMPT_STR_FOR_QWEN = {
    "cnn": """<|im_start|>system
        You are a text summarization system. Output only the summary.
        <|im_end|>
        <|im_start|>user
        Here is a news article:
        <text>
        Write a summary for the article in 2-3 sentences.
        <|im_end|>
        <|im_start|>assistant""",
    "arxiv": """<|im_start|>system
        You are a text summarization system. Output only the summary.
        <|im_end|>
        <|im_start|>user
        Here is a research paper:
        <text>
        Write an abstract of about 3 sentences.
        <|im_end|>
        <|im_start|>assistant"""
}

ZS_NAIVE_PROMPT_STR_FOR_LLAMA = {
    "cnn": (
        "<|begin_of_text|>\n"
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are a text summarization system.\n"
        "Output only the summary text.\n"
        "Do not include introductions, labels, explanations, apologies, or multiple versions.\n"
        "Write exactly one summary and stop.\n"
        "<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        "Here is a news article:\n"
        "<text>\n\n"
        "Please write a short summary for the article in 1–2 sentences.\n"
        "<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    ),
    "arxiv": (
        "<|begin_of_text|>\n"
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are a text summarization system.\n"
        "Output only the abstract text.\n"
        "Do not include introductions, labels, explanations, apologies, or multiple versions.\n"
        "Write exactly one abstract and stop.\n"
        "<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        "Here is a research paper:\n"
        "<text>\n\n"
        "Please write a short abstract of about 3 sentences.\n"
        "<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
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

ZS_KEYWORD_PROMPT_STR_FOR_QWEN = {
    "cnn": """<|im_start|>system
        You are a text summarization system. Output only the summary.
        <|im_end|>
        <|im_start|>user
        Here is a news article:
        <text>
        Write a summary for the article in 2-3 sentences.
        Consider include the following information: <keywords>.
        <|im_end|>
        <|im_start|>assistant""",
    "arxiv": """<|im_start|>system
        You are a text summarization system. Output only the summary.
        <|im_end|>
        <|im_start|>user
        Here is a research paper:
        <text>
        Write an abstract of about 3 sentences.
        Consider include the following information: <keywords>.
        <|im_end|>
        <|im_start|>assistant"""
}

ZS_KEYWORD_PROMPT_STR_FOR_LLAMA = {
    "cnn": (
        "<|begin_of_text|>\n"
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are a text summarization system.\n"
        "Output only the summary text.\n"
        "Do not include introductions, labels, explanations, apologies, or multiple versions.\n"
        "Write exactly one summary and stop.\n"
        "<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        "Here is a news article:\n"
        "<text>\n\n"
        "Please write a short summary for the article in 1–2 sentences. "
        "Make sure to include the following information: <keywords>\n"
        "<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    ),
    "arxiv": (
        "<|begin_of_text|>\n"
        "<|start_header_id|>system<|end_header_id|>\n"
        "You are a text summarization system.\n"
        "Output only the abstract text.\n"
        "Do not include introductions, labels, explanations, apologies, or multiple versions.\n"
        "Write exactly one abstract and stop.\n"
        "<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        "Here is a research paper:\n"
        "<text>\n\n"
        "Please write a short abstract of about 3 sentences. "
        "Make sure to include the following information: <keywords>\n"
        "<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
}

ZS_COT_KEYWORD_PROMPT_STR_FOR_MISTRAL = {
    "cnn": """<s>[INST]
        You are an expert news analyst and summarizer trained to produce CNN/DailyMail-style highlights.
        You will be given a news article and a list of key phrases extracted from it.
        Your task has two phases in ONE single response:

        PHASE 1 – ANALYSIS (SHORT)
        Carefully read the article and briefly answer the following questions:
        1. Who are the main people, organizations or entities mentioned in this article?
        2. Where and when does the main event (or events) take place?
        3. What are the main events described in the article?
        4. What is the immediate outcome or development of these events?

        Write the answers as a numbered list. Each item should correspond to ONE distinct event. Be factual and concise.

        PHASE 2 – FINAL SUMMARY
        Using ONLY the information from the article and your analysis above, write a concise news-style summary.

        Strict requirements for the FINAL SUMMARY:
        - Write one sentence per key fact identified in the analysis.
        - Each sentence should contain approximately 12–20 words.
        - The total summary length will naturally vary depending on how many events were found.
        - Each sentence should describe a key fact or event. 
        - Focus on what happened, who was involved, and the immediate outcome.
        - It MUST be neutral and factual.
        - It MUST NOT add unsupported information.
        - Try to naturally include some of the key phrases, but do NOT force all of them.

        Article:
        <text>
        
        Key phrases:
        <keywords>
        
        Now produce your answer in the following format:

        Analysis:
        1) ...
        2) ...
        3) ...
        
        Final summary:
        <1 sentence per event, about 12–20 words each>[/INST]""",
    "arxiv": """<s>[INST]
        You are an expert scientific paper summarizer.
        You will be given a research paper and a list of key phrases extracted from it.
        Your task has two phases in ONE single response:
        
        PHASE 1 – ANALYSIS (SHORT)
        Carefully read the paper and briefly answer the following questions:
        1. What is the main research problem or question addressed in this paper?
        2. What methods or approaches are used to tackle the problem?
        3. What are the key results or findings of the study?
        4. What is the main contribution or novelty of the paper?
        5. Are there any limitations or open questions mentioned?
        
        Write the answers as a short numbered list.

        PHASE 2 – FINAL SUMMARY
        Using ONLY the information from the paper and your analysis above, write a concise abstract-style summary.
        
        Strict requirements for the FINAL ABSTRACT:
        - Write one or more coherent sentences per key item identified in the analysis.
        - The total abstract length should naturally vary depending on the number and complexity of the key items.
        - Typically, abstracts are 150–250 words, but can extend up to 400 words for complex papers or reduce to 60 words for simple papers.
        - Sentences should be medium length (approximately 20–40 words) and fully coherent.
        - Maintain a neutral, factual, scientific tone.
        - It MUST NOT add unsupported information.
        - Try to naturally include some of the key phrases, but do NOT force all of them.
        
        Paper:
        <text>
        
        Key phrases:
        <keywords>

        Produce your response exactly in the following format:
        
        Analysis:
        1) ...
        2) ...
        3) ...
        
        Final summary:
        <abstract, typically 3-6 sentences>[/INST]"""
}

ZS_COT_KEYWORD_PROMPT_STR_FOR_QWEN = {
    "cnn": """<|im_start|>system
        You are an expert news analyst and summarizer trained to produce CNN/DailyMail-style highlights. Output only the summary.
        <|im_end|>
        <|im_start|>user
        You will be given a news article and a list of key phrases extracted from it.
        Your task has two phases in ONE single response:

        PHASE 1 – ANALYSIS (SHORT)
        Carefully read the article and briefly answer the following questions:
        1. Who are the main people, organizations or entities mentioned in this article?
        2. Where and when does the main event (or events) take place?
        3. What are the main events described in the article?
        4. What is the immediate outcome or development of these events?

        Write the answers as a numbered list. Each item should correspond to ONE distinct event. Be factual and concise.

        PHASE 2 – FINAL SUMMARY
        Using ONLY the information from the article and your analysis above, write a concise news-style summary.

        Strict requirements for the FINAL SUMMARY:
        - Write one sentence per key fact identified in the analysis.
        - Each sentence should contain approximately 12–20 words.
        - The total summary length will naturally vary depending on how many events were found.
        - Each sentence should describe a key fact or event. 
        - Focus on what happened, who was involved, and the immediate outcome.
        - It MUST be neutral and factual.
        - It MUST NOT add unsupported information.
        - Try to naturally include some of the key phrases, but do NOT force all of them.

        Article:
        <text>
        
        Key phrases:
        <keywords>
        
        Now produce your answer in the following format:

        Analysis:
        1) ...
        2) ...
        3) ...
        
        Final summary:
        <1 sentence per event, about 12–20 words each>
        <|im_end|>
        <|im_start|>assistant""",
    "arxiv": """<|im_start|>system
        You are an expert scientific paper summarizer. Output only the summary.
        <|im_end|>
        <|im_start|>user
        You will be given a research paper and a list of key phrases extracted from it.
        Your task has two phases in ONE single response:
        
        PHASE 1 – ANALYSIS (SHORT)
        Carefully read the paper and briefly answer the following questions:
        1. What is the main research problem or question addressed in this paper?
        2. What methods or approaches are used to tackle the problem?
        3. What are the key results or findings of the study?
        4. What is the main contribution or novelty of the paper?
        5. Are there any limitations or open questions mentioned?
        
        Write the answers as a short numbered list.

        PHASE 2 – FINAL SUMMARY
        Using ONLY the information from the paper and your analysis above, write a concise abstract-style summary.
        
        Strict requirements for the FINAL ABSTRACT:
        - Write one or more coherent sentences per key item identified in the analysis.
        - The total abstract length should naturally vary depending on the number and complexity of the key items.
        - Typically, abstracts are 150–250 words, but can extend up to 400 words for complex papers or reduce to 60 words for simple papers.
        - Sentences should be medium length (approximately 20–40 words) and fully coherent.
        - Maintain a neutral, factual, scientific tone.
        - It MUST NOT add unsupported information.
        - Try to naturally include some of the key phrases, but do NOT force all of them.
        
        Paper:
        <text>
        
        Key phrases:
        <keywords>

        Produce your response exactly in the following format:
        
        Analysis:
        1) ...
        2) ...
        3) ...
        
        Final summary:
        <abstract, typically 3-6 sentences>
        <|im_end|>
        <|im_start|>assistant"""
}

ZS_COT_KEYWORD_PROMPT_STR_FOR_LLAMA = {
    "cnn": """<|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        You are an expert news analyst and summarizer trained to produce CNN/DailyMail-style highlights.
        <|end_of_text|>

        <|start_header_id|>user<|end_header_id|>
        You will be given a news article and a list of key phrases extracted from it.
        Your task has two phases in ONE single response:

        PHASE 1 – ANALYSIS (SHORT)
        Carefully read the article and briefly answer the following questions:
        1. Who are the main people, organizations or entities mentioned in this article?
        2. Where and when does the main event (or events) take place?
        3. What are the main events described in the article?
        4. What is the immediate outcome or development of these events?

        Write the answers as a numbered list. Each item should correspond to ONE distinct event. Be factual and concise.

        PHASE 2 – FINAL SUMMARY
        Using ONLY the information from the article and your analysis above, write a concise news-style summary.

        Strict requirements for the FINAL SUMMARY:
        - Write one sentence per key fact identified in the analysis.
        - Each sentence should contain approximately 12–20 words.
        - The total summary length will naturally vary depending on how many events were found.
        - Each sentence should describe a key fact or event. 
        - Focus on what happened, who was involved, and the immediate outcome.
        - It MUST be neutral and factual.
        - It MUST NOT add unsupported information.
        - Try to naturally include some of the key phrases, but do NOT force all of them.

        Article:
        <text>
        
        Key phrases:
        <keywords>
        
        Now produce your answer in the following format:

        Analysis:
        1) ...
        2) ...
        3) ...
        
        Final summary:
        <1 sentence per event, about 12–20 words each>
        <|end_of_text|>
        <|start_header_id|>assistant<|end_header_id|>""",
    "arxiv": """<|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        You are an expert scientific paper summarizer.
        <|end_of_text|>
        
        <|start_header_id|>user<|end_header_id|>
        You will be given a research paper and a list of key phrases extracted from it.
        Your task has two phases in ONE single response:
        
        PHASE 1 – ANALYSIS (SHORT)
        Carefully read the paper and briefly answer the following questions:
        1. What is the main research problem or question addressed in this paper?
        2. What methods or approaches are used to tackle the problem?
        3. What are the key results or findings of the study?
        4. What is the main contribution or novelty of the paper?
        5. Are there any limitations or open questions mentioned?
        
        Write the answers as a short numbered list.

        PHASE 2 – FINAL SUMMARY
        Using ONLY the information from the paper and your analysis above, write a concise abstract-style summary.
        
        Strict requirements for the FINAL ABSTRACT:
        - Write one or more coherent sentences per key item identified in the analysis.
        - The total abstract length should naturally vary depending on the number and complexity of the key items.
        - Typically, abstracts are 150–250 words, but can extend up to 400 words for complex papers or reduce to 60 words for simple papers.
        - Sentences should be medium length (approximately 20–40 words) and fully coherent.
        - Maintain a neutral, factual, scientific tone.
        - It MUST NOT add unsupported information.
        - Try to naturally include some of the key phrases, but do NOT force all of them.
        
        Paper:
        <text>
        
        Key phrases:
        <keywords>

        Produce your response exactly in the following format:
        
        Analysis:
        1) ...
        2) ...
        3) ...
        
        Final summary:
        <abstract, typically 3-6 sentences>
        <|end_of_text|>
        <|start_header_id|>assistant<|end_header_id|>
        """
}
